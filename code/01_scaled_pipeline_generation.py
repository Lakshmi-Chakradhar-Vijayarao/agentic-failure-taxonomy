"""
Paper 3 — Scaled + cross-model self-refinement pipeline generation.

WHY THIS EXISTS
---------------
The original FAIL-CHAIN run (fail-chain/experiments/01_pipeline_generation.py)
produced 600 pipelines on Qwen 2.5 3B / TruthfulQA only. Two things block
Paper 3 from being defensible:

1. LATENT (correct->correct->wrong) had only 3 samples at n=600 (~0.5% rate).
   No statistic computed on 3 samples survives review. TruthfulQA itself only
   has 817 questions total, so "just use more TruthfulQA" caps out fast --
   this script folds in NQ-open (thousands of available examples, same
   protocol HaRP already used to blend datasets) to reach a real N.
2. Everything so far is Qwen 2.5 3B only. A reviewer's first question is
   "is this a Qwen quirk?" -- this script accepts --model to run the exact
   same pipeline on a second architecture (e.g. Llama-3.1-8B-Instruct or
   Mistral-7B-Instruct-v0.3) for cross-model replication.

This is a DIRECT extension of fail-chain/src/pipeline.py (run_pipeline /
run_pipeline_noctx are reused unchanged) -- only the dataset construction,
scale, and model-selection are new. Output schema matches the original
pipeline_data.json exactly, with a "dataset_source" and "model_name" field
added so downstream analysis (fail-chain's failure_modes.py, cascade_predict.py)
works without modification.

COMPUTE NOTE (read before running)
-----------------------------------
The original 600-pipeline Qwen run took ~4 hours on a Kaggle T4. This script
scales that linearly with N_PIPELINE and adds a second full model, so the
default config below (N=2000, one model) is already ~13 GPU-hours, and
running BOTH models at N=2000 each is ~26 GPU-hours. This MUST run on Kaggle
(or other GPU) with the checkpoint/resume logic below -- it will not run on
an 8GB Apple Silicon laptop with no CUDA. Pick N_PIPELINE deliberately before
committing GPU time; see the three presets below.

Usage (on Kaggle, T4 GPU):
    python 01_scaled_pipeline_generation.py --model qwen --n 2000
    python 01_scaled_pipeline_generation.py --model llama --n 2000
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
# Fixed post-final-audit: this is a genuine cross-repo dependency (not
# fixable by ROOT-relative pathing alone), disclosed above -- this script
# is a direct extension of fail-chain/src/pipeline.py, a separate sibling
# project, not vendored into this repo. Resolution order: FAIL_CHAIN_ROOT
# env var, then the sibling-of-papers-2026 default this project's own
# disk layout implies, with a clear error if neither exists (rather than
# a silent hardcoded personal path that breaks on any other machine).
FAIL_CHAIN_ROOT = Path(os.environ.get("FAIL_CHAIN_ROOT", ROOT.parent.parent / "fail-chain"))
if not FAIL_CHAIN_ROOT.exists():
    raise FileNotFoundError(
        f"fail-chain sibling project not found at {FAIL_CHAIN_ROOT}. "
        "Set the FAIL_CHAIN_ROOT environment variable to its location "
        "(this script imports src.pipeline / src.failure_modes from it "
        "and does not vendor a copy)."
    )
sys.path.insert(0, str(FAIL_CHAIN_ROOT))

from src.pipeline import run_pipeline, run_pipeline_noctx  # noqa: E402
from src.failure_modes import classify_failure_mode, is_boundary_sample  # noqa: E402

MODEL_REGISTRY = {
    "qwen":   {"hf_id": "Qwen/Qwen2.5-3B-Instruct",         "target_layer": 32, "n_layers": 36},
    "llama":  {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "target_layer": 28, "n_layers": 32},
    "mistral": {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "target_layer": 28, "n_layers": 32},
}
# target_layer chosen to match ~89% relative depth used throughout the
# mech-int/harp/fail-chain arc (Qwen: 32/36; Llama/Mistral 32-layer models: 28/32).

ROUGE_THRESHOLD = 0.3
MAX_NEW_TOKENS = 80
SAVE_EVERY = 100
N_ABLATION_FRACTION = 0.10   # context-free ablation subset size, as fraction of N


def compute_rouge_l(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def build_question_pool(n_target: int):
    """TruthfulQA (all 817, factual/adversarial) + NQ-open (sampled) to reach n_target.
    Same blending precedent as HaRP's 426 TruthfulQA + 274 NQ combination."""
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    tqa_items = [{"question": x["question"], "best_answer": x["best_answer"], "source": "truthfulqa"}
                 for x in tqa]

    n_nq_needed = max(0, n_target - len(tqa_items))
    pool = list(tqa_items)
    if n_nq_needed > 0:
        nq = load_dataset("nq_open", split="train")
        nq = nq.shuffle(seed=42).select(range(min(n_nq_needed, len(nq))))
        for x in nq:
            answer = x["answer"][0] if isinstance(x["answer"], list) and x["answer"] else str(x["answer"])
            pool.append({"question": x["question"], "best_answer": answer, "source": "nq_open"})

    print(f"Question pool: {len(pool)} total "
          f"({len(tqa_items)} TruthfulQA + {len(pool) - len(tqa_items)} NQ-open)")
    return pool[:n_target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    ap.add_argument("--n", type=int, default=2000, help="total pipeline samples")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint if present")
    args = ap.parse_args()

    # Fixed post-final-audit: no seeding was set for this process at all.
    # This covers everything under this script's own control; full
    # determinism additionally depends on fail-chain's run_pipeline /
    # run_pipeline_noctx generation settings (do_sample/temperature),
    # which live in that sibling project and are not visible or owned
    # from here -- disclosed rather than silently assumed deterministic.
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = MODEL_REGISTRY[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model} ({cfg['hf_id']}) | N={args.n} | "
          f"target_layer={cfg['target_layer']}")

    out_dir = Path(__file__).resolve().parent.parent / "results" / args.model
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "hidden_states").mkdir(parents=True, exist_ok=True)

    pool = build_question_pool(args.n)
    n_ablation = max(50, int(len(pool) * N_ABLATION_FRACTION))

    print(f"\nLoading {cfg['hf_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    ckpt_path = out_dir / "logs" / "pipeline_data_checkpoint.json"
    pipeline_data = []
    start_i = 0
    if args.resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            pipeline_data = json.load(f)
        start_i = len(pipeline_data)
        print(f"Resumed from checkpoint: {start_i} samples already done.")

    skipped = 0
    for i, sample in enumerate(tqdm(pool[start_i:], desc="Pipeline", initial=start_i, total=len(pool))):
        idx = start_i + i
        question, best_answer = sample["question"], sample["best_answer"]
        try:
            result = run_pipeline(model, tokenizer, question,
                                   target_layer=cfg["target_layer"],
                                   max_new_tokens=MAX_NEW_TOKENS, device=device)
            rouge_s1 = compute_rouge_l(result["step1_output"], best_answer)
            rouge_s2 = compute_rouge_l(result["step2_output"], best_answer)
            rouge_s3 = compute_rouge_l(result["step3_output"], best_answer)
            failure_mode = classify_failure_mode(rouge_s1, rouge_s2, rouge_s3, ROUGE_THRESHOLD)
            boundary = is_boundary_sample(rouge_s1, rouge_s2, rouge_s3, ROUGE_THRESHOLD)

            pipeline_data.append({
                "id": idx, "question": question, "best_answer": best_answer,
                "dataset_source": sample["source"], "model_name": args.model,
                "step1_output": result["step1_output"], "step2_output": result["step2_output"],
                "step3_output": result["step3_output"],
                "h_s1": result["h_s1"].tolist(), "h_s2": result["h_s2"].tolist(),
                "h_s3": result["h_s3"].tolist(),
                "rouge_s1": rouge_s1, "rouge_s2": rouge_s2, "rouge_s3": rouge_s3,
                "label_s1": int(rouge_s1 < ROUGE_THRESHOLD),
                "label_s2": int(rouge_s2 < ROUGE_THRESHOLD),
                "label_s3": int(rouge_s3 < ROUGE_THRESHOLD),
                "failure_mode": failure_mode, "is_boundary": boundary,
            })
        except Exception as e:
            print(f"\nSample {idx} failed: {e}")
            skipped += 1
            continue

        if (idx + 1) % SAVE_EVERY == 0:
            with open(ckpt_path, "w") as f:
                json.dump(pipeline_data, f)
            print(f"\n[{idx + 1}/{len(pool)}] checkpoint saved (skipped so far: {skipped})")

    out_path = out_dir / "logs" / "pipeline_data.json"
    with open(out_path, "w") as f:
        json.dump(pipeline_data, f, indent=2)

    hs = {
        "h_s1": np.array([d["h_s1"] for d in pipeline_data]),
        "h_s2": np.array([d["h_s2"] for d in pipeline_data]),
        "h_s3": np.array([d["h_s3"] for d in pipeline_data]),
        "labels_s1": np.array([d["label_s1"] for d in pipeline_data]),
        "labels_s2": np.array([d["label_s2"] for d in pipeline_data]),
        "labels_s3": np.array([d["label_s3"] for d in pipeline_data]),
    }
    np.savez(out_dir / "hidden_states" / "pipeline_hidden_states.npz", **hs)

    modes = Counter(d["failure_mode"] for d in pipeline_data)
    total = len(pipeline_data)
    print("\n" + "=" * 60)
    print(f"SCALED PIPELINE COMPLETE — model={args.model} N={total} (skipped {skipped})")
    for mode, count in modes.most_common():
        print(f"  {mode:<16}: {count:>5} ({count/total*100:.2f}%)")
    latent_n = modes.get("LATENT", 0)
    print(f"\nLATENT count at this N: {latent_n} "
          f"({'STILL TOO FEW for stats' if latent_n < 20 else 'usable for basic stats'})")
    print("=" * 60)

    # ── Context-free ablation subset (same P4 test as original) ──────────────
    print(f"\nRunning context-free ablation on first {n_ablation} questions...")
    ablation_data, ablation_skipped = [], 0
    for i, sample in enumerate(tqdm(pool[:n_ablation], desc="Ablation (noctx)")):
        question, best_answer = sample["question"], sample["best_answer"]
        try:
            result = run_pipeline_noctx(model, tokenizer, question,
                                         target_layer=cfg["target_layer"],
                                         max_new_tokens=MAX_NEW_TOKENS, device=device)
            rouge_s1 = compute_rouge_l(result["step1_output"], best_answer)
            rouge_s2 = compute_rouge_l(result["step2_output"], best_answer)
            rouge_s3 = compute_rouge_l(result["step3_output"], best_answer)
            failure_mode = classify_failure_mode(rouge_s1, rouge_s2, rouge_s3, ROUGE_THRESHOLD)
            boundary = is_boundary_sample(rouge_s1, rouge_s2, rouge_s3, ROUGE_THRESHOLD)
            ablation_data.append({
                "id": i, "question": question, "best_answer": best_answer,
                "dataset_source": sample["source"], "model_name": args.model,
                "step1_output": result["step1_output"], "step2_output": result["step2_output"],
                "step3_output": result["step3_output"],
                "h_s1": result["h_s1"].tolist(), "h_s2": result["h_s2"].tolist(),
                "h_s3": result["h_s3"].tolist(),
                "rouge_s1": rouge_s1, "rouge_s2": rouge_s2, "rouge_s3": rouge_s3,
                "label_s1": int(rouge_s1 < ROUGE_THRESHOLD),
                "label_s2": int(rouge_s2 < ROUGE_THRESHOLD),
                "label_s3": int(rouge_s3 < ROUGE_THRESHOLD),
                "failure_mode": failure_mode, "is_boundary": boundary,
                "pipeline_type": "noctx",
            })
        except Exception as e:
            print(f"\nAblation sample {i} failed: {e}")
            ablation_skipped += 1
            continue

    abl_path = out_dir / "logs" / "ablation_data.json"
    with open(abl_path, "w") as f:
        json.dump(ablation_data, f, indent=2)

    std_matched = [d for d in pipeline_data if d["id"] < n_ablation]
    std_amp = sum(1 for d in std_matched if d["failure_mode"] == "AMPLIFIED") / max(len(std_matched), 1) * 100
    noctx_amp = sum(1 for d in ablation_data if d["failure_mode"] == "AMPLIFIED") / max(len(ablation_data), 1) * 100
    print(f"\nAMPLIFIED — Standard: {std_amp:.2f}%  |  Context-free: {noctx_amp:.2f}%  "
          f"Drop: {std_amp - noctx_amp:+.2f}pp")
    print(f"\nSaved: {out_path}")
    print(f"Saved: {abl_path}")


if __name__ == "__main__":
    main()
