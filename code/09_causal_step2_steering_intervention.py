"""
Paper 3 -- final-audit fix, responding directly to the elite-reviewer critique
that the paper's "early-warning" claim rests entirely on a PASSIVE probe
(05_full_vector_probe.py) with no causal write-side test, unlike Papers 1/4's
causal-patching experiments in this project family.

WHY THIS EXPERIMENT, SPECIFICALLY
----------------------------------
fail-chain/src/pipeline.py's run_pipeline() calls each step as a SEPARATE
model.generate() on the growing TEXT context -- step 2 and step 3 never see
step 1's raw hidden state, only step 1's generated text. So "patch h_s1
between two pipelines" doesn't correspond to anything the real pipeline
does. The mechanistically meaningful causal question is instead: does
STEERING the model's residual stream, during step-2 generation, along the
direction the passive probe already found predictive of eventual step-3
outcome, actually change what step 2 generates -- and does that change
propagate to step 3's correctness? This is the same found-vs-random
additive-steering design used in code/01_ffn_causal_patch.py (Paper 1) and
geom-proof's Exp 19/21, applied here to the cascade-prediction direction
instead of a hallucination direction.

PROTOCOL
--------
1. Restrict to pipelines already correct at step 1 (relabel_label_s1==1) --
   the exact population code/06_difficulty_confound_and_significance.py's
   passive-probe decisive test uses, so this causal test is directly
   comparable to that passive result.
2. TRAIN split (fit the direction): difference-of-means between h_s1 of
   eventually-still-correct (relabel_label_s3==1, SELF_HEAL) vs.
   eventually-wrong (relabel_label_s3==0, LATENT) pipelines. This direction
   points from "cascades to wrong" toward "stays correct."
3. TEST split (held out from direction-fitting): for each pipeline, rebuild
   the exact step-2 prompt from the ACTUAL recorded step-1 answer text
   (format_step2, fail-chain's own template) and regenerate step 2 with an
   additive steering hook at layer 32 injecting either (a) the found
   direction or (b) a random direction of the same norm, at each of several
   alpha strengths. Step 3 is then built from the ORIGINAL step-1 text and
   the newly-generated (steered) step-2 text, and generated normally (no
   further intervention) -- isolating whether the step-2 steering alone
   propagates to step-3 correctness.
4. Step-3 outputs are rescored with the identical corrected metric already
   used throughout this paper (SQuAD EM/F1 for NQ-open, Jaccard>0.12 for
   TruthfulQA -- functions copied verbatim from 02_semantic_relabel.py, not
   reimplemented differently).
5. McNemar test on paired (found, random) step-3 correctness outcomes.

This is a genuine causal test, not a passive probe -- if found-direction
steering measurably improves step-3 correctness over random-direction
steering, that is a positive, deployable-adjacent result (a mid-generation
corrective nudge). If not, it is a stronger, causal-test-based null than
the passive full-vector probe result already reported.
"""
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "qwen" / "results" / "qwen" / "logs" / "pipeline_data_relabeled.json"
OUT_PATH = ROOT / "results" / "causal_step2_steering_intervention.json"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TARGET_LAYER = 32
RANDOM_STATE = 42
N_TEST = 50
ALPHAS = [15.0, 30.0]
MAX_NEW_TOKENS = 80

STEP2_PROMPT = """You previously answered a question. Review your answer carefully.
Question: {question}
Your previous answer: {step1_answer}
Is this answer accurate? Provide a verified or corrected answer.
Verified answer:"""

STEP3_PROMPT = """Based on your analysis, provide your final definitive answer.
Question: {question}
Initial answer: {step1_answer}
Verified answer: {step2_answer}
Final answer:"""


# ── scoring functions, copied verbatim from 02_semantic_relabel.py ──────────

def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match_any(prediction, references):
    norm_pred = normalize_answer(prediction)
    for ref in references:
        norm_ref = normalize_answer(ref)
        if not norm_ref:
            continue
        if norm_pred == norm_ref or norm_ref in norm_pred:
            return True
    return False


def token_f1_any(prediction, references):
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for ref in references:
        ref_tokens = normalize_answer(ref).split()
        if not pred_tokens or not ref_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return best_f1


def word_overlap(a, b):
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def rescore_step3(generated, dataset_source, question, nq_map, tqa_map):
    if dataset_source == "nq_open":
        aliases = nq_map.get(question, [])
        if not aliases:
            return None
        em = exact_match_any(generated, aliases)
        f1 = token_f1_any(generated, aliases)
        return 1 if (em or f1 >= 0.5) else 0
    else:
        entry = tqa_map.get(question)
        if entry is None:
            return None
        best_correct = max((word_overlap(generated, a) for a in entry["correct_answers"]), default=0.0)
        best_incorrect = max((word_overlap(generated, a) for a in entry["incorrect_answers"]), default=0.0)
        if best_correct > 0.12 or best_incorrect > 0.12:
            return 1 if best_correct >= best_incorrect else 0
        return 0


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} Qwen pipelines")

    s1_correct = [d for d in data if d["relabel_label_s1"] == 1 and d.get("h_s1") is not None]
    print(f"{len(s1_correct)} step-1-correct pipelines with h_s1 available")

    rng = random.Random(RANDOM_STATE)
    rng.shuffle(s1_correct)
    n_train = len(s1_correct) - N_TEST
    train, test = s1_correct[:n_train], s1_correct[n_train:n_train + N_TEST]
    print(f"Train (direction-fitting): {len(train)}  Test (causal intervention): {len(test)}")

    h_train = np.array([d["h_s1"] for d in train])
    y_train = np.array([d["relabel_label_s3"] for d in train])
    mu_success = h_train[y_train == 1].mean(axis=0)
    mu_fail = h_train[y_train == 0].mean(axis=0)
    direction = mu_success - mu_fail
    direction = direction / np.linalg.norm(direction)
    print(f"Found direction fitted on {len(train)} pipelines "
          f"({int(y_train.sum())} eventually-correct, {int((1-y_train).sum())} eventually-wrong)")

    rng_np = np.random.default_rng(RANDOM_STATE)
    random_dir = rng_np.standard_normal(direction.shape)
    random_dir = random_dir / np.linalg.norm(random_dir)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
    model.eval()

    print("Loading NQ-open / TruthfulQA answer maps for step-3 rescoring...")
    nq = load_dataset("nq_open", split="train").shuffle(seed=42).select(range(1683))
    nq_map = {x["question"].strip(): list(x["answer"]) for x in nq}
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    tqa_map = {x["question"].strip(): {"correct_answers": list(x["correct_answers"]),
                                        "incorrect_answers": list(x["incorrect_answers"])}
               for x in tqa}

    def generate(prompt, steer_vec=None, alpha=0.0):
        handle = None
        if steer_vec is not None:
            vec = torch.tensor(steer_vec, dtype=model.dtype, device=device) * alpha

            def hook(module, inp, out):
                out_t = out[0] if isinstance(out, tuple) else out
                out_t = out_t + vec
                return (out_t,) + out[1:] if isinstance(out, tuple) else out_t

            handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
        if handle is not None:
            handle.remove()
        return tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    results_per_alpha = []
    for alpha in ALPHAS:
        found_correct, random_correct = [], []
        n_valid = 0
        for i, d in enumerate(test):
            question, step1_answer = d["question"], d["step1_output"]
            p2 = STEP2_PROMPT.format(question=question, step1_answer=step1_answer)

            s2_found = generate(p2, steer_vec=direction, alpha=alpha)
            p3_found = STEP3_PROMPT.format(question=question, step1_answer=step1_answer, step2_answer=s2_found)
            s3_found = generate(p3_found)

            s2_random = generate(p2, steer_vec=random_dir, alpha=alpha)
            p3_random = STEP3_PROMPT.format(question=question, step1_answer=step1_answer, step2_answer=s2_random)
            s3_random = generate(p3_random)

            label_found = rescore_step3(s3_found, d["dataset_source"], question, nq_map, tqa_map)
            label_random = rescore_step3(s3_random, d["dataset_source"], question, nq_map, tqa_map)
            if label_found is None or label_random is None:
                continue
            n_valid += 1
            found_correct.append(label_found)
            random_correct.append(label_random)
            if (i + 1) % 10 == 0:
                print(f"  alpha={alpha} [{i+1}/{len(test)}]", flush=True)

        found_correct = np.array(found_correct)
        random_correct = np.array(random_correct)
        b = int(((found_correct == 1) & (random_correct == 0)).sum())  # found-only-correct
        c = int(((found_correct == 0) & (random_correct == 1)).sum())  # random-only-correct
        mcnemar_p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
        results_per_alpha.append({
            "alpha": alpha, "n_valid": n_valid,
            "found_correct_rate": float(found_correct.mean()) if n_valid else None,
            "random_correct_rate": float(random_correct.mean()) if n_valid else None,
            "mcnemar_b_found_only": b, "mcnemar_c_random_only": c,
            "mcnemar_p": round(float(mcnemar_p), 4),
        })
        print(f"alpha={alpha}: found_rate={found_correct.mean():.3f} random_rate={random_correct.mean():.3f} "
              f"b={b} c={c} mcnemar_p={mcnemar_p:.4f}")

    summary = {
        "model": MODEL_ID, "target_layer": TARGET_LAYER,
        "n_train_direction_fit": len(train), "n_test": len(test),
        "results_per_alpha": results_per_alpha,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
