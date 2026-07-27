"""
Closes the last remaining disclosed-but-not-run gap in Paper 3: "the
corrected metric is still surface-form (normalized exact-match/F1,
multi-reference Jaccard), not semantic judgment by an independent model
-- an LLM-judge pass remains a natural, stronger next step we did not
run." Ollama is not installed on this machine, so this uses a small
instruction-tuned model loaded directly via transformers (the same
pattern used throughout this project) as the judge instead -- no new
infrastructure dependency, same result.

Judge: Qwen2.5-3B-Instruct (already used elsewhere in this project
family as a semantic judge, e.g. mech-int/HaRP's GPT-4o-alternative
audits), CPU, chat-template prompted for a single CORRECT/HALLUCINATED
verdict on each step-3 final answer.

Sample: a stratified sample from each model's relabeled pipeline data
(matching the corrected relabel_label_s3 distribution as closely as
possible, not just the first N rows), scored against relabel_label_s3
for agreement / Cohen's kappa -- the same diagnostic used in the
companion geom-proof project's own GPT-4o judge audit (Exp 07).
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RANDOM_STATE = 42
N_PER_MODEL = {"qwen": 150, "mistral": 100}

PATHS = {
    "qwen": f"{ROOT}/results/qwen/results/qwen/logs/pipeline_data_relabeled.json",
    "mistral": f"{ROOT}/results/mistral/results/mistral/logs/pipeline_data_relabeled.json",
}

JUDGE_PROMPT = """You are a factuality judge. You will be given a question, the correct reference answer, and a model's final answer.
Decide if the model's final answer is CORRECT or HALLUCINATED.
A response is CORRECT if its core factual claim matches or is consistent with the reference answer, even if phrased differently.
A response is HALLUCINATED if it asserts something factually wrong, contradicts the reference, or fails to answer.
Ignore phrasing differences and focus only on factual accuracy.

Question: {question}
Reference answer: {reference}
Model's final answer: {response}

Reply with exactly one word: CORRECT or HALLUCINATED."""


def stratified_sample(data, n, seed=RANDOM_STATE):
    rng = random.Random(seed)
    correct = [d for d in data if d["relabel_label_s3"] == 1]
    wrong = [d for d in data if d["relabel_label_s3"] == 0]
    n_correct = round(n * len(correct) / len(data))
    n_wrong = n - n_correct
    n_correct = min(n_correct, len(correct))
    n_wrong = min(n_wrong, len(wrong))
    sample = rng.sample(correct, n_correct) + rng.sample(wrong, n_wrong)
    rng.shuffle(sample)
    return sample


def query_judge(model, tokenizer, device, question, reference, response):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, response=response)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
    verdict = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    if "CORRECT" in verdict and "HALLUCINAT" not in verdict:
        return 1
    if "HALLUCINAT" in verdict:
        return 0
    return -1  # ambiguous


def run_model(model_key, model, tokenizer, device):
    with open(PATHS[model_key]) as f:
        data = json.load(f)
    sample = stratified_sample(data, N_PER_MODEL[model_key])
    print(f"\n[{model_key}] Judging {len(sample)} samples "
          f"(relabel_label_s3: {sum(d['relabel_label_s3'] for d in sample)} correct / "
          f"{len(sample) - sum(d['relabel_label_s3'] for d in sample)} wrong)...")

    surface_labels, judge_labels, ambiguous = [], [], 0
    for i, d in enumerate(sample):
        verdict = query_judge(model, tokenizer, device, d["question"], d["best_answer"], d["step3_output"])
        if verdict == -1:
            ambiguous += 1
            continue
        surface_labels.append(d["relabel_label_s3"])
        judge_labels.append(verdict)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(sample)}]", flush=True)

    surface_labels = np.array(surface_labels)
    judge_labels = np.array(judge_labels)
    agreement = float((surface_labels == judge_labels).mean())
    kappa = float(cohen_kappa_score(surface_labels, judge_labels)) if len(set(surface_labels)) > 1 and len(set(judge_labels)) > 1 else float("nan")
    cm = confusion_matrix(surface_labels, judge_labels).tolist()
    surface_correct_judge_hall = int(((surface_labels == 1) & (judge_labels == 0)).sum())
    surface_hall_judge_correct = int(((surface_labels == 0) & (judge_labels == 1)).sum())

    print(f"[{model_key}] n_scored={len(surface_labels)} (ambiguous={ambiguous})")
    print(f"[{model_key}] Agreement={agreement:.4f}  Cohen kappa={kappa:.4f}")
    print(f"[{model_key}] Confusion matrix (rows=surface, cols=judge): {cm}")
    print(f"[{model_key}] surface=correct,judge=hallucinated: {surface_correct_judge_hall} "
          f"({surface_correct_judge_hall/len(surface_labels)*100:.1f}%)")
    print(f"[{model_key}] surface=hallucinated,judge=correct: {surface_hall_judge_correct} "
          f"({surface_hall_judge_correct/len(surface_labels)*100:.1f}%)")

    return {
        "n_sampled": len(sample), "n_scored": len(surface_labels), "n_ambiguous": ambiguous,
        "agreement": agreement, "cohen_kappa": kappa, "confusion_matrix": cm,
        "surface_correct_judge_hallucinated": surface_correct_judge_hall,
        "surface_hallucinated_judge_correct": surface_hall_judge_correct,
        "surface_hall_rate": float(1 - surface_labels.mean()),
        "judge_hall_rate": float(1 - judge_labels.mean()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading judge model {JUDGE_MODEL} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, torch_dtype=torch.float32).to(device)
    model.eval()

    results = {}
    for model_key in PATHS:
        results[model_key] = run_model(model_key, model, tokenizer, device)
        out_path = Path(f"{ROOT}/results/llm_judge_validation.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Checkpoint saved: {out_path}")

    print("\nFinal save complete.")


if __name__ == "__main__":
    main()
