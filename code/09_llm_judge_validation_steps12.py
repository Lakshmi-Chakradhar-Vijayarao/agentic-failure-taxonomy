"""
Paper 3 -- Tier 1 final-audit pass: extends 08_llm_judge_validation.py's
LLM-judge check from step 3 (the final answer) only to steps 1 and 2 as
well. This closes a scope gap the elite-reviewer to-do list flagged: the
only semantic-judge validation of this paper's relabeled surface-form
metric covered the pipeline's last step, leaving the intermediate steps'
labels checked only by the surface-form (Jaccard/ROUGE) relabeling
itself.

No new generation or upstream data is needed: `pipeline_data_relabeled.json`
already contains `step1_output`/`step2_output` and
`relabel_label_s1`/`relabel_label_s2` for both models (identical fields to
`step3_output`/`relabel_label_s3`, just for the earlier steps), so this is
the exact same judge, prompt, sampling, and metrics as 08, generalized by
a `step` parameter instead of hard-coded to step 3.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RANDOM_STATE = 42
N_PER_MODEL = {"qwen": 150, "mistral": 100}
STEPS = [1, 2]

PATHS = {
    "qwen": f"{ROOT}/results/qwen/results/qwen/logs/pipeline_data_relabeled.json",
    "mistral": f"{ROOT}/results/mistral/results/mistral/logs/pipeline_data_relabeled.json",
}

JUDGE_PROMPT = """You are a factuality judge. You will be given a question, the correct reference answer, and a model's answer.
Decide if the model's answer is CORRECT or HALLUCINATED.
A response is CORRECT if its core factual claim matches or is consistent with the reference answer, even if phrased differently.
A response is HALLUCINATED if it asserts something factually wrong, contradicts the reference, or fails to answer.
Ignore phrasing differences and focus only on factual accuracy.

Question: {question}
Reference answer: {reference}
Model's answer: {response}

Reply with exactly one word: CORRECT or HALLUCINATED."""


def stratified_sample(data, n, label_key, seed=RANDOM_STATE):
    rng = random.Random(seed)
    correct = [d for d in data if d[label_key] == 1]
    wrong = [d for d in data if d[label_key] == 0]
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


def run_model_step(model_key, step, model, tokenizer, device):
    output_key = f"step{step}_output"
    label_key = f"relabel_label_s{step}"
    with open(PATHS[model_key]) as f:
        data = json.load(f)
    sample = stratified_sample(data, N_PER_MODEL[model_key], label_key)
    print(f"\n[{model_key}/step{step}] Judging {len(sample)} samples "
          f"({label_key}: {sum(d[label_key] for d in sample)} correct / "
          f"{len(sample) - sum(d[label_key] for d in sample)} wrong)...")

    surface_labels, judge_labels, ambiguous = [], [], 0
    for i, d in enumerate(sample):
        verdict = query_judge(model, tokenizer, device, d["question"], d["best_answer"], d[output_key])
        if verdict == -1:
            ambiguous += 1
            continue
        surface_labels.append(d[label_key])
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

    print(f"[{model_key}/step{step}] n_scored={len(surface_labels)} (ambiguous={ambiguous})")
    print(f"[{model_key}/step{step}] Agreement={agreement:.4f}  Cohen kappa={kappa:.4f}")
    print(f"[{model_key}/step{step}] Confusion matrix (rows=surface, cols=judge): {cm}")
    print(f"[{model_key}/step{step}] surface=correct,judge=hallucinated: {surface_correct_judge_hall} "
          f"({surface_correct_judge_hall/len(surface_labels)*100:.1f}%)")
    print(f"[{model_key}/step{step}] surface=hallucinated,judge=correct: {surface_hall_judge_correct} "
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

    out_path = Path(f"{ROOT}/results/llm_judge_validation_steps12.json")
    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming: found existing results for {list(results.keys())}")

    for model_key in PATHS:
        for step in STEPS:
            key = f"{model_key}_step{step}"
            if key in results:
                print(f"Skipping {key} (already completed)")
                continue
            results[key] = run_model_step(model_key, step, model, tokenizer, device)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Checkpoint saved: {out_path}")

    print("\nFinal save complete.")


if __name__ == "__main__":
    main()
