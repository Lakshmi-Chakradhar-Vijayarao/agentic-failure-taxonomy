"""
Paper 3 -- Tier 2 final-audit pass: extends 04_substring_match_audit.py
(Qwen NQ-open only) to Mistral NQ-open, closing the paper's own disclosed
gap: "We do not extend this audit to Mistral... Both remain un-audited."

Identical procedure to 04_substring_match_audit.py: identify all step-3
Mistral NQ-open pipelines labeled correct via substring-containment only
(no exact full-string match), sample 30 at random (seed=42 for the
NQ-open question pool -- matching 04's own seed, so the SAME underlying
question/answer-alias map is reused -- and seed=7 for the audit sample,
also matching 04), and print the sample for manual reading/classification
(manual classification is not automatable; results are recorded in the
paper draft directly, as with the original Qwen audit).
"""
import json
import random
import re
import string
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match_any(prediction, references):
    norm_pred = normalize_answer(prediction)
    hits = []
    for ref in references:
        norm_ref = normalize_answer(ref)
        if not norm_ref:
            continue
        if norm_pred == norm_ref:
            hits.append(("EQUAL", ref))
        elif norm_ref in norm_pred:
            hits.append(("SUBSTRING", ref))
    return hits


N_NQ_NEEDED = 1683


def main():
    nq = load_dataset("nq_open", split="train")
    nq = nq.shuffle(seed=42).select(range(min(N_NQ_NEEDED, len(nq))))
    qmap = {x["question"].strip(): list(x["answer"]) for x in nq}

    with open(f"{ROOT}/results/mistral/results/mistral/logs/pipeline_data_relabeled.json") as f:
        data = json.load(f)
    nq_data = [d for d in data if d["dataset_source"] == "nq_open"]

    substring_only_correct = []
    for d in nq_data:
        aliases = qmap.get(d["question"].strip())
        if not aliases:
            continue
        gen = d["step3_output"]
        hits = exact_match_any(gen, aliases)
        is_substring_only = (
            any(h[0] == "SUBSTRING" for h in hits)
            and not any(h[0] == "EQUAL" for h in hits)
        )
        if is_substring_only and d.get("relabel_label_s3") == 1:
            substring_only_correct.append((d["id"], d["question"], gen, hits))

    print(f"Mistral substring-only-correct pool: {len(substring_only_correct)}/{len(nq_data)}"
          f" ({len(substring_only_correct)/len(nq_data)*100:.1f}%)")
    random.seed(7)
    n_sample = min(30, len(substring_only_correct))
    sample = random.sample(substring_only_correct, n_sample)
    out = []
    for item in sample:
        print(f"\nQ: {item[1]}\n  Gen: {item[2]}\n  Matched: {item[3]}")
        out.append({"id": item[0], "question": item[1], "generation": item[2], "matched": item[3]})

    with open(ROOT / "results" / "substring_match_audit_mistral_sample.json", "w") as f:
        json.dump({"pool_size": len(substring_only_correct), "nq_total": len(nq_data),
                    "sample": out}, f, indent=2)
    print(f"\nSaved sample: {ROOT}/results/substring_match_audit_mistral_sample.json")


if __name__ == "__main__":
    main()
