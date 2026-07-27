"""
Manual audit of the NQ-open substring-match false-positive risk flagged
in round-2/3 review. Identifies all step-3 pipelines labeled correct via
substring-containment only (no exact full-string match), samples 30 at
random (seed=42 for question pool, seed=7 for the audit sample), and
reports the manually-verified false-positive rate with a Wilson CI.

Result (recorded here for reproducibility -- the manual classification
itself is not automatable and is embedded in the paper draft directly):
2/30 confirmed false positives (6.7%, Wilson 95% CI [1.8%, 21.3%]),
propagating to an estimated ~1.2% of the full N=2500 Qwen dataset.
"""
import json
import re
import string
import random
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path

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

    with open(
        f"{ROOT}/"
        "results/qwen/results/qwen/logs/pipeline_data_relabeled.json"
    ) as f:
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

    print(f"Substring-only-correct pool: {len(substring_only_correct)}/{len(nq_data)}"
          f" ({len(substring_only_correct)/len(nq_data)*100:.1f}%)")
    random.seed(7)
    sample = random.sample(substring_only_correct, 30)
    for item in sample:
        print(f"\nQ: {item[1]}\n  Gen: {item[2]}\n  Matched: {item[3]}")


if __name__ == "__main__":
    main()
