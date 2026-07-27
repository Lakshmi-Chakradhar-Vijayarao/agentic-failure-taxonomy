"""
Paper 3 -- elite-review follow-up: extends 13_jaccard_audit_truthfulqa_qwen.py
(Qwen only) to Mistral, closing the paper's own disclosed gap: "Mistral's
TruthfulQA arm remains un-audited, a further disclosed scoping choice."

Identical procedure: weak-evidence-correct bucket defined as
relabel_score_s3 <= 0.25 among step-3-correct TruthfulQA pipelines,
30-item sample (seed 7, matching the Qwen audit's convention) for manual
reading.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCORE_THRESHOLD = 0.25


def main():
    with open(f"{ROOT}/results/mistral/results/mistral/logs/pipeline_data_relabeled.json") as f:
        data = json.load(f)
    tqa = [d for d in data if d["dataset_source"] == "truthfulqa"]
    correct = [d for d in tqa if d.get("relabel_label_s3") == 1]

    weak_evidence_correct = [d for d in correct if d["relabel_score_s3"] <= SCORE_THRESHOLD]

    print(f"Mistral TruthfulQA correct pool: {len(correct)}/{len(tqa)} "
          f"({len(correct)/len(tqa)*100:.1f}%)")
    print(f"Weak-evidence-correct pool (score<={SCORE_THRESHOLD}): "
          f"{len(weak_evidence_correct)}/{len(correct)} "
          f"({len(weak_evidence_correct)/len(correct)*100:.1f}%)")

    random.seed(7)
    n_sample = min(30, len(weak_evidence_correct))
    sample = random.sample(weak_evidence_correct, n_sample)

    out = []
    for d in sample:
        print(f"\nQ: {d['question']}\n  Best answer (reference): {d['best_answer']}\n"
              f"  Gen: {d['step3_output']}\n  Jaccard score: {d['relabel_score_s3']}")
        out.append({"id": d["id"], "question": d["question"], "best_answer": d["best_answer"],
                     "generation": d["step3_output"], "score": d["relabel_score_s3"]})

    with open(ROOT / "results" / "jaccard_audit_truthfulqa_mistral_sample.json", "w") as f:
        json.dump({"correct_pool": len(correct), "tqa_total": len(tqa),
                    "weak_evidence_pool": len(weak_evidence_correct), "sample": out}, f, indent=2)
    print(f"\nSaved sample: {ROOT}/results/jaccard_audit_truthfulqa_mistral_sample.json")


if __name__ == "__main__":
    main()
