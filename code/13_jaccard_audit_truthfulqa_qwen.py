"""
Paper 3 -- Tier 2 final-audit pass: extends the manual false-positive
audit (04_substring_match_audit.py, NQ-open exact/substring match) to
TruthfulQA's Jaccard word-overlap labeling, closing the paper's own
disclosed gap: "We do not extend this audit to... TruthfulQA's
Jaccard-based labeling on either model."

TruthfulQA's Jaccard scheme (relabel_tqa, code/02_semantic_relabel.py)
has no clean analogue to NQ-open's "substring but not exact match"
bucket -- nearly all (275/280, 98.2%) step-3-correct Qwen/TruthfulQA
pipelines already sit at Jaccard <= 0.35, since full-sentence answers
rarely reach high word-overlap even when genuinely correct. The
analogous "weakest evidence" risk bucket here is score close to the
0.12 pass/fail threshold rather than "substring vs. exact" -- pipelines
labeled correct on the thinnest possible lexical-overlap margin. We
define this bucket as relabel_score_s3 <= 0.25 (244/280 = 87.1% of the
correct pool -- itself informative about how much of this paper's
TruthfulQA "correct" count rests on weak overlap) and sample 30 at
random (seed=7, matching the NQ-open audit's convention) for manual
reading. Qwen only (TruthfulQA's larger, primary arm in this paper);
Mistral's TruthfulQA arm is not audited here, disclosed as a further
scoping choice.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCORE_THRESHOLD = 0.25


def main():
    with open(f"{ROOT}/results/qwen/results/qwen/logs/pipeline_data_relabeled.json") as f:
        data = json.load(f)
    tqa = [d for d in data if d["dataset_source"] == "truthfulqa"]
    correct = [d for d in tqa if d.get("relabel_label_s3") == 1]

    weak_evidence_correct = [d for d in correct if d["relabel_score_s3"] <= SCORE_THRESHOLD]

    print(f"TruthfulQA correct pool: {len(correct)}/{len(tqa)} "
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

    with open(ROOT / "results" / "jaccard_audit_truthfulqa_qwen_sample.json", "w") as f:
        json.dump({"correct_pool": len(correct), "tqa_total": len(tqa),
                    "weak_evidence_pool": len(weak_evidence_correct), "sample": out}, f, indent=2)
    print(f"\nSaved sample: {ROOT}/results/jaccard_audit_truthfulqa_qwen_sample.json")


if __name__ == "__main__":
    main()
