"""
Paper 3 — Recompute the early-detection classifier and the context-ablation
(P4) comparison under the corrected (SQuAD EM/F1 + multi-ref Jaccard)
labels, instead of the original ROUGE-L labels that the adversarial review
showed were massively undercounting correctness (see 02_semantic_relabel.py,
which produced pipeline_data_relabeled.json).

This reuses the already-extracted hidden states (h_s1/h_s2/h_s3, already on
disk, no new GPU inference needed) and recomputes:
  1. The early-vs-late detection classifier (predicting eventual step-3
     failure from step-1/step-2 hidden-state features) using the corrected
     relabel_label_s3 as the target instead of the original label_s3.
  2. The context-ablation (noctx) comparison of AMPLIFIED-mode rate,
     relabeled the same way, to check whether the original P4 "REFUTED"
     finding (context-stripping doesn't reduce AMPLIFIED, in fact makes it
     slightly worse) survives under a labeling scheme that isn't
     collapsing everything to CASCADE.
"""
import json
import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RELABELED_PATH = Path(
    f"{ROOT}/results/qwen/results/qwen/logs/pipeline_data_relabeled.json"
)
ABLATION_PATH = Path(
    f"{ROOT}/results/qwen/results/qwen/logs/ablation_data.json"
)
ABLATION_OUT_PATH = Path(
    f"{ROOT}/results/qwen/results/qwen/logs/ablation_data_relabeled.json"
)


# ── Relabeling functions (duplicated from 02_semantic_relabel.py for a
#    self-contained script; keep in sync if either changes) ──────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


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


def _word_overlap(a, b):
    words_a, words_b = set(a.lower().split()), set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def relabel_nq(generated, aliases):
    em = exact_match_any(generated, aliases)
    f1 = token_f1_any(generated, aliases)
    return 1 if (em or f1 >= 0.5) else 0


def relabel_tqa(generated, entry):
    best_correct = max((_word_overlap(generated, a) for a in entry["correct_answers"]), default=0.0)
    best_incorrect = max((_word_overlap(generated, a) for a in entry["incorrect_answers"]), default=0.0)
    if best_correct > 0.12 or best_incorrect > 0.12:
        return 1 if best_correct >= best_incorrect else 0
    return 0


def classify_failure_mode(wrong_s1, wrong_s2, wrong_s3):
    if not wrong_s1 and not wrong_s2 and not wrong_s3:
        return "CORRECT"
    if wrong_s1 and not wrong_s2 and not wrong_s3:
        return "ISOLATED"
    if not wrong_s1 and wrong_s2 and wrong_s3:
        return "AMPLIFIED"
    if wrong_s1 and wrong_s2 and wrong_s3:
        return "CASCADE"
    if wrong_s1 and wrong_s2 and not wrong_s3:
        return "LATE_RECOVERY"
    if not wrong_s1 and not wrong_s2 and wrong_s3:
        return "LATENT"
    if wrong_s1 and not wrong_s2 and wrong_s3:
        return "OSCILLATING"
    if not wrong_s1 and wrong_s2 and not wrong_s3:
        return "SELF_HEAL"
    return "PARTIAL"


def build_maps():
    print("Reloading reference datasets...")
    nq = load_dataset("nq_open", split="train")
    nq = nq.shuffle(seed=42).select(range(min(1683, len(nq))))
    nq_map = {x["question"].strip(): list(x["answer"]) for x in nq}
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    tqa_map = {x["question"].strip(): {"correct_answers": list(x["correct_answers"]),
                                        "incorrect_answers": list(x["incorrect_answers"])}
               for x in tqa}
    return nq_map, tqa_map


def relabel_sample(d, nq_map, tqa_map):
    q = d["question"].strip()
    source = d["dataset_source"]
    if source == "nq_open":
        aliases = nq_map.get(q)
        if aliases is None:
            return None
        l1 = relabel_nq(d["step1_output"], aliases)
        l2 = relabel_nq(d["step2_output"], aliases)
        l3 = relabel_nq(d["step3_output"], aliases)
    else:
        entry = tqa_map.get(q)
        if entry is None:
            return None
        l1 = relabel_tqa(d["step1_output"], entry)
        l2 = relabel_tqa(d["step2_output"], entry)
        l3 = relabel_tqa(d["step3_output"], entry)
    return l1, l2, l3


# ── 1. Early-vs-late detection classifier under corrected labels ─────────────

def run_detection_classifier(data):
    h_s1 = np.array([d["h_s1"] for d in data])
    h_s2 = np.array([d["h_s2"] for d in data])
    y = np.array([d["relabel_label_s3"] for d in data])  # 1=correct, 0=wrong under NEW labels
    y_wrong = 1 - y  # detection target: predict WRONG (matches original "predict failure" framing)

    print(f"N={len(data)}  wrong_s3 (corrected) rate: {y_wrong.mean():.4f}")

    # Late: step-3 hidden state alone (proxy: ||h_s3|| not available here without re-adding;
    # use step-2 as "late" since that's the last pre-output state we have cheaply) --
    # match original methodology: late = norm(h_s3) if present, else use h_s2 as before.
    h_s3 = np.array([d["h_s3"] for d in data])
    norm_s1 = np.linalg.norm(h_s1, axis=1, keepdims=True)
    norm_s2 = np.linalg.norm(h_s2, axis=1, keepdims=True)
    norm_s3 = np.linalg.norm(h_s3, axis=1, keepdims=True)
    cos_12 = np.sum(h_s1 * h_s2, axis=1, keepdims=True) / (
        np.linalg.norm(h_s1, axis=1, keepdims=True) * np.linalg.norm(h_s2, axis=1, keepdims=True) + 1e-8)
    drift_12 = np.linalg.norm(h_s2 - h_s1, axis=1, keepdims=True)

    late_features = norm_s3
    early_3feat = np.hstack([norm_s1, norm_s2, norm_s2 - norm_s1])
    early_5feat = np.hstack([norm_s1, norm_s2, norm_s2 - norm_s1, cos_12, drift_12])

    def cv_auroc(X, y):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(pipe, X, y, cv=skf, scoring="roc_auc")
        return scores["test_score"].mean(), scores["test_score"].std()

    late_auc, late_std = cv_auroc(late_features, y_wrong)
    early3_auc, early3_std = cv_auroc(early_3feat, y_wrong)
    early5_auc, early5_std = cv_auroc(early_5feat, y_wrong)

    print(f"Late (step-3 norm only):        AUROC={late_auc:.4f} (+/-{late_std:.4f})")
    print(f"Early (3-feat, steps 1+2):       AUROC={early3_auc:.4f} (+/-{early3_std:.4f})")
    print(f"Early (5-feat, +cosine drift):   AUROC={early5_auc:.4f} (+/-{early5_std:.4f})")

    # Output-level baseline: does label_s1 (corrected) alone predict label_s3 (corrected)?
    label_s1 = np.array([d["relabel_label_s1"] for d in data])
    baseline_auc, baseline_std = cv_auroc(label_s1.reshape(-1, 1), y_wrong)
    print(f"Output-level baseline (label_s1 predicts label_s3): AUROC={baseline_auc:.4f} (+/-{baseline_std:.4f})")

    return {
        "n": len(data), "wrong_s3_rate": float(y_wrong.mean()),
        "late_auroc": round(float(late_auc), 4), "late_std": round(float(late_std), 4),
        "early_3feat_auroc": round(float(early3_auc), 4), "early_3feat_std": round(float(early3_std), 4),
        "early_5feat_auroc": round(float(early5_auc), 4), "early_5feat_std": round(float(early5_std), 4),
        "output_baseline_auroc": round(float(baseline_auc), 4), "output_baseline_std": round(float(baseline_std), 4),
    }


def main():
    with open(RELABELED_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} relabeled pipeline samples")

    print("\n=== Early-detection classifier under CORRECTED labels ===")
    detection_results = run_detection_classifier(data)

    print("\n=== Context ablation (P4) under CORRECTED labels ===")
    if not ABLATION_PATH.exists():
        print(f"No ablation data found at {ABLATION_PATH}, skipping.")
        ablation_results = None
    else:
        with open(ABLATION_PATH) as f:
            ablation_data = json.load(f)
        print(f"Loaded {len(ablation_data)} ablation samples")

        nq_map, tqa_map = build_maps()

        # Relabel main pipeline subset matched to ablation (same n, same questions by id<n_ablation)
        n_ablation = len(ablation_data)
        std_matched = [d for d in data if d["id"] < n_ablation]

        relabeled_ablation = []
        missing = 0
        for d in ablation_data:
            res = relabel_sample(d, nq_map, tqa_map)
            if res is None:
                missing += 1
                relabeled_ablation.append(d)
                continue
            l1, l2, l3 = res
            mode = classify_failure_mode(l1 == 0, l2 == 0, l3 == 0)
            out = dict(d)
            out["relabel_label_s1"], out["relabel_label_s2"], out["relabel_label_s3"] = l1, l2, l3
            out["relabel_failure_mode"] = mode
            relabeled_ablation.append(out)
        print(f"Missing lookups in ablation set: {missing}/{len(ablation_data)}")

        with open(ABLATION_OUT_PATH, "w") as f:
            json.dump(relabeled_ablation, f, indent=2)

        std_amp = sum(1 for d in std_matched if d.get("relabel_failure_mode", d["failure_mode"]) == "AMPLIFIED")
        noctx_amp = sum(1 for d in relabeled_ablation if d.get("relabel_failure_mode", d["failure_mode"]) == "AMPLIFIED")
        std_amp_pct = std_amp / max(len(std_matched), 1) * 100
        noctx_amp_pct = noctx_amp / max(len(relabeled_ablation), 1) * 100

        print(f"AMPLIFIED (corrected labels) — Standard: {std_amp_pct:.2f}% ({std_amp}/{len(std_matched)})  "
              f"Context-free: {noctx_amp_pct:.2f}% ({noctx_amp}/{len(relabeled_ablation)})  "
              f"Drop: {std_amp_pct - noctx_amp_pct:+.2f}pp")

        ablation_results = {
            "n_standard": len(std_matched), "n_ablation": len(relabeled_ablation),
            "amplified_standard_pct": round(std_amp_pct, 2), "amplified_standard_n": std_amp,
            "amplified_noctx_pct": round(noctx_amp_pct, 2), "amplified_noctx_n": noctx_amp,
            "drop_pp": round(std_amp_pct - noctx_amp_pct, 2),
        }

    out = {"detection": detection_results, "context_ablation": ablation_results}
    out_path = Path(f"{ROOT}/results/recomputed_under_corrected_labels.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
