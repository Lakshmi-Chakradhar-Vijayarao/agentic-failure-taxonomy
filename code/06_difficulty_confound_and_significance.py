"""
Paper 3 -- round-5 review closed three gaps in the full-vector probe
(05_full_vector_probe.py) result:

1. SIGNIFICANCE: the "3-6 SD above chance" claim in the draft divided
   (AUROC-0.5) by the CV-fold standard deviation, which is not a valid
   null-hypothesis SE (5 folds are not 5 independent draws under
   H0: AUROC=0.5). Replaced here with a Mann-Whitney U test on
   out-of-fold predictions (a single cross_val_predict pass, not
   hundreds of permutation refits): AUROC is a linear rescaling of the
   U statistic (AUROC = U / (n_pos * n_neg)), so testing U against its
   null distribution is an exact, standard, and far cheaper equivalent
   of a label-permutation test on AUROC itself.

2. DIFFICULTY CONFOUND (the decisive test): h_s1 alone (0.61 Qwen) beats
   the dynamic drift feature h_s2-h_s1 (0.55 Qwen) by a wide margin --
   the reviewer's point is that this is the signature of a
   step-1-answerability/difficulty readout, not genuine cascade
   detection. Decisive, free test: restrict the probe to pipelines
   CORRECT at step 1 only (relabel_label_s1==1). If AUROC collapses to
   chance within that subset, the "signal" is difficulty, not
   early-warning, since there is no difficulty variation left to read
   once every pipeline in the subset already answered step 1 correctly
   (predicting step-3 failure among the answered-correctly-at-step-1
   pipelines is exactly the LATENT/SELF_HEAL/CORRECT distinction the
   taxonomy cares about, and difficulty alone cannot explain it).

3. CROSS-SOURCE TRANSFER: train on TruthfulQA, test on NQ-open (and
   vice versa) to check whether the full-vector probe is reading
   dataset-source identity rather than a source-general signal, since
   difficulty and dataset source are confounded in this data (cascade
   rate differs sharply by source).

All reuse the same cached hidden-state vectors already in
pipeline_data_relabeled.json -- no new model inference.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PATHS = {
    "qwen": f"{ROOT}/results/qwen/results/qwen/logs/pipeline_data_relabeled.json",
    "mistral": f"{ROOT}/results/mistral/results/mistral/logs/pipeline_data_relabeled.json",
}
RANDOM_STATE = 42


def build_pipe(n_components, seed=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed, C=1.0)),
    ])


def cv_auroc(X, y, n_components=100, n_splits=5, seed=RANDOM_STATE):
    n_splits_eff = min(n_splits, int(y.sum()), int((y == 0).sum()))
    n_splits_eff = max(2, n_splits_eff)
    k = min(n_components, X.shape[1], X.shape[0] - 2)
    pipe = build_pipe(k, seed)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    cv = cross_validate(pipe, X, y, cv=skf, scoring="roc_auc")
    return float(cv["test_score"].mean()), float(cv["test_score"].std()), n_splits_eff


def mann_whitney_auc_test(X, y, n_components=100, n_splits=5, seed=RANDOM_STATE):
    """Exact, cheap alternative to a label-permutation test: get
    out-of-fold predicted probabilities via a single cross_val_predict
    pass, then run a one-sided Mann-Whitney U test (positives ranked
    higher than negatives). AUROC = U / (n_pos * n_neg) exactly, so this
    tests the same null (AUROC=0.5) the permutation test would, without
    refitting the CV pipeline hundreds of times."""
    n_splits_eff = min(n_splits, int(y.sum()), int((y == 0).sum()))
    n_splits_eff = max(2, n_splits_eff)
    k = min(n_components, X.shape[1], X.shape[0] - 2)
    pipe = build_pipe(k, seed)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
    oof_probs = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    pos_scores = oof_probs[y == 1]
    neg_scores = oof_probs[y == 0]
    u_stat, p_value = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    auc_from_u = float(u_stat) / (len(pos_scores) * len(neg_scores))
    return {
        "auroc_from_oof": auc_from_u,
        "n_splits": n_splits_eff,
        "mannwhitney_u": float(u_stat),
        "p_value": float(p_value),
    }


def load(model_key):
    with open(PATHS[model_key]) as f:
        data = json.load(f)
    h_s1 = np.array([d["h_s1"] for d in data])
    h_s2 = np.array([d["h_s2"] for d in data])
    y3 = np.array([d["relabel_label_s3"] for d in data])
    y1 = np.array([d["relabel_label_s1"] for d in data])
    source = np.array([d["dataset_source"] for d in data])
    return h_s1, h_s2, y3, y1, source


def run_permutation_fix(model_key):
    h_s1, h_s2, y3, y1, source = load(model_key)
    y_wrong = 1 - y3
    concat = np.hstack([h_s1, h_s2])
    auc, std, splits = cv_auroc(concat, y_wrong)
    mw = mann_whitney_auc_test(concat, y_wrong)
    n_sd = (auc - 0.5) / std if std > 0 else float("nan")
    print(f"\n[{model_key}] concat(h_s1,h_s2) full-vector probe: AUROC={auc:.4f} +-{std:.4f} "
          f"({n_sd:.2f} fold-SD above chance -- NOT a valid test)")
    print(f"[{model_key}] Mann-Whitney U test on OOF predictions: "
          f"AUROC(from U)={mw['auroc_from_oof']:.4f}, p={mw['p_value']:.6f}")
    return {"observed_auroc": auc, "fold_std": std, "fold_sd_above_chance": n_sd,
            "mann_whitney_test": mw}


def run_step1_correct_subset(model_key):
    """The decisive test: restrict to pipelines correct at step 1. If the
    full-vector signal collapses to chance here, it was reading step-1
    answerability/difficulty, not a cascade-specific signal."""
    h_s1, h_s2, y3, y1, source = load(model_key)
    mask = y1 == 1
    n_sub = int(mask.sum())
    print(f"\n[{model_key}] Step-1-correct subset: n={n_sub}/{len(y1)} "
          f"({n_sub/len(y1)*100:.1f}%)")
    if n_sub < 20:
        print(f"[{model_key}] subset too small for a meaningful CV probe, skipping")
        return None

    h_s1_sub, h_s2_sub = h_s1[mask], h_s2[mask]
    y3_sub = y3[mask]
    y_wrong_sub = 1 - y3_sub
    print(f"[{model_key}] within-subset wrong_s3_rate={y_wrong_sub.mean():.4f} "
          f"(this is exactly the LATENT+SELF_HEAL+... distinction the taxonomy targets)")

    concat_sub = np.hstack([h_s1_sub, h_s2_sub])
    k_sub = min(100, n_sub - 2)
    auc, std, splits = cv_auroc(concat_sub, y_wrong_sub, n_components=k_sub)
    mw = mann_whitney_auc_test(concat_sub, y_wrong_sub, n_components=k_sub)
    print(f"[{model_key}] within-step1-correct-subset concat probe: AUROC={auc:.4f}+-{std:.4f} "
          f"({splits}-fold), Mann-Whitney p={mw['p_value']:.6f}")
    return {
        "n_subset": n_sub, "n_total": int(len(y1)), "wrong_s3_rate_in_subset": float(y_wrong_sub.mean()),
        "auroc": auc, "std": std, "n_splits": splits, "mann_whitney_test": mw,
    }


def run_cross_source_transfer(model_key):
    """Train on one dataset source, test on the other, using the cached
    activations -- checks whether the full-vector probe is reading
    dataset-source identity rather than a source-general signal."""
    h_s1, h_s2, y3, y1, source = load(model_key)
    y_wrong = 1 - y3
    concat = np.hstack([h_s1, h_s2])
    sources = sorted(set(source.tolist()))
    print(f"\n[{model_key}] Cross-source transfer. Sources present: {sources}")
    results = {}
    for train_src in sources:
        test_src = [s for s in sources if s != train_src]
        if not test_src:
            continue
        train_mask = source == train_src
        test_mask = source != train_src
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
        if min(int(y_wrong[train_mask].sum()), int((1 - y_wrong[train_mask]).sum())) < 5:
            print(f"[{model_key}] train={train_src} (n={n_train}): too few of one class, skipping")
            continue
        k = min(100, n_train - 2)
        pipe = build_pipe(k)
        pipe.fit(concat[train_mask], y_wrong[train_mask])
        probs = pipe.predict_proba(concat[test_mask])[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_wrong[test_mask], probs))
        except ValueError:
            auc = float("nan")
        print(f"[{model_key}] train={train_src} (n={n_train}) -> test={test_mask.sum()} "
              f"samples from {[s for s in sources if s != train_src]}: transfer AUROC={auc:.4f}")
        results[f"train_{train_src}"] = {
            "n_train": n_train, "n_test": n_test, "transfer_auroc": auc,
        }
    return results


def main():
    out = {}
    for model_key in PATHS:
        if not Path(PATHS[model_key]).exists():
            continue
        out[model_key] = {
            "permutation_fix": run_permutation_fix(model_key),
            "step1_correct_subset": run_step1_correct_subset(model_key),
            "cross_source_transfer": run_cross_source_transfer(model_key),
        }

    out_path = Path(f"{ROOT}/results/difficulty_confound_and_significance.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
