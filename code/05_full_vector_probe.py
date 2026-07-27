"""
Paper 3 -- closing the "full-dimensional probe not run" gap (Sec 2.3/3.2,
Contribution #3). The scalar early-detection features (L2 norm, cosine
drift) are a genuine null (AUROC 0.50-0.53), but they do not exhaust
"hidden-state geometry" broadly. This script feeds the FULL 2048-d
hidden-state vectors already cached in pipeline_data_relabeled.json
(h_s1, h_s2, their concatenation, and their difference) into a
PCA(100)+L2-logistic probe, 5-fold CV, against the corrected
relabel_label_s3 target -- no new model inference, reusing data already
on disk for both Qwen and Mistral.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # fixed post-final-audit: was hardcoded to a personal absolute path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PATHS = {
    "qwen": f"{ROOT}/results/qwen/results/qwen/logs/pipeline_data_relabeled.json",
    "mistral": f"{ROOT}/results/mistral/results/mistral/logs/pipeline_data_relabeled.json",
}


def cv_auroc(X, y, n_components=100, n_splits=5, seed=42):
    k = min(n_components, X.shape[1], X.shape[0] - 2)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=k, random_state=seed)),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed, C=1.0)),
    ])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv = cross_validate(pipe, X, y, cv=skf, scoring="roc_auc")
    return float(cv["test_score"].mean()), float(cv["test_score"].std())


def run_model(model_key, path):
    with open(path) as f:
        data = json.load(f)
    h_s1 = np.array([d["h_s1"] for d in data])
    h_s2 = np.array([d["h_s2"] for d in data])
    y = np.array([d["relabel_label_s3"] for d in data])
    y_wrong = 1 - y

    print(f"\n=== {model_key}  N={len(data)}  dim={h_s1.shape[1]}  wrong_s3_rate={y_wrong.mean():.4f} ===")

    feature_sets = {
        "h_s1_full": h_s1,
        "h_s2_full": h_s2,
        "concat_s1_s2": np.hstack([h_s1, h_s2]),
        "diff_s2_minus_s1": h_s2 - h_s1,
    }
    results = {}
    for name, X in feature_sets.items():
        auc, std = cv_auroc(X, y_wrong)
        results[name] = {"auroc": round(auc, 4), "std": round(std, 4)}
        print(f"  {name:20s} (PCA-100 + L2-logistic, 5-fold CV): AUROC={auc:.4f} (+/-{std:.4f})")

    out = {"model": model_key, "n": len(data), "wrong_s3_rate": float(y_wrong.mean()), "results": results}
    return out


def main():
    all_results = {}
    for model_key, path in PATHS.items():
        if not Path(path).exists():
            print(f"Skipping {model_key}: {path} not found")
            continue
        all_results[model_key] = run_model(model_key, path)

    out_path = Path(f"{ROOT}/results/full_vector_probe.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
