> **Note (October 2026).** This is an earlier version of the work ("Cascade Is Real But Overstated"). The current version, *Self-Refinement Failure in Multi-Step LLM Pipelines: A Re-audit with Measured Difficulty*, is in revision. It adds a pre-registered test showing that measured question difficulty alone predicts failure (AUROC 0.96–0.97 on NQ-open), with no detectable added information from the hidden-state probe.

---

# Agentic Failure Taxonomy

Code, cached results, and paper source for "Cascade Is Real But Overstated — A
Corrected Taxonomy of Multi-Step LLM Self-Refinement Failure."

## What's included
- `code/` — all pipeline, analysis, and experiment scripts, numbered by stage.
- `results/*.json` (top level) — every summary statistic reported in the paper
  (taxonomy bootstrap CIs, probe results, causal-steering results, LLM-judge
  validation, etc.) — sufficient to re-verify every number without re-running
  generation.
- `draft/` — the paper source.
- `related_work/`, `requirements.txt`, `KAGGLE_RUN_INSTRUCTIONS.md`.

## What's excluded, and why
Full per-item generation logs and hidden-state caches under
`results/qwen/results/qwen/` and `results/mistral/results/mistral/`
(`pipeline_hidden_states.npz`, `pipeline_data*.json`, `ablation_data*.json` —
100MB-300MB+ each) are excluded. These are raw per-item pipeline outputs, not
final results; every statistic computed from them is already saved in the small
top-level `results/*.json` files. Scripts that read these raw logs directly
(e.g. `03_recompute_under_corrected_labels.py`, `05_full_vector_probe.py`,
`09_causal_step2_steering_intervention.py`) require regenerating them first via
`01_scaled_pipeline_generation.py`, which requires API/model access this
repository does not itself provide.

## Verifying a reported number without regenerating raw logs
Every number in the paper traces to a small, included top-level `results/*.json`
file — open it directly rather than re-running the full generation pipeline.
