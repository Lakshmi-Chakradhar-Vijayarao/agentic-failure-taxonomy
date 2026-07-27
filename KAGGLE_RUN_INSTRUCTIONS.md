# Paper 3 — Kaggle Execution Runbook

Resource decision (given Kaggle GPU + HF API only, nothing else available):

| Model | N (pipelines) | Est. GPU-hours (T4) | Purpose |
|---|---|---|---|
| Qwen 2.5 3B (primary) | 2500 | ~16h | Fixes LATENT's n=3 problem -> ~12 expected LATENT samples at the observed ~0.5% rate. Report with exact (Wilson/Clopper-Pearson) CIs given the small count, not asymptotic normal CIs. |
| Mistral 7B Instruct v0.3 (cross-arch check) | 1200 | ~8h | Tests whether cascade rate, 8-mode taxonomy proportions, and early-detection AUROC replicate directionally on a second, larger, differently-trained model. Chosen over Llama-3.1 specifically because it is NOT gated on HuggingFace -- no license approval wait, works immediately with a plain HF API call. |

Total: ~24 GPU-hours, fits inside roughly one week of Kaggle's free tier
(~30 GPU-hrs/week). Kaggle session time-limits (9-12h continuous) mean each
model's run will likely span 2 Kaggle sessions -- the script already
checkpoints every 100 samples and supports `--resume`.

## Step-by-step

1. **Create a new Kaggle Notebook**, Settings -> Accelerator -> GPU T4 x2 (or x1
   if x2 isn't available on your account tier), Internet -> On.

2. **Upload as a Kaggle Dataset** (or just copy-paste into notebook cells):
   - `fail-chain/src/pipeline.py`
   - `fail-chain/src/failure_modes.py`
   - `papers-2026/paper3-agentic-failures/code/01_scaled_pipeline_generation.py`

   Simplest path: create a Kaggle Dataset named `fail-chain-src` containing
   the `src/` folder, add it as a notebook input, and put
   `01_scaled_pipeline_generation.py` directly in a notebook cell (edit the
   `FAIL_CHAIN_ROOT` path at the top to point at `/kaggle/input/fail-chain-src`
   instead of the local Desktop path before uploading).

3. **Install deps** (first cell):
   ```
   !pip install -q transformers accelerate rouge_score datasets
   ```

4. **Qwen run** (no HF token needed, Qwen2.5-3B-Instruct is ungated):
   ```
   !python 01_scaled_pipeline_generation.py --model qwen --n 2500
   ```
   If the session times out partway through, restart the notebook and
   re-run with `--resume` added -- it picks up from the last checkpoint
   in `results/qwen/logs/pipeline_data_checkpoint.json`:
   ```
   !python 01_scaled_pipeline_generation.py --model qwen --n 2500 --resume
   ```

5. **Mistral run** (also ungated, no HF token required, but set one anyway
   via Kaggle Secrets -> `HF_TOKEN` if you hit rate limits):
   ```
   !python 01_scaled_pipeline_generation.py --model mistral --n 1200
   ```
   Same `--resume` pattern if a session times out.

6. **Download results**: `results/qwen/logs/pipeline_data.json`,
   `results/qwen/hidden_states/pipeline_hidden_states.npz`,
   `results/mistral/logs/pipeline_data.json`,
   `results/mistral/hidden_states/pipeline_hidden_states.npz`, plus both
   `ablation_data.json` files. Save these back into
   `papers-2026/paper3-agentic-failures/results/{qwen,mistral}/` locally --
   that's what the next analysis step (taxonomy recompute + early-detection
   classifier re-fit at the larger N) will read.

## What NOT to do

- Don't run both models in the same notebook session back-to-back without
  checkpointing between them -- if the session dies mid-Mistral-run, you
  don't want to lose the completed Qwen data. Run Qwen to completion, save
  its output, then start Mistral fresh (or in a new session).
- Don't increase N further without re-checking the Kaggle weekly GPU quota
  -- 2500+1200 already uses most of a free-tier week.

## After the data lands

Once both `pipeline_data.json` files exist locally, the next steps (already
in the Paper 3 todo list) are: recompute the 8-mode taxonomy and LATENT
statistics at the new N with proper exact CIs, re-fit the early-detection
classifier, and compare cascade rate / taxonomy proportions / AUROC across
Qwen vs. Mistral to write the cross-architecture generalization section.
