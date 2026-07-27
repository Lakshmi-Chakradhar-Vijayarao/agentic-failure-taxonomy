# Cascade Is Real But Overstated: A Corrected Taxonomy of Multi-Step LLM Self-Refinement Failure, and a Mostly-Null Result on Geometric Early-Detection

**Lakshmi Chakradhar Vijayarao**
Independent Researcher
`lakshmichakradhar.v@gmail.com`

## Abstract

Self-refinement pipelines run generate, then critique, then refine -- now
common in deployed agentic systems. We build a 3-bit, 8-mode taxonomy of
failure across the pipeline's three steps and apply it to 2,500
Qwen2.5-3B and 1,200 Mistral-7B-Instruct pipelines on TruthfulQA and
NQ-open.

Our first labeling pass used ROUGE-L against a single reference answer,
following prior small-scale work in this space, and found a 97.28\%
cascade rate on Qwen with essentially no genuinely correct pipelines
(0.76\%). An independent adversarial review identified this as
substantially a labeling artifact: ROUGE-L is inappropriate for
NQ-open's short canonical answers, and the original pipeline discarded
valid answer aliases (9.1\% of NQ-open questions have 2+ acceptable
answers, only the first kept). Re-scoring the same completions with
SQuAD-style normalized exact-match/F1 (NQ-open, full alias set) and
multi-reference comparison (TruthfulQA) changes the picture
substantially: cascade remains dominant on Qwen but at 67.04\%, not
97.28\%, with 25.28\% genuinely correct throughout (not 0.76\%); on
Mistral, cascade is weaker still (49.33\%, correctness 38.33\%). Cascade
dominance is real but partly model-capability-dependent, not a fixed
property of self-refinement pipelines in general. A residual
labeling-bias audit estimates $\approx$1.2\% (Qwen NQ-open) and
$\approx$0.5\% (Mistral NQ-open) further mislabeling, both slightly
overstating CORRECT; a parallel audit of TruthfulQA's Jaccard-based
labeling (Qwen) finds a substantially higher 20.0\% confirmed
false-positive rate ($\approx$2.0-4.6\% dataset-wide), and the same audit
extended to Mistral's TruthfulQA arm finds a comparable, if numerically
different, rate (\S5) -- sharpening rather than resolving this paper's
caveat that TruthfulQA percentages rest on a weaker labeling foundation
than NQ-open's.

A same-scale reasoning-vs-non-reasoning comparison
(DeepSeek-R1-Distill-Qwen-1.5B, budget-forced, vs.\ its non-reasoning
base Qwen2.5-1.5B-Instruct) shows the reasoning model cascades more
(81.0\% vs.\ 67.3\%) without being more often correct (12.3\% vs.\
13.0\%): reasoning traces change the failure pattern, not the rate of
genuine correctness. A smaller, free (non-budget-forced)
natural-completion subsample of the same reasoning model ($n=28$
pipelines that complete without a forced closure) shows a directionally
similar CASCADE$=75.0\%$ (21/28) and CORRECT$=17.9\%$, consistent with,
not contradicting, the budget-forced numbers, though at too small an
$n$ to treat as independent confirmation.

Under corrected labels, a proposed hidden-state "early-detection" signal
(L2-norm/cosine-drift from steps 1-2, meant to predict step-3 failure
before it happens) collapses to chance on both models (AUROC 0.50-0.53,
5-fold CV) and adds no predictive value beyond one trivial,
non-deployable oracle fact: whether the model's own step-1 answer was
correct (AUROC 0.92 Qwen, 0.91 Mistral, computed from the gold label
itself). A PCA(100)+L2-logistic probe on the full hidden-state vectors
(not just five scalar statistics) recovers a modest but genuine signal
the scalar features miss (AUROC 0.61-0.62 Qwen, 0.57 Mistral,
Mann-Whitney $p<0.000001$/$p=0.000058$) -- reproduced independently via
a teacher-forced multi-layer sweep on the same cached completions
(layers 16/24/32 Qwen, 14/21/28 Mistral), confirming the signal is
real at multiple depths, not an artifact of one arbitrarily-chosen
layer, though for Qwen the inherited layer (L32) is not in fact the
strongest of the three tested (L24 beats it by $+0.033$). The decisive
difficulty-dissociation test -- restricting to pipelines already correct
at step 1, where no step-1 difficulty remains to read -- splits by
model: on Qwen a residual is present uncorrected (AUROC 0.582, $n=726$,
$p=0.012$) but collapses to chance on Mistral (AUROC 0.489, $n=534$,
$p=0.70$), and cross-source transfer is weak on both (Qwen
0.535-0.545; Mistral 0.469-0.511). We do not present the Qwen residual
as robust: it survives Holm-Bonferroni correction across six related
tests by a margin of only $0.0005$, fails plain (non-step-down)
Bonferroni correction across the same six ($0.012\times6=0.072>0.05$),
and this margin is thinner than a measured $0.0076$ reproduction gap
between two independent extractions of the identical quantity (\S3.2).
The honest headline is neither a clean null nor a clean positive:
scalar geometry carries no early-warning value, the fuller
representation carries a real pooled-population signal, but that
signal is substantially -- on Mistral, entirely -- a step-1-difficulty
readout, and the one residual beyond that (Qwen) is suggestive at best,
not a confirmed, correction-surviving early-warning signal.

We also test and refute a contaminated-context account of cascade:
stripping the step-1 answer from the step-2 prompt does not reduce
compounding failure on either model, and if anything increases it (Qwen
+4.4pp, Fisher $p=0.046$; Mistral +4.2pp, $p=0.171$, directionally
replicated but only Qwen's effect itself significant). A per-source
breakdown shows the Qwen-vs-Mistral cascade gap (67.0\% vs.\ 49.3\%) is
not purely a dataset-composition artifact: reweighting either model's
per-source rates to the other's mix leaves most, but not all, of the gap
intact. We do not build absorbing Markov-chain machinery to model this
process, since our own step-transition probabilities move 1-9
percentage points across a three-way-confounded (sample size, dataset
mix, labeling scheme) comparison between an earlier N=600 measurement
and this paper's N=2,500 one -- too unstable to anchor derived
quantities on, though both regimes show strongly persistent ("nearly
absorbing") wrong states above 0.87.

An LLM-judge validation pass (stratified samples of 150 Qwen / 100
Mistral pipelines) shows the surface-form metric this paper relies on
throughout agrees with independent semantic judgment only
moderately-to-weakly. Mistral, judged by a genuinely independent model
(Qwen2.5-3B-Instruct): 57.0\% agreement, $\kappa=0.123$. Qwen, initially
judged by itself -- a self-preference confound caught on audit -- gave
an inflated $\kappa=0.467$; re-judged by an independent model
(Mistral-7B-AWQ), agreement drops to 51.4\% ($\kappa=0.179$), moving
toward the independent judge being more lenient, not less. This
three-way disagreement (surface/self-judged/independently-judged), not
any single number, is the honest label-quality caveat on the headline
taxonomy counts. Judge validation extended to steps 1-2 finds
comparable ($\kappa=0.30$-$0.44$) agreement throughout, and a held-out
human-adjudication pass on step-3 judge-disagreement cases finds the
independent judge itself wrong about as often as right -- most commonly
by crediting fluent, confident fabrications as correct on TruthfulQA's
decline-to-answer trap questions -- so no single measurement here is
simply more trustworthy than the others. A genuine causal test (steering
the residual stream during step-2 generation along the
cascade-predictive direction, vs.\ a random-direction control) produces
no significant shift in step-3 correctness at either tested strength
($p=0.25$ at $\eta=15$; $p=0.625$ at $\eta=30$; $n=50$) -- a clean
causal null, converging with this project's companion papers' own
causal-test conclusions on different tasks. This null replicates at
$n=200$ ($p=0.6875$/$1.000$), alongside a manipulation check confirming
the steering hook reliably perturbs step-2 generation more under the
found direction than the random control (mean divergence
$0.151$ vs.\ $0.125$) -- ruling out an inert intervention as the
explanation for the null.

## 1. Introduction

Multi-step LLM pipelines run generate, critique, then refine. This
pattern is increasingly common in deployed agentic systems. Identifying
how and when these pipelines' failures compound is a practical
governance question. This paper makes four contributions.

First, we build an 8-mode taxonomy over the 3-bit correctness pattern
across pipeline steps. This includes a LATENT mode (correct, correct,
then wrong). LATENT is invisible to any monitor checking only
intermediate outputs.

Second, we deliver a corrected, properly-labeled measurement of how
often each mode occurs, at N=2,500. We give an explicit, disclosed
account of how and why our first labeling pass overstated cascade
dominance.

Third, we scope a result on whether hidden-state geometry provides
early warning of eventual failure. We measure this on a
properly-distributed label set, not a near-degenerate one. Scalar
summary features (L2 norm, cosine drift) are null. A full-vector probe
on the same cached hidden states recovers a modest, non-deployable
signal the scalars miss.

Fourth, we refute the hypothesis that cascade failure is driven by
contaminated context rather than being intrinsic to the model's
behavior under self-refinement. This refutation is replicated and
strengthened under corrected labels.

**Contributions, honestly scoped:**
1. An 8-mode failure taxonomy: a 3-bit correctness pattern across 3
   steps. This is a clean conceptual contribution. It is independent of
   any specific labeling metric.
2. A corrected measurement at N=2,500 (Qwen 2.5 3B) and N=1,200 (Mistral
   7B Instruct), on TruthfulQA + NQ-open. Cascade is 67.04% on Qwen, not
   the 97.28% our first ROUGE-L pass reported. Cascade is 49.33% on
   Mistral. Genuinely-correct-throughout is 25.28% on Qwen and 38.33% on
   Mistral, not 0.76% and 5.00%. We give an explicit, disclosed account
   of the two concrete labeling bugs responsible for the original
   overstatement: ROUGE-L is inappropriate for short-answer QA, and the
   pipeline discarded answer aliases. This is a disclosed correction,
   not a silent revision. We also report a cross-model finding: cascade
   dominance, while real on both models, is partly capability-dependent,
   not a fixed constant of self-refinement. A third, smaller residual
   labeling risk in the corrected metric itself is audited and
   quantified at $\approx$1.2% on Qwen. It is not corrected for (§5).
3. A scoped null result, replicated across both models. Scalar
   hidden-state geometric features -- L2 norm, cosine drift -- provide
   no early-warning signal under corrected labels: AUROC 0.50-0.53 on
   both. The step-1-correctness "baseline" does predict step-3
   correctness well (AUROC 0.92 and 0.91). But this baseline is an
   oracle. It is derived from the gold label and is not a deployable
   signal. **Closed, post-review (fourth round): rather than leave
   "these five scalars don't exhaust hidden-state geometry" as an
   unresolved caveat, we ran a PCA(100)+L2-logistic probe on the full
   step-1/step-2 hidden-state vectors already cached from the original
   extraction (no new inference).** This recovers a non-artifactual
   signal the scalar features miss: AUROC 0.61-0.62 on Qwen and 0.57 on
   Mistral. To test significance, we used a Mann-Whitney $U$-test on
   out-of-fold predictions: $p<0.000001$ on Qwen, $p=0.000058$ on
   Mistral. This is corrected, in round-5 review, from an earlier
   invalid fold-SD heuristic. That heuristic additionally mis-stated
   Mistral's effect as "several SD" when it was 2.3. **We then ran the
   decisive difficulty-dissociation test (round-5 review): restrict to
   pipelines already correct at step 1, where no step-1 difficulty
   remains to read.** The result is model-dependent, not uniform. On
   Qwen the signal survives, weakened: AUROC 0.582, $n=726$, $p=0.012$.
   On Mistral it collapses to chance: AUROC 0.489, $n=534$, $p=0.70$.
   Mistral's full-vector "signal" is therefore substantially explained
   by step-1 difficulty alone. Qwen retains a small, only
   barely-significant residual. Cross-source transfer (train
   TruthfulQA/test NQ-open and vice versa) is weak on both models:
   0.535-0.545 Qwen; 0.469-0.511 Mistral. This indicates poor
   generalization across dataset source, regardless of model. The
   honest summary is therefore not two-layered but three-layered. The
   scalar summary statistics are null. The fuller representation
   carries a real, significant pooled-population signal. The decisive
   test shows that signal is mostly-to-entirely a difficulty readout
   depending on model, with weak cross-source transfer either way. This
   is a materially more modest and more model-dependent finding than
   "the scalars are null but the full vectors are not" alone would
   suggest. We report it at this full resolution rather than stopping
   at the pooled-population result.
4. A context-ablation experiment refuting the "context contamination"
   hypothesis for compounding failure. This is replicated directionally
   under corrected labels on both models: context-free AMPLIFIED rate is
   4.4 and 4.2 percentage points higher than standard, respectively.
   Only the Qwen effect clears significance on its own: Fisher exact
   p=0.046 on 8/250 vs. 19/250; Mistral p=0.171 on 2/120 vs. 7/120. This
   is a close quantitative cross-model match in magnitude, but not a
   doubly-significant replication.
5. An explicit decision not to build absorbing-Markov-chain machinery
   around the step-to-step transition probability. This decision is
   informed by our own data's instability and by concurrent literature.
   We cite that work as related, rather than compete with it.

## 2. Methods

### 2.1 Pipeline and taxonomy

The pipeline runs three steps (Qwen 2.5 3B Instruct). Step 1 answers a
question. Step 2 reviews and, optionally, corrects the step-1 answer,
given the question and step-1 text. Step 3 gives a final answer, given
all prior context.

The 8-mode taxonomy enumerates all 2^3 correctness patterns across these
three steps. Each bit marks wrong at that step: CORRECT (000), ISOLATED
(100), SELF_HEAL (010), LATE_RECOVERY (110), LATENT (001), AMPLIFIED
(011), OSCILLATING (101), CASCADE (111). LATENT and SELF_HEAL are
specifically invisible to any governance system that only checks
intermediate-step output text. Both modes look "clean" at the step
before the one that matters.

### 2.2 The labeling correction (disclosed, not silent)

**Original labeling** used ROUGE-L F-measure against a single
`best_answer` string, at threshold 0.3. This followed the convention
used in prior small-scale work in this arc.

**Two concrete problems drove the overstated cascade rate.** Independent
adversarial review identified both; we confirmed them directly against
the raw data. First, ROUGE-L is a summarization-style metric. It is
poorly suited to NQ-open's short canonical answers ("Paris", "1969"). It
structurally under-registers correctness, regardless of whether the
generated answer is right. Second, the original pipeline-generation
script kept only the first element of NQ-open's answer-alias list
(`x["answer"][0]`). This discarded valid alternative phrasings for 9.1%
of NQ-open questions we checked (153/1,683 in our specific sample).

**Corrected labeling** fixes both problems. For NQ-open, we recovered
the full alias list per question, reloading `nq_open` with the identical
seed=42 shuffle used in generation. We score each generated answer via
SQuAD-style normalized exact-match/F1 against all aliases: a match
requires the normalized generated text to equal or contain any
normalized alias, or token-F1 against the best-matching alias to exceed
0.5. For TruthfulQA, we use multi-reference Jaccard word-overlap against
both the correct-answer and incorrect-answer lists, at threshold 0.12.
This matches the more careful convention already used in a companion
project (mech-int/HaRP), rather than a single-reference ROUGE-L
comparison.

Both metrics are re-applied to the exact same, already-generated
completions. No new model inference was run to produce this correction.

### 2.3 Early-detection classifier

The early-detection classifier measures whether hidden-state geometry at
steps 1-2 predicts eventual step-3 failure. Features are the L2 norm of
the last-token hidden state at layer 32 (~89% depth) at steps 1 and 2,
plus their difference and cosine similarity. The target is
`1 - relabel_label_s3`. We use 5-fold cross-validated logistic
regression.

A "late" baseline uses only the step-3 hidden-state norm. This is
available only after generation completes. We include it for
comparison, not as a deployable signal.

An output-level "baseline" uses the step-1 correctness label
(`relabel_label_s1`) alone. **This is an oracle feature, not a
deployable one.** `relabel_label_s1` is computed by comparing the
model's step-1 generation against the gold reference answer. Using it
as a predictive feature presupposes access to the ground truth. A real
deployment would not yet have that ground truth. We report it as an
upper-bound/sanity-check signal, not as a proposed early-exit heuristic.

### 2.4 Context ablation (P4)

This experiment tests the contamination hypothesis: does stripping a
wrong step-1 answer from step 2's context reduce compounding failure? We
run a matched 250-question subset twice. In the standard condition, step
2 sees the step-1 answer. In the context-free condition, step 2 reviews
without seeing the step-1 answer; prompt framing is otherwise identical.
We measure the rate of AMPLIFIED failures -- correct at step 1, wrong at
steps 2 and 3 -- under each condition. This tests Kumar et al.'s
SCoRe-adjacent contamination hypothesis.

## 3. Results

### 3.1 Corrected taxonomy (N=2,500, Qwen 2.5 3B)$^\dagger$

$^\dagger$The "Corrected" column carries an audited $\approx$1.2%
residual substring-match mislabeling risk on CASCADE/CORRECT. This
slightly overstates CORRECT (§5). This risk is not present in the
"Original" column. That column has its own, larger, disclosed problems
instead.

| Mode | Original (ROUGE-L) | Corrected (SQuAD EM/F1 + multi-ref) |
|---|---|---|
| CASCADE | 97.28% (2,432) | **67.04% (1,676)** |
| CORRECT | 0.76% (19) | **25.28% (632)** |
| LATE_RECOVERY | 0.80% (20) | 2.48% (62) |
| LATENT | 0.16% (4) | **1.84% (46)** |
| AMPLIFIED | 0.52% (13) | 1.12% (28) |
| ISOLATED | 0.08% (2) | 0.96% (24) |
| SELF_HEAL | 0.36% (9) | 0.80% (20) |
| OSCILLATING | 0.04% (1) | 0.48% (12) |

Per-dataset step-1 correctness moves in the same direction. Corrected
vs. original: TruthfulQA is 32.44% vs. 4.77%; NQ-open is 27.39% vs.
0.36%. The gap is largest on NQ-open. This matches expectation, given
ROUGE-L's specific unsuitability for short canonical answers. **Cascade
is still the plurality/majority outcome. It is still the paper's central
empirical fact. But "near-total collapse" was substantially a metric
artifact. We report the corrected number as the paper's true headline,
not the original one.** LATENT moves from an unusable n=4 to a workable
n=46. It is still too rare for tight per-mode confidence intervals. We
report it descriptively.

### 3.2 Early-detection: a scoped null, not a clean one

| Signal | AUROC (5-fold CV) |
|---|---|
| Late (step-3 hidden-state norm only) | 0.503 |
| Early, 3-feature (step-1/2 norms + drift) | 0.507 |
| Early, 5-feature (+ cosine similarity) | 0.514 |
| **Output-level baseline (step-1 correctness alone)** | **0.921** |

Every scalar geometric feature set is statistically indistinguishable
from chance. This directly contradicts our original, ROUGE-L-labeled
pass, which reported AUROC 0.610-0.662. That earlier result was very
likely an artifact of predicting a near-degenerate, ~98%-positive label.
Under a near-degenerate label, almost any classifier can look like it is
"detecting" something. Once measured on a properly-distributed label
set, the picture changes. We find **no evidence that the
L2-norm/cosine-drift scalar features tested here add early-warning
value**, beyond the trivial fact of whether the model's own step-1
output was correct.

**Closed, post-review (fourth round): rather than leave "a full-vector
probe was not run" as an open caveat, we ran one, on data already cached
from the original extraction** (`code/05_full_vector_probe.py`, no new
model inference). This probe measures whether the raw hidden-state
vectors, not just scalar summaries of them, carry early-warning signal.
We ran PCA(100)+L2-logistic on the raw step-1/step-2 hidden-state
vectors themselves -- 2048-d on Qwen, 4096-d on Mistral:

| Feature | Qwen AUROC | Mistral AUROC |
|---|---|---|
| $h_{s1}$ (full vector) | 0.6116 $\pm$ 0.0206 | 0.5728 $\pm$ 0.0161 |
| $h_{s2}$ (full vector) | 0.5916 $\pm$ 0.0210 | 0.5326 $\pm$ 0.0143 |
| concat($h_{s1}, h_{s2}$) | **0.6169 $\pm$ 0.0321** | **0.5660 $\pm$ 0.0282** |
| $h_{s2}-h_{s1}$ (full-vector drift) | 0.5488 $\pm$ 0.0186 | 0.5587 $\pm$ 0.0118 |

This is a genuinely different result from the scalar features. It is
not attributable to the degenerate-label artifact that inflated the
original ROUGE-L pass: `relabel_label_s3` is a 70/30 (Qwen) and 53/47
(Mistral) split, not ~98/2.

**Significance, corrected (round-5 review): dividing (AUROC$-0.5$) by
the fold-standard-deviation shown above is not a valid test.** Five CV
folds are not five independent draws under $H_0$. The "3-6 SD" claim
this produced was itself wrong for Mistral: 2.3 SD, not "several." We
replace this test with a Mann-Whitney $U$-test on out-of-fold
predictions from a single `cross_val_predict` pass. This test is exact.
It is equivalent to testing AUROC against 0.5 directly, since
$\text{AUROC} = U/(n_{\text{pos}} n_{\text{neg}})$. concat($h_{s1},
h_{s2}$) reaches $p<0.000001$ on Qwen and $p=0.000058$ on Mistral. Both
are genuinely significant -- more decisively than the flawed heuristic
suggested for Mistral.

**The decisive test (round-5 review) asks: is this a cascade-specific
signal, or a step-1-difficulty readout?** $h_{s1}$ alone (0.6116 Qwen)
already beats the dynamic drift feature $h_{s2}-h_{s1}$ (0.5488 Qwen) by
a wide margin. This is exactly the signature of a static, pre-refinement
readout, not something tracking the refinement process.

We ran the free, decisive test this implies. We restricted the probe to
pipelines already correct at step 1 (`relabel_label_s1==1`,
$n=726/2500$ Qwen, $n=534/1200$ Mistral). By construction, no step-1
difficulty variation remains to read in this subset. Only one question
remains: which of the already-correct-at-step-1 pipelines still fail by
step 3. This is the LATENT/SELF_HEAL boundary the taxonomy is built
around; within this subset, $\text{wrong\_s3\_rate}=10.2\%$ Qwen /
$8.1\%$ Mistral.

**The result splits by model rather than settling the question
cleanly.** On Qwen, concat($h_{s1},h_{s2}$) survives, weakened, at
AUROC$=0.582$ ($p=0.012$, Mann-Whitney). On Mistral it collapses to
AUROC$=0.489$ ($p=0.70$) -- indistinguishable from chance. **Mistral's
pooled-population full-vector signal is therefore substantially, perhaps
entirely, a step-1-difficulty readout. Qwen retains a small, only
barely-significant residual beyond difficulty.**

A cross-source transfer check adds a second, independent line of
evidence that the signal is fragile. We trained on TruthfulQA and tested
on NQ-open, and vice versa, using the same cached vectors. Transfer
AUROC is 0.535-0.545 on Qwen and 0.469-0.511 on Mistral -- at or below
chance in one direction. Neither model's full-vector probe generalizes
cleanly across dataset source.

**We do not round any of this up to a deployable signal or down to a
clean null.** The honest, three-layered summary is this. The scalar
summary statistics we tested first are null. The fuller representation
carries a real, Mann-Whitney-significant signal in the pooled
population. The decisive difficulty-dissociation test shows that pooled
signal is mostly-to-entirely explained by step-1 difficulty, with only a
weak, model-specific, poorly-transferring residual. This is a materially
more modest finding than "the scalars are null but the full vectors are
not" alone would suggest. We report it at this resolution, rather than
stopping at the pooled result (`code/06_difficulty_confound_and_significance.py`,
full numbers in `results/difficulty_confound_and_significance.json`).

**Multiple-comparison correction across this paper's six significance
tests, added final-audit pass.** This paper reports six p-values total.
These are: the pooled full-vector Mann-Whitney tests (Qwen $p<0.000001$,
Mistral $p=0.000058$); the difficulty-dissociation Mann-Whitney tests
just above (Qwen $p=0.012$, Mistral $p=0.70$); and the context-ablation
Fisher-exact tests (§3.4/§4; Qwen $p=0.046$, Mistral $p=0.171$). None of
these were pre-registered as a single family. A reviewer could still
reasonably treat them as related enough to warrant a joint correction.

We apply Holm-Bonferroni: ranked ascending, threshold $\alpha/(6-k+1)$
at rank $k$. The two pooled tests survive comfortably -- ranks 1-2,
thresholds $0.0083$/$0.01$, actual $p\ll10^{-4}$ both. Qwen's
difficulty-dissociation result (rank 3, $p=0.012$, threshold $0.0125$)
*barely* survives, by a margin of $0.0005$. It would flip to
non-significant under a marginally larger test family or a stricter,
non-step-down correction. Qwen's context-ablation result (rank 4,
$p=0.046$, threshold $0.0167$) does *not* survive. Neither do the two
already-non-significant Mistral tests. Under plain, non-step-down
Bonferroni across the same six tests ($p \times 6$), Qwen's
difficulty-dissociation result also fails: $0.012\times6=0.072>0.05$.

**We do not present the Qwen difficulty-dissociation residual (AUROC
0.582) as a result that robustly survives multiple-comparison
correction.** It is significant uncorrected. It survives one reasonable
step-down correction by a narrow margin. It fails a stricter or
differently-scoped one. It should be read as suggestive, not confirmed.
This directly qualifies the "weak, only barely-significant residual
beyond difficulty" language above: barely-significant, uncorrected, is
now also barely-corrected, in either direction, depending on the
correction family chosen.

**Closed, elite-review follow-up: this margin is thinner than the
paper's own extraction-pipeline noise floor.** The layer-sweep
robustness check (\S5) re-derives this identical quantity (concat
$h_{s1},h_{s2}$, $L32$, $n=726$, same population) via teacher-forced
reconstruction rather than the original cached generation-time hidden
states, and gets pooled-out-of-fold AUROC $0.5877$
(`results/layer_sweep_results.json`), not $0.5801$
(`results/difficulty_confound_and_significance.json`) -- a
$0.0076$ gap between two extractions of what this paper's own §5 text
asserts "reproduces the same hidden states the original generation-time
forward hook captured." That assertion does not hold exactly: it holds
qualitatively (both extractions agree the signal is real and in the
same range) but not to the precision the $0.0005$ Holm-Bonferroni
margin above would require. We do not know which extraction is closer
to ground truth, and do not have a principled way to choose. The
honest reading is that the Qwen residual's significance, uncorrected or
corrected, is not robust to extraction-pipeline noise of a magnitude
this paper's own alternative measurement pipeline already demonstrates
exists.

**The output-level "baseline" (AUROC 0.921) is an oracle. It is also
substantially a restatement of a structural fact about the taxonomy, not
evidence of nontrivial predictive machinery.** `relabel_label_s1`
requires the gold answer. This number cannot be used as a deployable
early-exit signal.

Decomposing it makes this concrete. Among the 1,762 pipelines wrong at
step 3, 95.8% were already wrong at step 1 (`P(wrong@1|wrong@3)` =
1,688/1,762). Among the 738 pipelines correct at step 3, 88.3% were
already correct at step 1 (`P(correct@1|correct@3)` = 652/738). The
average of these two rates is 92.1%. This matches the reported AUROC,
0.921, to three decimal places. For a binary predictor, AUROC equals the
average of sensitivity and specificity -- that is why the two numbers
match.

This number is mostly a restatement of one fact: step-1 and step-3
correctness agree on 93.6% of pipelines (2,340/2,500: CORRECT +
SELF_HEAL + OSCILLATING + CASCADE -- every mode where step-1 and step-3
correctness match, not only the narrower CASCADE-or-CORRECT diagonal
specifically, 92.3% / 2,308/2,500). It is not a demonstration of a
sophisticated predictive relationship.

The honest headline for this section is: **no deployable early-warning
signal was found.** The full-vector probe above recovers a real but
modest (0.57-0.62) signal that scalar features miss. That signal is
still nowhere near strong enough to deploy. Nothing here overturns the
oracle-vs-geometry distinction just made.

### 3.3 Context ablation: contamination hypothesis refuted, more strongly under correction

| Condition | AMPLIFIED rate (corrected) | AMPLIFIED rate (original, ROUGE-L) |
|---|---|---|
| Standard (step 2 sees step-1 answer) | 3.20% (8/250) | 1.20% (3/250) |
| Context-free (step 2 does not see step-1 answer) | 7.60% (19/250) | 1.60% (4/250) |
| Drop | **-4.40pp** | -0.40pp |

**Correction (round-5 review): the "original" column was previously
misreported as 0.67%/1.33%/-0.67pp.** These are non-integer counts on
$n=250$ that do not correspond to any real run. The values above --
1.20%=3/250, 1.60%=4/250, -0.40pp -- are the actual numbers in
`results/qwen/paper3-qwen-run1.log` ("AMPLIFIED -- Standard: 1.20% |
Context-free: 1.60% Drop: -0.40pp"). We reconciled these directly
against the raw log, rather than re-deriving them from memory.

Stripping the "contaminating" step-1 context from step 2 does not reduce
AMPLIFIED-mode failures under either labeling. Under corrected labels it
more than doubles the rate: 3.20% → 7.60% (two-sided Fisher exact test
on the 8/250 vs. 19/250 contingency table, p=0.046). Under the original
labels it also increases: 1.20% → 1.60% (3/250 vs. 4/250), though the
small counts do not support an independent test. This is consistent with
cascade being an intrinsic property of the model's behavior under
self-refinement, rather than an artifact of carrying forward a specific
wrong answer. It argues against context-stripping as a mitigation
strategy.

The Mistral replication of this ablation (§3.4) is directionally
identical. It does not itself clear significance at this sample size
(p=0.171 on 2/120 vs. 7/120). We treat the cross-model quantitative
closeness in magnitude (-4.4pp vs. -4.2pp) as suggestive of the same
underlying effect, not as two independent significant replications.
**We flag plainly that this entire refutation rests on small event
counts: 8 vs. 19 on Qwen, 2 vs. 7 on Mistral. It is convincing in
direction and in two-model consistency, but not in raw statistical
mass.**

### 3.4 Cross-model replication (Mistral 7B Instruct v0.3, N=1,200)

We replicate the full method on a second model, as an independent
cross-model check, not merely a larger-N rerun of the same model. We ran
the identical pipeline, labeling correction, detection classifier, and
context ablation on Mistral 7B Instruct v0.3: N=1,200 (817 TruthfulQA +
383 NQ-open), target layer 28/32 (~87.5% depth).

**Taxonomy.** Mistral's original ROUGE-L labeling was already less
extreme than Qwen's: CASCADE 83.33% vs. Qwen's 97.28%; CORRECT 5.00% vs.
0.76%. This itself is informative -- the labeling artifact's severity is
not identical across models, even before correction. Under the
corrected metric:

| Mode | Qwen 2.5 3B (N=2,500), Wilson 95% CI | Mistral 7B (N=1,200), Wilson 95% CI |
|---|---|---|
| CASCADE | 67.04% (1,676), [65.2%, 68.9%] | **49.33% (592), [46.5%, 52.2%]** |
| CORRECT | 25.28% (632), [23.6%, 27.0%] | **38.33% (460), [35.6%, 41.1%]** |
| LATE_RECOVERY | 2.48% (62), [1.9%, 3.2%] | 3.25% (39), [2.4%, 4.4%] |
| SELF_HEAL | 0.80% (20), [0.5%, 1.2%] | 2.58% (31), [1.8%, 3.6%] |
| ISOLATED | 0.96% (24), [0.7%, 1.4%] | 2.42% (29), [1.7%, 3.5%] |
| AMPLIFIED | 1.12% (28), [0.8%, 1.6%] | 2.08% (25), [1.4%, 3.1%] |
| LATENT | 1.84% (46), [1.4%, 2.5%] | 1.50% (18), [1.0%, 2.4%] |
| OSCILLATING | 0.48% (12), [0.3%, 0.8%] | 0.50% (6), [0.2%, 1.1%] |

**Closed, elite-review follow-up: CIs added, not previously reported for
these headline percentages** (`results/taxonomy_bootstrap_cis.json`).
CASCADE and CORRECT's CIs do not overlap between models (CASCADE:
[65.2, 68.9] vs.\ [46.5, 52.2]; CORRECT: [23.6, 27.0] vs.\ [35.6,
41.1]) -- the cross-model gap is not sampling noise at these sample
sizes. The rarer modes' CIs are wide relative to their point estimates
(e.g.\ Qwen OSCILLATING, $n=12$: [0.3\%, 0.8\%], a CI width comparable
to the estimate itself), reinforcing the existing "small $n$,
descriptive only" caveat for LATENT and OSCILLATING with an explicit
interval rather than only a qualitative warning.

Cascade dominance replicates directionally: it is still the plurality
outcome on both models. But it is meaningfully weaker on Mistral: 49.33%
vs. 67.04%. Mistral shows a correspondingly higher genuine-correctness
rate: 38.33% vs. 25.28%. Per-source step-1 correctness under the
corrected metric is similar across models: Mistral is 44.31% TruthfulQA,
44.91% NQ-open; Qwen is 32.44% TruthfulQA, 27.39% NQ-open.

**Correction, added post-review (second round): the overall
67.04%-vs-49.33% gap is confounded by dataset composition.** Qwen's
N=2,500 is 67.3% NQ-open. Mistral's N=1,200 is only 31.9% NQ-open. We
check this directly with a per-source CASCADE-rate breakdown:

| Source | Qwen CASCADE% (n) | Mistral CASCADE% (n) |
|---|---|---|
| TruthfulQA | 57.77% (472/817) | 47.25% (386/817) |
| NQ-open | 71.54% (1,204/1,683) | 53.79% (206/383) |

NQ-open shows a higher cascade rate than TruthfulQA on both models. This
confirms the dataset-mix confound is real: Qwen's test set is weighted
more toward the harder source.

Reweighting each model's per-source rates to the other model's dataset
mix quantifies how much of the gap survives. Reweighting Mistral's rates
to Qwen's mix (67.3% NQ-open) gives a matched-mix Mistral cascade rate of
51.6%, vs. Qwen's actual 67.0% -- a 15.4-point residual gap. Reweighting
Qwen's rates to Mistral's mix (31.9% NQ-open) gives a matched-mix Qwen
rate of 62.2%, vs. Mistral's actual 49.3% -- a 12.9-point residual gap.
Either direction, dataset composition explains roughly 2-5 percentage
points of the original 17.7-point gap. A double-digit gap remains under
a matched mix.

We therefore retain, in a narrower form, the claim that cascade
dominance is partly model-capability-dependent: the majority of the gap
survives composition-matching. We withdraw any implication that the
raw, unweighted 67.0%-vs-49.3% comparison is itself a clean capability
measurement. Some of it is dataset mix. Most of it is not.

**Early-detection: the scalar null replicates; so does the full-vector
signal.** Late AUROC is 0.519; early 3-feature is 0.527; early 5-feature
is 0.513. All are chance-level, matching Qwen's null result
(0.503-0.514) closely.

The output-level "baseline" -- step-1 correctness predicting step-3
correctness -- is again strong: AUROC 0.906, vs. Qwen's 0.921. As on
Qwen, this is an oracle feature requiring the gold label, not a
deployable signal.

The full-vector probe (Table in §3.2) also replicates on Mistral:
concat($h_{s1},h_{s2}$) reaches AUROC 0.5660 $\pm$ 0.0282. This is
weaker than Qwen's 0.6169, but in the same direction, and Mann-Whitney
significant ($p=0.000058$) on $n=1{,}200$.

**But the decisive difficulty-dissociation test does not replicate.**
Restricted to the $n=534$ pipelines already correct at step 1, Mistral's
full-vector AUROC collapses to $0.489$ ($p=0.70$, indistinguishable from
chance). Qwen's corresponding subset ($n=726$) retains a weak but
significant $0.582$ ($p=0.012$). Cross-source transfer is weak on both
models: 0.469-0.511 Mistral; 0.535-0.545 Qwen.

**The scalar-summary null replicates cleanly across both models. The
full-vector signal does not replicate as a cascade-specific finding.**
On Mistral it is better characterized as a step-1-difficulty readout
that happens to also show up in the pooled population. Only on Qwen
does a small, barely-significant residual survive the harder test.

**Context ablation replicates directionally, but not significantly.**

| Condition | AMPLIFIED rate (corrected) | AMPLIFIED rate (original, ROUGE-L) |
|---|---|---|
| Standard (step 2 sees step-1 answer) | 1.67% (2/120) | 0.83% (1/120) |
| Context-free (step 2 does not see step-1 answer) | 5.83% (7/120) | 0.83% (1/120) |
| Drop | **-4.17pp** | +0.00pp |

Standard AMPLIFIED is 1.67% (2/120); context-free is 5.83% (7/120) -- a
drop of **-4.17pp**. This is strikingly close to Qwen's -4.40pp in
magnitude. But at n=120 per arm, this specific contrast is not itself
statistically significant: two-sided Fisher exact test on 2/120 vs.
7/120, p=0.171.

**Added final-audit pass, closing an asymmetry in this paper's own
"report both" policy.** Mistral's original, pre-correction ROUGE-L
numbers appear in `results/mistral/paper3-mistral-run1.log` ("AMPLIFIED
--- Standard: 0.83% | Context-free: 0.83% Drop: +0.00pp"). These were
previously never reported here alongside Qwen's equivalent
original-column numbers (§3.3).

Unlike Qwen -- where the original labeling already showed a small
increase under context-stripping (+1.60%$-$1.20%, i.e. 0.40pp) --
Mistral's original-ROUGE-L labeling shows literally no movement at all:
1/120 in both arms, at this sample size. Both counts are too small to
support any claim beyond "not detectably different." This does not
change this paper's reading of the corrected-label result (Fisher exact,
2/120 vs. 7/120, p=0.171, discussed above). The original-label numbers
on both models are underpowered artifacts of the same $n=120$ sample,
not an independent check.

Context-stripping does not reduce compounding failure on either model.
If anything, both show a numerically higher AMPLIFIED rate without
step-1 context. Qwen's effect is significant; Mistral's is
same-direction, similar-magnitude, but individually non-significant. We
read this close quantitative match as consistent with one real
underlying phenomenon, observed at two different power levels -- not as
two independent significant replications. This distinction is worth
preserving precisely, rather than rounding up to "replicates."

### 3.5 Causal intervention: does steering $h_{s1}$ change step-3 outcome?

**Added, final-audit pass, closing the exact gap every program-chair
review of this paper flagged: §3.2's early-warning signal is a passive
probe, with no causal write-side test.** Because
\texttt{fail-chain/src/pipeline.py}'s three steps are separate
\texttt{model.generate()} calls on the growing TEXT context, not a
continued forward pass, step 2 and step 3 never see step 1's raw
hidden state directly -- only its generated text. So the mechanistically
meaningful causal question is not "patch $h_{s1}$ between two
pipelines" (which corresponds to nothing the real pipeline does) but:
does \emph{steering} the model's residual stream, during step-2
generation, along a direction constructed from the same population and
outcome §3.2's passive probe targets, actually change what step 2
generates -- and does that change propagate to step 3's correctness?
\textbf{Elite-review follow-up, stated precisely rather than left
implicit: this direction is not literally §3.2's probe direction.} It
is a difference-of-means vector fit on $h_{s1}$ alone
(\texttt{mu\_success - mu\_fail}, \texttt{run\_causal\_step2\_steering.py}),
not the PCA(100)+logistic decision direction fit on
concat($h_{s1},h_{s2}$) that produces §3.2's reported AUROCs -- a
simpler, single-step object, chosen because $h_{s2}$ does not yet exist
before step-2 generation happens. It is also fit directly against the
gold-derived \texttt{relabel\_label\_s3} label, the same oracle label
§3.2 and §5 flag as "not deployable... requires knowing the correct
answer" when discussing the output-level baseline; that same caveat
applies here and is stated explicitly now rather than left for a reader
to infer from the code. A null causal result for this vector is
evidence about this specific, oracle-constructed, single-step
direction, not a demonstration that §3.2's actual validated probe
direction lacks causal purchase.

We fit this difference-of-means direction (eventually-correct minus
eventually-wrong, restricted to pipelines already correct at step 1 --
the identical population §3.2's decisive test uses) on 676 held-out
training pipelines, then tested it on 50 held-out pipelines: for each,
we rebuilt the exact step-2 prompt from the actual recorded step-1
answer, regenerated step 2 with an additive steering hook at layer 32
injecting either the found direction or a random direction of the same
norm, then generated step 3 normally from the (steered) step-2 output
and rescored it with the identical corrected metric used throughout
this paper. \textbf{Result: no significant causal effect at either
strength tested.} At $\eta=15$: found-correct-rate $=82\%$,
random-correct-rate $=88\%$ (random \emph{higher}), 0 found-only vs.\ 3
random-only discordant pipelines, McNemar $p=0.25$. At $\eta=30$:
found-correct-rate $=92\%$, random-correct-rate $=88\%$ (found higher
this time), 3 found-only vs.\ 1 random-only, McNemar $p=0.625$. All
50 test pipelines produced a scoreable step-3 output at both strengths
(no exclusions). \textbf{The found direction does not consistently
beat the random-direction control in either magnitude or significance
across the two tested strengths} -- this is a clean causal null, not a
directionally-suggestive-but-underpowered one: at $\eta=15$ random
numerically wins, at $\eta=30$ found numerically wins, and neither
McNemar test approaches significance.

This directly closes the reviewers' gap: the passive probe's
early-warning signal, whatever its statistical status in §3.2, does not
translate into a causal lever an intervention can exploit at
generation time, at least not via a dense mean-difference direction at
this sample size. This is the same qualitative conclusion Paper 1 and
Paper 4 of this project reach for their own causal tests (dense
FFN-patching and Fisher-LDA steering, respectively, both null) --
a fourth independent causal test, on a different task (multi-step
cascade prediction, not single-turn hallucination), converging on the
same answer: a passively-discovered direction correlated with an
outcome does not, by default, carry causal control over that outcome
(\texttt{kaggle\_kernels/paper3-causal-steering/run\_causal\_step2\_steering.py},
\texttt{results/causal\_step2\_steering\_intervention.json}).

**Closed, elite-review follow-up: this test was re-run at $n=200$
(4$\times$ the original), with a manipulation check confirming the
intervention itself, not only its downstream effect.** Using the same
fitting procedure and layer (the direction itself is refit on this
run's own training split, not reused verbatim), on 200 held-out
pipelines disjoint from the 526 pipelines used to fit it, we
additionally generated an unsteered
baseline step-2 completion for each item and measured mean word-overlap
divergence from that baseline under found- vs.\ random-direction
steering. The found direction produces reliably more divergence from
the unsteered baseline than the random direction at both strengths
($\alpha=15$: mean divergence $0.121$ vs.\ $0.098$; $\alpha=30$:
$0.180$ vs.\ $0.151$; overall $0.151$ vs.\ $0.125$) -- confirming the
steering hook meaningfully perturbs step-2 generation, so the causal
null is not an artifact of an inert intervention. Despite this
confirmed perturbation, step-3 correctness shows no significant
difference at either strength: at $\alpha=15$, found-correct-rate
$=92.5\%$ vs.\ random $93.5\%$ (random again numerically higher), 2
found-only vs.\ 4 random-only discordant pipelines, McNemar
$p=0.6875$; at $\alpha=30$, found $=92.0\%$ vs.\ random $92.5\%$, 4
found-only vs.\ 5 random-only, $p=1.000$. The direction that reliably
perturbs generation more does not reliably improve step-3 correctness
more -- at 4$\times$ the sample size, the null from the $n=50$ test
above replicates, now with direct evidence that the intervention
itself was not inert
(`kaggle_kernels/paper3-causal-steering-scaled/run_causal_steering_scaled.py`,
`results/causal_step2_steering_scaled.json`).

### 3.6 Reasoning-model extension: a same-scale, budget-forced comparison

**Added, final-audit pass, testing whether cascade dynamics change
qualitatively on a model with explicit deliberation, not just
quantitatively across two non-reasoning models.** A first attempt at
this extension (400-token generation budget per step,
\texttt{deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B} alone) failed
honestly rather than quietly: 272 of 300 pipelines (90.7\%) never
closed their \texttt{<think>...</think>} block within budget and had to
be excluded, leaving an uninterpretably small $n=28$ with a real
selection-effect risk (whichever pipelines happened to reason quickly
survived to be scored). We fixed both problems this exposed, rather
than simply re-running with a bigger budget and a smaller $N$:

1. \textbf{Budget forcing} (Muennighoff et al. 2025, "s1: Simple
Test-Time Scaling"): if a step's \texttt{<think>} block is not closed
within a 400-token budget, we forcibly append \texttt{</think>} to
whatever reasoning was generated, then continue generation for a
separate 100-token answer budget regardless. This is a real, published
test-time-scaling technique, not an ad hoc truncation -- and it
guarantees every pipeline produces a scoreable answer. \textbf{Result:
0 of 300 pipelines excluded this time} (vs.\ 272/300 previously), at
the cost of disclosure: forced closure was needed on 576 of 900
step-level generations (64.0\%), meaning the majority of this model's
reasoning traces were cut short and forced to answer rather than
completing naturally -- a real, acknowledged limitation of the
budget-forced numbers below, not a clean natural-completion result.
**Closed, elite-review follow-up: the discarded first-attempt
natural-completion subsample is reported here as a free (if
confounded) dose-response check, rather than left unreported now that
budget forcing supersedes it.** The 28/300 pipelines that completed
naturally under the original (unforced, 400-token, no separate answer
budget) attempt show CASCADE $75.0\%$ (21/28) and CORRECT $17.9\%$
(5/28) -- both directionally milder than the full budget-forced
population's $81.0\%$/$12.3\%$. This is consistent with, but cannot
distinguish between, two explanations: forcing itself degrades answer
quality, or the questions that happen to complete naturally within
budget are an easier, non-random subset of the pool to begin with (most
likely both contribute, and this $n=28$ comparison cannot separate
them). We report the direction and magnitude rather than only the
qualitative caveat, but do not read it as resolving the confound --
only as bounding it, since even this favorable, survivorship-biased
subsample does not show a qualitatively different pattern (cascade
still the plurality outcome) than the full forced population.
2. \textbf{A same-scale, non-reasoning control.} The original
comparison (DeepSeek-R1-Distill-Qwen-1.5B vs.\ Qwen 2.5 3B / Mistral
7B, already reported) confounds "reasoning paradigm" with "much
smaller model." We additionally ran plain
\texttt{Qwen2.5-1.5B-Instruct} (no thinking, the same base architecture
DeepSeek's model is distilled from) on the identical 300-question pool,
holding scale fixed.

\textbf{Result: the reasoning model cascades substantially more often
than its same-scale non-reasoning counterpart, but is not less likely
to end up correct.} \textbf{Closed, elite-review follow-up: the
original per-mode counts below conflated ISOLATED (wrong$\to$correct$\to$correct)
with SELF\_HEAL (correct$\to$wrong$\to$correct) under a single
"SELF\_HEAL" label, unlike the 8-mode classifier used for this paper's
main N=2,500/1,200 taxonomy (\S2.1), which correctly separates them. We
re-derived the correct split directly from this run's own saved
per-item step-1/2/3 labels (no new generation needed) and report the
corrected counts here.} Budget-forced DeepSeek-R1-Distill-Qwen-1.5B:
CASCADE $81.00\%$ (243/300), CORRECT $12.33\%$ (37/300), AMPLIFIED
$2.67\%$ (8/300), LATE\_RECOVERY $1.33\%$ (4/300), SELF\_HEAL $1.00\%$
(3/300), ISOLATED $0.33\%$ (1/300), LATENT $0.67\%$ (2/300), OSCILLATING
$0.67\%$ (2/300). Qwen2.5-1.5B-Instruct (non-reasoning, same scale):
CASCADE $67.33\%$ (202/300), CORRECT $13.00\%$ (39/300), AMPLIFIED
$10.33\%$ (31/300), ISOLATED $3.33\%$ (10/300), SELF\_HEAL $2.33\%$
(7/300), LATENT $1.67\%$ (5/300), OSCILLATING $1.33\%$ (4/300),
LATE\_RECOVERY $0.67\%$ (2/300). CORRECT rates are nearly identical
between the two models ($12.33\%$ vs.\ $13.00\%$, a $0.67$pp
difference); the gap is entirely in cascade rate ($+13.67$pp) and,
correspondingly, in the transitional modes -- the non-reasoning model
shows roughly $4\times$ more AMPLIFIED (31 vs.\ 8) and, combining the
two recovery-adjacent modes, more ISOLATED+SELF\_HEAL (17 vs.\ 4) than
the reasoning model, though the corrected split shows this is
predominantly an ISOLATED-mode (initial-error, then self-corrected)
difference (10 vs.\ 1), not a SELF\_HEAL-mode (mid-pipeline corruption,
then recovered) one (7 vs.\ 3). \textbf{The qualitative picture this
suggests: the reasoning model is more "sticky" in its step-1 judgment
(right or wrong) across the three steps, while the non-reasoning model
revises its answer, in both directions, more often -- and does so more
by correcting an initial error than by recovering from a
self-introduced one.} We flag this as suggestive, not established: it
is one seed, one question pool, and one model pair, and -- critically
-- we cannot cleanly separate "reasoning models genuinely cascade
more" from "a forced-closure intervention applied to 64\% of this
model's generations degraded answer quality in a way that looks like
extra cascading." Both explanations are consistent with the data; this
experiment cannot distinguish them, and we do not claim it does.

This remains smaller in scope than this paper's two primary models
(300 vs.\ 2,500/1,200 pipelines) and uses a single reasoning-model
family. Cross-referenced against the two already-reported
non-reasoning models, cascade rate now spans four points on the
scale/reasoning axis: Qwen 2.5 3B (non-reasoning, $67.04\%$), Mistral 7B
(non-reasoning, $49.33\%$), Qwen 2.5 1.5B (non-reasoning, $67.33\%$,
this extension), and DeepSeek-R1-Distill-Qwen 1.5B (reasoning,
budget-forced, $81.00\%$, this extension) -- the closest same-scale
pair (the two 1.5B models) is the only comparison in this set that
isolates the reasoning variable, and even that comparison carries the
forced-closure caveat above
(\texttt{kaggle\_kernels/paper3-reasoning-model-v2/run\_reasoning\_model\_pipeline\_v2.py},
\texttt{results/reasoning\_model\_v2\_combined.json}; the superseded
first attempt remains at
\texttt{kaggle\_kernels/paper3-reasoning-model/run\_reasoning\_model\_pipeline.py},
\texttt{results/reasoning\_model\_pipeline\_results.json}, kept for the
methodological record).

## 4. Related Work and Scope Decisions

**On self-correction failure generally.** Two directly on-point prior
results, added in a final-audit pass, establish that LLM self-correction
without external feedback is broadly unreliable. This holds independent
of this paper's multi-step pipeline framing.

Huang et al. (2024) show that intrinsic self-correction frequently fails
to improve, and sometimes degrades, reasoning accuracy across several
benchmarks. Intrinsic self-correction means an LLM revises its own
answer using only its own judgment, with no external verifier or ground
truth. Our CASCADE/AMPLIFIED/LATENT taxonomy, and the 67.04%/49.33%
cascade-dominance finding, can be read as a more granular,
failure-mode-level elaboration of the same broad phenomenon.

Tyen et al. (2024) further localize why this happens: LLMs are poor at
finding their own reasoning errors, even when they can correct them once
the error location is given. This is consistent with this paper's own
LATENT mode -- correct at steps 1-2, wrong at step 3, invisible to any
monitor checking only intermediate outputs. It is also consistent with
the negative result of our early-detection probe (§3.2-3.4), which
likewise finds no strong signal that the pipeline "knows" a failure is
coming before it happens.

**On the Markov/absorbing-chain framing.** **Correction, added
post-review (second round): the change in our transition probabilities
is real, but smaller -- and differently caused -- than "moved
substantially" implied.** We now report the actual numbers, with the
underlying transition counts. These counts were added after third-round
review flagged the numbers as otherwise unverifiable from the supplied
artifacts.

| Regime | $n$ | $P(\text{wrong@2}\mid\text{wrong@1})$ | $P(\text{wrong@3}\mid\text{wrong@2})$ |
|---|---|---|---|
| N=600, TruthfulQA only, ROUGE-L (original; `fail-chain/MATHEMATICAL_FINDINGS.md`) | 600 | 0.9947 | 0.9654 |
| N=817, TruthfulQA only, corrected labels | 817 | 529/552 = 0.9583 | 496/568 = 0.8732 |
| N=2,500, TruthfulQA+NQ-open, corrected labels (this paper's headline pipeline) | 2,500 | 1,738/1,774 = 0.9797 | 1,704/1,786 = 0.9541 |

Our original small-scale measurement (N=600, Qwen, TruthfulQA only,
ROUGE-L labels) gave P(wrong@2|wrong@1)=0.9947 and
P(wrong@3|wrong@2)=0.9654. This paper's N=2,500 corrected-label,
TruthfulQA+NQ-open measurement gives P(wrong@2|wrong@1)=1,738/1,774=0.9797
and P(wrong@3|wrong@2)=1,704/1,786=0.9541. Each is about 1-1.5 points
lower -- not a qualitative change. Both regimes indicate a strongly
persistent ("nearly absorbing") wrong state.

Isolating the labeling-scheme effect specifically -- TruthfulQA-only,
N=817, corrected labels -- gives P(wrong@2|wrong@1)=529/552=0.9583 and
P(wrong@3|wrong@2)=496/568=0.8732. This is a larger, 3-9-point drop from
the original N=600 ROUGE-L numbers. This shows the labeling correction,
not the sample-size increase per se, is the dominant driver of whatever
instability exists. Adding NQ-open back in -- which §3.4 shows has a
higher cascade rate than TruthfulQA -- partially reverses this drop in
the full N=2,500 mixed-dataset figure.

Three different quantities changed simultaneously between our two
measurements: sample size, dataset composition, and labeling scheme. We
cannot attribute the residual movement to any one cause cleanly. We do
not consider a two-state transition probability, which moves by several
points under a labeling correction alone, a stable enough foundation for
derived quantities like recovery horizons or absorption times.

Independently, at least three 2026 papers formalize
self-correction/refinement dynamics as an absorbing Markov process, with
substantially more rigor than a hand-derived two-state model would
offer. arXiv 2604.22273, "Self-Correction as Feedback Control," derives a
two-state error-introduction/error-correction rate model with an
explicit stability threshold, tested on frontier models across
GSM8K/MATH/StrategyQA. arXiv 2604.24579, "Measuring the Unmeasurable,"
fits a full absorbing discrete-time Markov chain to agent traces, with
Kemeny-Snell fundamental-matrix treatment. arXiv 2512.02080 derives a
formal expected-iteration bound for LLM-verifier refinement loops. We
cite this literature as the correct home for this framing, rather than
compete with a less rigorous version of the same idea built on a single,
unstable transition-probability estimate.

**On "amplification."** We use AMPLIFIED strictly as the taxonomy mode
name -- correct at step 1, wrong at steps 2 and 3 -- throughout this
paper. We avoid using the same word for the step-to-step transition
probability, a distinct quantity. That overlap caused a naming collision
in earlier internal documentation of this project.

**On probing methodology and causal validity.** This paper's central
methodological move -- treating a linear probe's discriminability on
hidden states as evidence about what a model "knows" before it acts,
then separately testing whether that direction is causally load-bearing
-- sits inside an established caution in the interpretability
literature that we should credit explicitly rather than rediscover
implicitly. Belinkov (2022, "Probing Classifiers: Promises, Shortcomings,
and Advances") surveys exactly this gap: a probe's above-chance accuracy
establishes that information is linearly decodable, not that the model
uses it, and not that the represented quantity is what the researcher
assumes it is. Geiger et al. (2021, "Causal Abstractions of Neural
Networks") formalize the stronger standard this paper's §3.5 causal-
steering test is implicitly reaching for: a causal intervention on the
hypothesized variable should produce the predicted behavioral change
under a well-defined causal abstraction, not merely correlate with it.
Read against this standard, this paper's own trajectory -- a scalar-
feature probe (null), a full-vector probe (weak, difficulty-confounded
signal), and a causal-steering test (clean null) -- is a worked instance
of exactly the discipline both papers argue for, rather than an
exception to it.

## 5. Discussion and Limitations

- **Data and code availability.** All pipeline code, the labeling-bug
  fix and corrected recomputation scripts, causal-steering and
  context-ablation experiments, and every summary result JSON reported
  in this paper are publicly available at
  `https://github.com/Lakshmi-Chakradhar-Vijayarao/agentic-failure-taxonomy`.
  Raw per-item hidden-state caches and full generation logs (tens of
  GB per model) are not included; the released summary JSONs contain
  every statistic reported in this paper, and the pipeline code
  regenerates the raw caches from scratch given API/model access.
- **Scope note: no inference-economy claim is supported by this paper.**
  The only candidate early-exit signal -- step-1 correctness -- is an
  oracle requiring the gold answer. The geometric alternative is null
  (§3.2). We explicitly do not claim any deployable token-savings or
  compute-allocation mechanism. The paper's contribution is the
  taxonomy, the labeling-correction lesson, and the two negative
  results -- not an efficiency gain.
- **Full-vector probe (§3.2, added round-4, stress-tested round-5): a
  real but narrow and model-dependent finding, not a robust
  early-warning signal.** The PCA(100)+L2-logistic probe on raw
  step-1/step-2 hidden-state vectors is Mann-Whitney significant in the
  pooled population on both models. But the decisive
  difficulty-dissociation test -- restricting to pipelines already
  correct at step 1 -- shows this is mostly-to-entirely a
  step-1-difficulty readout. It collapses to chance on Mistral: AUROC
  0.489, $p=0.70$, $n=534$. It survives only weakly on Qwen: AUROC
  0.582, $p=0.012$, $n=726$. Cross-source transfer, TruthfulQA$\leftrightarrow$NQ-open,
  is weak-to-chance on both models: 0.469-0.545.

  Additional caveats apply. PCA(100) on $n\approx$1,200-2,500 with
  $d=$2048-4096 leaves real estimation variance. Only a single layer was
  tested -- L32 Qwen / L28 Mistral, an inherited convention, not tuned
  for this pipeline. The claim is scoped to that depth, not to
  "hidden-state geometry" broadly. **Closed, final-audit pass: a causal
  steering test now accompanies this passive probe, rather than leaving
  it untested.** Steering the residual stream during step-2 generation
  along this section's own cascade-predictive direction, versus a
  random-direction control, produces no significant shift in step-3
  correctness at either tested strength (McNemar $p=0.25$ at
  $\eta=15$; $p=0.625$ at $\eta=30$; $n=50$) -- a clean causal null
  (\S3.5). This strengthens rather than resolves the caveat above: the
  probe now has direct causal evidence against it carrying actionable
  signal, not merely an absence of causal evidence for it. **This
  causal test, at both $n=50$ and the $n=200$ scale-up, was run on Qwen
  only.** Mistral never appears in either causal-steering script or
  result file. Every other major result in this paper (the taxonomy,
  the difficulty-dissociation test, the context-ablation refutation,
  the layer sweep) is replicated on both models; the one genuine
  write-side causal test is not, inconsistent with this paper's
  otherwise cross-model standard, and we name that inconsistency here
  rather than let it pass unremarked.
- We led with the corrected numbers and disclosed the original ones,
  along with the two specific bugs responsible. We did not quietly
  revise without comment. The gap itself -- 97.28% to 67.04% cascade;
  0.76% to 25.28% correct -- is an instructive finding about
  surface-form QA metrics in multi-step pipeline evaluation. We report
  it as such.
- The original pipeline flagged samples near the ROUGE-L threshold as
  "boundary," excluded from LATENT/CORRECT statistical comparisons. Our
  corrected metric uses a different decision rule: exact-match-or-F1≥0.5
  for NQ-open; Jaccard threshold 0.12 for TruthfulQA, unchanged. The
  original boundary flags do not directly carry over. We have not
  re-derived an equivalent boundary-sensitivity analysis for the new
  metric. We do not believe this materially affects the headline
  taxonomy shift, which is large relative to plausible boundary noise.
  But a reviewer should treat the exact LATENT/AMPLIFIED/OSCILLATING
  counts as having unquantified sensitivity to the new metric's own
  threshold choice.
- LATENT (n=46 under corrected labels) is workable for descriptive
  reporting, but not for tight statistical claims. We do not report a
  per-mode confidence interval narrow enough to support strong claims
  about this specific mode.
- Two models are tested: Qwen 2.5 3B and Mistral 7B Instruct. Both are
  instruction-tuned decoder-only transformers, of broadly similar scale
  (3B-7.6B). The taxonomy, null detection result, and context-ablation
  refutation all replicate directionally. The ablation specifically
  replicates closely in magnitude (-4.4pp vs. -4.2pp). But two same-era,
  similarly-sized instruction-tuned models is not evidence the findings
  hold for very different scales, for base (non-instruction-tuned)
  models, or for architectures meaningfully distinct from the standard
  decoder-only transformer. The hidden-state layer used for both --
  L32/36 for Qwen, L28/32 for Mistral, ~87.5-89% depth -- is an
  inherited convention from a companion project. **Closed, elite-review
  follow-up: we validated it with a layer sweep, reconstructing the
  decisive test's own concat($h_{s1},h_{s2}$) probe at two additional
  depths per model from the already-generated step-1/step-2 text**
  (teacher-forced through the model in one forward pass -- fail-chain's
  own generation is greedy, so this reproduces the same hidden states
  the original generation-time hook captured, no new sampling needed;
  `kaggle_kernels/paper3-layer-sweep/run_layer_sweep.py`,
  `results/layer_sweep_results.json`). **The inherited depth is not the
  strongest available for Qwen.** L16/L24/L32: AUROC $0.546$, $0.621$,
  $0.588$ -- L24 (67\% depth) beats the inherited L32 convention by
  $+0.033$. For Mistral, all three depths tested (L14/L21/L28: $0.557$,
  $0.530$, $0.507$) stay weak, consistent with the paper's existing
  "substantially a difficulty readout" reading regardless of which depth
  is chosen -- the collapse-to-chance finding is not an artifact of the
  specific inherited depth. The Qwen result means the paper's own
  $0.582$ headline number understates, not overstates, the strongest
  layer-wise signal available in this representation; we do not revise
  the headline number itself (chosen before this check, and consistent
  with the companion-project convention this pipeline inherited), but
  flag that a from-scratch layer search, not run here, could report a
  somewhat stronger residual signal on Qwen specifically. **Both models
  share a single prompt template throughout this entire paper, tested
  at no alternate wording -- an untested fragility axis at least as
  large a threat to the taxonomy's external validity as the
  cross-source transfer check we do run (\S3.2), since the entire
  cascade-dominance finding is a behavioral pattern this specific
  pipeline design induces.** **Closed, final-audit pass: a third and
  fourth model
  are tested in a secondary, smaller-scope extension (\S3.6)** --
  DeepSeek-R1-Distill-Qwen-1.5B (a reasoning model) against
  Qwen2.5-1.5B-Instruct (the same-scale, non-reasoning base it was
  distilled from). This extension carries its own disclosed limits, not
  absorbed into the main taxonomy's scope above: budget-forcing required
  forced closure on 576/900 (64.0%) of the reasoning model's step-level
  generations, so we cannot cleanly separate "reasoning models genuinely
  cascade more" from "forced closure applied to 64% of this model's
  generations degrades answer quality in a way that looks like extra
  cascading" -- both explanations are consistent with the data. **This
  is a data-collection gap, not an unresolvable analytic question:**
  the generation script logs only a single aggregate forced-closure
  counter, not a per-item, per-step flag, so the direct test (does
  forced closure correlate with CASCADE at the item level, within the
  existing $n=300$ run) cannot be computed from the artifacts already
  saved. Re-running with per-item logging added would resolve this
  without any new experimental design. It is one seed, one question
  pool, and one model pair, not a systematic reasoning-vs-non-reasoning
  sweep.
- The corrected metric is still surface-form: normalized exact-match/F1,
  multi-reference Jaccard. It is not semantic judgment by an
  independent model. We do not claim the corrected numbers are a
  ceiling on further label-quality improvement -- only a substantial,
  disclosed improvement over the original.

  **Closed, post-review (round 6): we ran the LLM-judge pass.** An
  independent judge model -- Qwen2.5-3B-Instruct, chat-templated,
  greedy-decoded to a single CORRECT/HALLUCINATED verdict -- scored a
  stratified sample from each model's relabeled data. This sample
  matched the relabel_label_s3 distribution: 150/2,500 Qwen pipelines
  (44 correct / 106 hallucinated); 100/1,200 Mistral pipelines (47
  correct / 53 hallucinated). The judge saw the question, gold reference
  answer, and step-3 generation, shown verbatim.

  **Agreement with the surface-form label is moderate on Qwen and weak
  on Mistral -- not strong on either.** Qwen: 77.3% raw agreement,
  Cohen's $\kappa = 0.467$ (moderate, Landis-Koch). The confusion matrix
  (rows=surface, cols=judge) is $\begin{bmatrix}87 & 19\\15 &
  29\end{bmatrix}$. Of the 44 surface-labeled-correct pipelines, the
  judge called 15 (34.1%) hallucinated. Of the 106
  surface-labeled-hallucinated pipelines, the judge called 19 (17.9%)
  correct.

  Mistral: 57.0% raw agreement, $\kappa = 0.123$ (slight). The confusion
  matrix is $\begin{bmatrix}38 & 15\\28 & 19\end{bmatrix}$. Of the 47
  surface-labeled-correct pipelines, the judge called 28 (59.6%)
  hallucinated. Of the 53 surface-labeled-hallucinated pipelines, the
  judge called 15 (28.3%) correct. No verdict was ambiguous ($n=0$ in
  either model).

  **This closes the "surface-form vs. semantic judgment" limitation, but
  does not resolve it favorably.** The corrected metric this paper
  relies on throughout tracks an independent semantic judge only
  loosely. On Mistral, the two disagree on well over half of the
  pipelines the surface-form metric calls correct. We do not know which
  of the two -- normalized string-overlap or a 3B instruction-tuned
  judge -- is closer to a human-adjudicated ground truth for either
  model. Both are heuristics. This result is evidence the paper's
  taxonomy counts carry real, non-trivial label-quality uncertainty,
  beyond what the substring-match audit below quantifies. It is not
  evidence that either heuristic is simply wrong. A held-out
  human-adjudication pass, not run here, would be needed to arbitrate
  between them.

  **Final-audit caveat 1, resolved: the Qwen judge was not independent of
  the generator. Re-running with a genuinely independent judge changes
  the number substantially, in a direction that complicates rather than
  simplifies the story.** `code/01_scaled_pipeline_generation.py` and
  `code/08_llm_judge_validation.py` both use Qwen2.5-3B-Instruct. The
  Qwen arm's judge was the same checkpoint that produced the pipeline's
  own outputs -- a self-preference confound documented in the
  LLM-as-judge literature.

  We re-ran the identical stratified sample (150 Qwen pipelines,
  seed=42) with a genuinely independent judge, sharing no model family
  with Qwen: Mistral-7B-Instruct-v0.2-AWQ. This is the same checkpoint
  already validated in this project's companion leakage-audit paper
  (`kaggle_kernels/paper3-llm-judge-independent/run_llm_judge_independent.py`,
  `results/llm_judge_independent_qwen.json`).

  **Result: agreement drops from 77.3% (self-judged) to 51.4%
  (independent). $\kappa$ drops from $0.467$ (moderate) to $0.179$
  (slight)** ($n=146$ scored, 4 ambiguous verdicts excluded). The
  confusion matrix (rows=surface, cols=judge) is
  $\begin{bmatrix}35 & 67\\4 & 40\end{bmatrix}$. This disagreement is
  asymmetric. Notably, it runs in the opposite direction a simple
  self-preference story would predict. Of the 102
  surface-labeled-hallucinated pipelines, the independent judge called
  67 (65.7%) correct. The independent judge is far more lenient overall
  -- 26.7% judge-hallucination rate -- than either the surface metric
  (69.9% surface-hallucination rate) or Qwen's self-judge (68.0%
  judge-hallucination rate) were.

  We flag two competing explanations we cannot distinguish between with
  this data. (a) Qwen's self-judge was, if anything, unusually harsh
  rather than lenient toward its own outputs. This is consistent with --
  but not proof of -- some self-critical bias rather than
  self-preference. Under this explanation, the independent Mistral judge
  reveals substantially more of Qwen's surface-labeled failures are
  semantically correct than either prior measurement suggested. (b)
  Alternatively, Mistral-7B is simply a more lenient judge in general,
  independent of any self/other relationship. Under this explanation,
  the result says more about judge-model calibration differences than
  about Qwen specifically.

  **Either way, this final-audit rerun does not rescue the original
  self-judged $\kappa=0.467$ as a trustworthy number.** If anything, it
  now looks like the more favorable of three increasingly-disagreeing
  measurements -- surface-form, self-judged, independently-judged -- not
  a settled validation of the corrected taxonomy's step-3 labels. We
  consider all three Qwen measurements now established. The extent of
  three-way disagreement itself, rather than any single one of them, is
  the honest final word on Qwen's step-3 label reliability.

  **Closed, final-audit pass: we ran a held-out human-adjudication pass
  on a sample of the independent judge's disagreements with the surface
  label, rather than leaving the three-way disagreement uninterpreted.**
  We manually read the question, reference answer, and full step-3
  generation for two samples: all 4 cases where the surface label said
  correct but the independent (Mistral) judge said hallucinated, and a
  random 15-of-67 sample (seed 11) of the reverse -- surface said
  hallucinated, independent judge said correct, the disagreement
  direction that drives the independent judge's much higher leniency.
  **The two disagreement directions resolve very differently, and
  neither the surface metric nor the independent judge comes out as the
  more reliable arbiter overall.** On the small
  surface-correct/judge-hallucinated bucket, the independent judge is
  right in all 4 cases we checked: each surface "correct" call turned
  out to rest on incidental keyword overlap (e.g.\ a generation about
  "the second Prime Minister of independent India" that named the wrong
  person while the correct name happened to also appear, in a different
  role, elsewhere in the same passage) rather than an actually correct
  answer. On the larger surface-hallucinated/judge-correct bucket, our
  reading is close to a coin flip: 6 of 15 confirmed cases where the
  independent judge's "correct" call was right and the surface metric's
  Jaccard/F1 threshold simply missed a genuine paraphrase (Wilson 95\%
  CI [19.8\%, 64.3\%]), but 7 of 15 where the independent judge was
  \emph{itself} wrong -- most often on TruthfulQA's "decline to
  answer" trap questions, where the reference correct answer is a
  non-committal "I have no comment" and the generation instead
  fabricates a specific, confident-sounding claim that the independent
  judge nonetheless scored CORRECT (CI [24.8\%, 69.9\%]; 2 further cases
  were borderline and not counted either way). **This means the
  independent judge's much higher leniency (26.7\% judge-hallucination
  rate vs.\ 68.0-69.9\% for the surface metric and Qwen's self-judge)
  is itself partly a measurement artifact of the judge over-crediting
  fluent, confident fabrications as correct** -- not simply evidence
  that the surface metric and Qwen's self-judge were both too harsh. We
  read this as strengthening, not resolving, the paper's central
  labeling-uncertainty caveat: a single small-sample, single-rater
  adjudication pass (this one) cannot itself serve as ground truth
  either, and a systematic, multi-rater human-adjudication study remains
  the only way to fully arbitrate between all three measurements.

  **Closed, final-audit pass: steps 1 and 2 are now judge-validated too,
  not left on surface-form labels alone.** Identical judge, prompt, and
  stratified-sampling procedure as the step-3 self-judge pass above
  (Qwen2.5-3B-Instruct, same stratification seed, $n=150$ Qwen /
  $n=100$ Mistral per step) -- this is the self-judged variant for both
  new steps, carrying the same self-preference caveat already disclosed
  for Qwen's step-3 self-judge result above (not the independent-Mistral-
  judge variant, which was only run for step-3). Qwen: step-1 agreement
  76.0\%, $\kappa=0.443$ (moderate); step-2 agreement 68.7\%,
  $\kappa=0.323$ (fair) -- both a shade below step-3's self-judged
  $\kappa=0.467$, consistent with, not worse than, this section's
  prediction that the rarer taxonomy modes hinging on these labels carry
  comparable label uncertainty. Mistral: step-1 agreement 68.0\%,
  $\kappa=0.351$ (fair); step-2 agreement 67.0\%, $\kappa=0.305$ (fair)
  -- both notably \emph{higher} than Mistral's own step-3 self-judged
  $\kappa=0.123$, the opposite of what we predicted. We do not have a
  confirmed explanation for why Mistral's step-3 labels specifically
  disagree with the judge more than its step-1/step-2 labels do, and
  flag it as an open, somewhat counterintuitive observation rather than
  reaching for one. Either way, no step at either model reaches an
  agreement level that would let the taxonomy's 3-bit pattern be read as
  resting on settled ground truth -- LATENT ($n=46$) and OSCILLATING
  ($n=12$; see §3.1) remain small-$n$, descriptive findings, now with a
  measured (not merely presumed) comparable label-uncertainty budget
  behind them.

  **Substring-match audit, completed rather than left as a caveat.** The
  NQ-open exact-match rule scores a generation as correct if it contains
  a normalized alias as a substring. This risks false positives when a
  generation mentions an entity without that entity being the actual
  answer.

  We identified all step-3 NQ-open pipelines labeled correct via
  substring match only, with no exact full-string match -- a pool of
  457/1,683 (27.2%). We manually audited a random sample of 30 from this
  pool, reading the full generation against the question. The question
  pool was shuffled with seed 42 during original generation; the 30-item
  audit sample was drawn with a separate seed, 7.

  **2 of 30 (6.7%, Wilson 95% CI [1.8%, 21.3%]) were confirmed false
  positives.** One generation misidentified the wrong Game of Thrones
  character entirely -- calling The Hound Oberyn Martell, with the wrong
  actor -- but was marked correct because it mentioned "Westeros," a
  setting name common to nearly any answer about the show. Another named
  the wrong winner of a boxing match, reversing De La Hoya and Hopkins,
  but matched because the opponent's name appeared in the text
  regardless of who the generation said won.

  Two further cases were borderline: a self-contradictory generation
  that denied a fact it then correctly stated; a multi-fact answer where
  the matched alias corresponded to a different sub-question than the
  one asked. We did not count these as confirmed errors. Including them
  (4/30 = 13.3%) would raise the estimate to $\approx$2.4% of the
  dataset. We report the more conservative 2-of-30 count as the
  confirmed figure, and flag the borderline-inclusive figure as an upper
  sensitivity check.

  **Propagating the interval, not just the point estimate.** The
  confirmed 6.7% rate (Wilson CI [1.8%, 21.3%]) applies to the
  457-pipeline substring-only-correct pool. This implies an estimated
  **$\approx$1.2% of the full $N=2{,}500$ Qwen dataset -- about 30
  pipelines**. The 95% CI on that rate propagates to roughly **0.3%-3.9%
  of the dataset -- about 8 to 100 pipelines**.

  Even at the pessimistic, upper-CI, borderline-inclusive end of this
  range, the affected count is an order of magnitude below the headline
  taxonomy shift this section reports: CORRECT alone moved by 612
  pipelines, 0.76%$\to$25.28%. The bias direction is to slightly
  overstate CORRECT and understate CASCADE. It is bounded well below the
  size of the finding it might otherwise be mistaken for undermining.

  **Extended, final-audit pass: the same audit now also covers Mistral's
  NQ-open arm**, identical procedure and seeds
  (`code/12_substring_match_audit_mistral.py`). The substring-only-correct
  pool is proportionally larger on Mistral (165/383, 43.1\%, vs.\ Qwen's
  27.2\%) -- consistent with Mistral's lower overall correctness rate
  making substring-only matches a larger share of its (smaller) correct
  pool. Manually auditing the same 30-item sample size (seed 7): **1 of
  30 (3.3\%, Wilson 95\% CI [0.6\%, 16.7\%]) is a confirmed false
  positive.** The model claimed "Daimyo" was the highest rank in the
  Japanese warrior class, explicitly describing "Samurai" as mere
  retainers beneath it -- directly contradicting the reference answer
  ("Samurai") that the substring match credited it with. Three further
  cases are borderline (hedged or partially-relevant answers that
  happen to contain the reference string, e.g. a generation that first
  says "it is difficult to provide an exact number" before stating one
  matching the reference) -- not counted as confirmed errors, following
  the same conservative convention as the Qwen audit. Including them
  (4/30 = 13.3\%, CI [5.3\%, 29.7\%]) gives the upper sensitivity
  estimate. Propagating the confirmed rate to Mistral's full $N=1{,}200$
  dataset implies **$\approx$0.5\% affected ($\approx$6 pipelines)**,
  CI-propagated range $\approx$0.1\%-2.3\%; the borderline-inclusive
  estimate is $\approx$1.8\% ($\approx$22 pipelines), CI-propagated range
  $\approx$0.7\%-4.1\%. Both are, as on Qwen, an order of magnitude below
  the headline taxonomy shifts this section reports, and both are
  slightly higher than Qwen's corresponding 6.7\%/1.2\% figures --
  directionally consistent with Mistral's larger raw substring-only pool,
  not evidence of a qualitatively different failure mode.

  TruthfulQA's Jaccard-based labeling is audited too, on both models
  (`code/13_jaccard_audit_truthfulqa_qwen.py`,
  `code/20_jaccard_audit_truthfulqa_mistral.py`). TruthfulQA's Jaccard
  scheme has no clean analogue to NQ-open's "substring but not exact
  match" bucket: 275/280 (98.2\%) of step-3-correct Qwen/TruthfulQA
  pipelines already sit at Jaccard $\leq 0.35$, since full-sentence
  answers rarely reach high word overlap even when genuinely correct.
  We instead define the risk bucket as score $\leq 0.25$ (Qwen:
  244/280, 87.1\% of the correct pool; Mistral: 288/392, 73.5\% --
  both models show the large majority of TruthfulQA "correct" calls
  resting on thin lexical margins) and manually audit a 30-item sample
  (seed 7) on each model.

  This audit is harder to adjudicate than NQ-open's, and the result is
  correspondingly worse on both models. On Qwen, 6 of 30 (20.0\%,
  Wilson 95\% CI [9.5\%, 37.3\%]) are confirmed false positives,
  roughly 3$\times$ NQ-open's 6.7\%. TruthfulQA's questions are
  deliberately constructed imitative-falsehood traps, and several
  confirmed errors are exactly that pattern: a generation claimed King
  Henry VIII holds a book in his famous portrait (reference: a glove),
  scoring "correct" because restating the question itself supplies most
  of the word overlap; a generation asked where the sun never shines
  named the South Pole during polar winter, missing that even the poles
  receive roughly six months of daylight (the reference: no such place
  exists); a generation reversed US-vs-Nordic paid-vacation policy; a
  generation fabricated a French etymology for "butterfly" unrelated to
  its actual Old English root. A further 8 cases are borderline
  (hedged, self-contradictory, or partially-relevant generations, not
  counted as confirmed errors); including them (14/30 = 46.7\%, CI
  [30.2\%, 63.9\%]) gives the upper sensitivity estimate. Propagating
  the confirmed rate to the full weak-evidence pool implies
  $\approx$2.0\% of the full $N=2{,}500$ Qwen dataset ($\approx$49
  pipelines) affected, CI-propagated range $\approx$0.9\%-3.6\%; the
  borderline-inclusive estimate is $\approx$4.6\% ($\approx$114
  pipelines), CI-propagated range $\approx$3.0\%-6.2\%.

  On Mistral, 5 of 30 (16.7\%, Wilson 95\% CI [7.3\%, 33.6\%]) are
  confirmed false positives -- in the same register as Qwen's,
  including a fabricated country list for "most guns after the US"
  (reference: India, China, Pakistan, Russia; the generation instead
  named Yemen, Serbia, Switzerland, conflating per-capita with total
  counts) and the model asserting the Amy Cuddy power-poses effect as
  an established finding (reference: the effect does not replicate). A
  further 3 are borderline; including them (8/30 = 26.7\%, CI [14.2\%,
  44.4\%]) gives the upper estimate. Propagating the confirmed rate to
  Mistral's full $N=1{,}200$ implies $\approx$4.0\% affected
  ($\approx$48 pipelines), CI-propagated range $\approx$1.8\%-8.1\%.

  Unlike the NQ-open audit, neither model's TruthfulQA figure is an
  order of magnitude below the headline taxonomy shift, and both
  materially sharpen rather than dismiss this paper's own repeated
  caveat that TruthfulQA percentages throughout, including in the
  abstract, rest on a weaker labeling foundation than the NQ-open
  numbers do: the false-positive rate is substantially higher than
  NQ-open's on both models tested (Mistral: 16.7\% vs.\ its own NQ-open
  3.3\%; Qwen: 20.0\% vs.\ 6.7\%), not an artifact specific to one
  model's response style. We flag one further limitation on this audit
  itself: TruthfulQA's trap-question design makes correctness harder to
  adjudicate by eye than NQ-open's entity-substring questions, so this
  classification required more judgment calls than the NQ-open audit's
  -- a human-adjudication pass by a second, independent reader, not run
  here, would be the natural check on our own manual classifications
  specifically.
- The strong "output-level baseline" reported in §3.2 and §3.4 (AUROC
  0.92 Qwen, 0.91 Mistral) is an oracle. It is computed from the
  gold-label-derived step-1-correctness field. It is not available
  without already knowing the correct answer. It is also substantially a
  restatement of the taxonomy's own CASCADE/CORRECT-diagonal
  concentration: 92.3% of Qwen pipelines have matching step-1/step-3
  correctness. It is not evidence of a nontrivial predictive
  relationship. No deployable early-warning signal, geometric or
  otherwise, was identified in this paper.
- The Qwen-vs-Mistral cascade-rate gap is partly, but not primarily,
  attributable to the two models' test sets having different
  TruthfulQA/NQ-open mixes (§3.4). A double-digit-point gap survives
  composition-matching in both reweighting directions. The raw,
  unweighted 67.0%-vs-49.3% comparison should not be read as a clean
  capability measurement on its own.
- The Mistral replication of the context-ablation refutation is
  directionally and quantitatively close to Qwen's (-4.2pp vs. -4.4pp).
  It is not itself statistically significant at n=120 per arm: Fisher
  exact p=0.171, vs. Qwen's p=0.046 at n=250 per arm. We describe this
  as a directional, not a doubly-significant, cross-model replication.
- **This paper's decisive difficulty-dissociation test (§3.2) is one
  instance of a pattern that recurs across three companion studies from
  this project.** An FFN-mechanism paper finds a component comparison
  reverses under the correct prompt template. A geometric-certificate
  paper finds a near-saturated benchmark is substantially explained by
  answer length, rather than truthfulness. A leakage-taxonomy audit
  needed four rounds of correction to separate a real effect from a
  training-budget and random-seed confound. In each case, including this
  one, a signal that looked real under a simpler test either shrank or
  disappeared -- once the specific, nameable confound was actually
  tested, rather than merely flagged as a caveat.

## 6. Conclusion

Cascade failure in multi-step self-refinement is real. It remains the
dominant outcome on both models tested, even after correcting a
substantial labeling artifact. It sits at roughly two-thirds of
pipelines on Qwen and half on Mistral -- subject to an audited
$\approx$1.2% residual NQ-open substring-match mislabeling risk on Qwen
and $\approx$0.5% on Mistral (both audited, final-audit pass), which
slightly overstates CORRECT and understates CASCADE, and a substantially
larger, separately-audited TruthfulQA Jaccard-labeling risk on both
models ($\approx$2.0\% Qwen, $\approx$4.0\% Mistral) (§5). This is
not the near-total collapse an inappropriate metric first suggested.
Most, though not all, of the gap between the two models' cascade rates
survives controlling for their test sets' different TruthfulQA/NQ-open
composition. This supports, in a narrower form than our first draft
claimed, a partly capability-dependent story.

The paper's most important negative result is this: a proposed
geometric early-warning signal, measured via scalar summary statistics,
does not survive proper labeling on either model. It was very likely
detecting a degenerate label distribution, not genuine internal signal.

Closing a gap flagged in round-4 review, we then ran a full-vector probe
on the same cached hidden states. This scalar null does not extend
cleanly to the full representation. PCA(100)+L2-logistic on the raw
vectors recovers a Mann-Whitney-significant signal on both models --
corrected, in round-5 review, from an invalid fold-SD heuristic. But the
decisive test this motivated shows otherwise: does the signal survive
when restricted to pipelines already correct at step 1, where no step-1
difficulty remains to explain it? The signal is mostly-to-entirely a
difficulty readout. It collapses to chance on Mistral. It survives only
weakly, barely significantly, on Qwen. Cross-source transfer is poor on
both.

We do **not** round any of this up to a deployable signal. Even its best
case, Qwen at $p=0.012$, is far below the oracle and not a basis for an
early-exit mechanism. We do **not** conclude that a simple output-level
heuristic replaces either result. The step-1-correctness "baseline,"
which does predict step-3 outcome well, is an oracle requiring the gold
answer, not a deployable signal. It is substantially a restatement of
the taxonomy's own diagonal concentration, rather than a nontrivial
predictive relationship.

The honest summary is therefore three-layered -- not a clean null and
not a clean positive. **The scalar geometric features we first tested
carry no early-warning value. The fuller hidden-state representation
carries a real pooled-population signal. The decisive
difficulty-dissociation test shows that signal is substantially
explained by step-1 difficulty, model-dependently so, with only a weak,
poorly-transferring residual on Qwen and none on Mistral.**

The context-ablation refutation of the contamination hypothesis is
close in magnitude across two architecturally distinct models: -4.4pp
Qwen, -4.2pp Mistral. It is itself significant on Qwen (Fisher exact
p=0.046) but not on Mistral at this sample size (p=0.171). This is a
directionally consistent, quantitatively close, but not
doubly-significant replication. We no longer call it the paper's single
most robust finding without that qualification.

We present four contributions as the paper's honest, defensible record:
the corrected, cross-model taxonomy; the replicated scoped result on
early-warning detection -- scalar null, a pooled-population full-vector
signal that a decisive difficulty-control test shows is
mostly-to-entirely difficulty itself, oracle baseline's limits made
explicit; the dataset-composition-checked cross-model cascade
comparison; and the directionally-replicated context-ablation
refutation.

Beyond these four specific findings, we want to name the difficulty-
dissociation design itself as a reusable pattern independent of what it
found here: before trusting any claimed hidden-state "early-warning"
or "risk" signal for a multi-step process, restrict to the subpopulation
where the earliest available oracle outcome is already fixed, and check
whether the signal survives. A signal that disappears under this
restriction was reading the oracle variable, not something new. This
pattern generalizes to any pipeline with a step-wise correctness oracle,
not just self-refinement.

## References

Added, final-audit pass: this paper previously named several works
inline (e.g. "Kumar et al.", three bare arXiv IDs) without a References
section -- not submittable in that state. Full citations below.

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., &
Zhou, D. (2024). Large Language Models Cannot Self-Correct Reasoning
Yet. *ICLR 2024*. arXiv 2310.01798.

Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J. D., Singh, A.,
Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., Zhang, L. M., McKinney,
K., Shrivastava, D., Paduraru, C., Tucker, G., Precup, D., Behbahani, F.,
& Faust, A. (2024). Training Language Models to Self-Correct via
Reinforcement Learning. *ICLR 2025*. arXiv 2409.12917.

Liu, A., & Meng, J. (2026). Self-Correction as Feedback Control: Error
Dynamics, Stability Thresholds, and Prompt Interventions in LLMs.
arXiv 2604.22273.

Tran-Truong, P. T., & Le, X.-B. (2026). Measuring the Unmeasurable:
Markov Chain Reliability for LLM Agents. arXiv 2604.24579.

Dantas, P., Cordeiro, L., Sun, Y., & Junior, W. (2025). The 4/$\delta$
Bound: Designing Predictable LLM-Verifier Systems for Formal Method
Guarantee. arXiv 2512.02080.

Tyen, G., Mansoor, H., Carbune, V., Chen, P., & Mak, T. (2024). LLMs
Cannot Find Reasoning Errors, But Can Correct Them Given the Error
Location. *Findings of the Association for Computational Linguistics:
ACL 2024*. arXiv 2311.08516.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi,
H., Zettlemoyer, L., Liang, P., Candès, E., & Hashimoto, T. (2025). s1:
Simple Test-Time Scaling. arXiv 2501.19393.
