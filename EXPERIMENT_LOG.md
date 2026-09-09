# HiF4 online experiment log

This file is the authoritative record for experiments on the current contest
dataset. Historical scores in the organizer repository came from a different
dataset and must not be used to predict the current leaderboard.

## Current online results

| Revision | Server score | Server time | Decision | Main change |
| --- | ---: | ---: | --- | --- |
| `d75e03a` | 15300 | 261 s | superseded | Removed the expensive full Linear Hessian path |
| `5b922c8` | 15600 | 248 s | superseded | Attention alpha change without extra runtime |
| `a649209` | 16000 | 243 s | superseded | Rank-8 low-rank Linear Hessian |
| `237b142` | 0 | >300 s | rejected | Dynamic Q+K Attention Hessian; timed out |
| `b7248c6` | 16000 | not recorded | rejected | Timeout-safe adaptive Q/K and guarded V path |
| `def4524` | 16775 | 242 s | previous baseline | Rank-32 Linear plus pre-factored K-only Hessian |
| `097c2e5` | 16775 | 244 s | rejected | Rank-40/two sweeps plus position-aware output Hessian |
| `e0a19b0` | 16700 | 250 s | rejected | Cross-validated K and per-KV-head alpha selection |
| `3c40705` | 16991 | 250 s | previous baseline | Guarded K centering plus fixed-scale K mantissa refinement |
| `988385e` | 17440 | 252 s | previous baseline | V output-error coupling across 16-token groups |
| `be6ffae` | 17508 | 254 s | previous baseline | Linear mantissa refinement with covariance reliability |
| `fe4b879` | 17675 | not reported | previous baseline | Use full calibrated V token coupling |
| `04a23c1` | not measured | not measured | pending parent | Nested 8/16-token V coupling plus bit-identical speedups |
| `a2b0de7` | not measured | not measured | pending parent | Test-time softmax-invariant K translation selection |
| `92ed4fe` | not measured | not measured | pending parent | Add nested 4-token V error coupling |
| `768a670` | 18178 | 233 s | previous baseline | Speedups, nested V and dynamic K translation selection |
| `9fa93f2` | not measured | not measured | submitted code parent | Fixed-scale K refinement in softmax quotient space |
| `8d88fbc` | 18182 | 230 s | previous baseline | Append fixed-scale K quotient-space refinement |
| `0ff81b5` | not measured | not measured | submitted candidate | Asymmetric Dynamic Activation reconstruction against quantized Weight |
| `06250ad` | not measured | not measured | submitted candidate | Token-level guard for asymmetric Dynamic Activation refinement |
| `3e93bd2` | 18417 | 232 s | previous baseline | Record token-guarded asymmetric Activation candidate |
| `c11d987` | **18471** | **236 s** | **online baseline** | Calibration guards plus analytic V-prefix correction |

The user reported `768a670` at 18178 points in 233 seconds.  Relative to the
previous online baseline `fe4b879`, the bundled branch gains 503 points.  Its
233-second runtime also leaves 67 seconds below the hard timeout and is 21
seconds faster than `be6ffae`, the nearest parent with a reported time.  The
bundle contains bit-identical kernel speedups, nested V coupling, dynamic K
translation selection and the four-token V term, so their individual server
contributions cannot be separated.

The user then reported `8d88fbc` at 18182 points in 230 seconds.  It differs
numerically only by the appended quotient-space K mantissa pass, so that pass
contributes just 4 server points; the 3-second reduction is runtime noise in
the favorable direction.  The mechanism remains in the best revision, but
further K Hessian/mantissa expansion is stopped because the much larger local
proxy gain did not transfer.  The remaining gap to 20000 is 1818 points, with
70 seconds of measured timeout headroom.

Revision `0ff81b5` leaves Weight and all Attention parameters bitwise identical
to `8d88fbc`.  It stores block-local `What^T What` and `What^T W` statistics in
the Linear activation state, then performs one fixed-scale legal-mantissa pass
on each dynamic Activation.  A conservative strength of `0.125` was selected
on generic distribution families, and each changed block must improve the
asymmetric objective while increasing plain reconstruction MSE by at most 1%.
This directly models the fact that the opposite Weight operand has already
been quantized; it does not compute a calibration `A @ W` target.

- Four independent nine-family robust seeds improve in aggregate by
  `+0.0152` to `+0.0188`; 35 of 36 individual distribution cases improve and
  the sole regression is `0.00027`.
- The non-Qwen public Linear relative score changes `0.77598 -> 0.78208`.
- All three captured Qwen Linear layers improve, with their mean changing
  `0.44797 -> 0.45382`; Qwen is confirmation-only.
- Final same-process public timing changes `26.01 -> 26.60 s`.  The Weight
  output is bitwise unchanged, all five Q/K/V outputs are bitwise unchanged,
  and the official format check passes `22/22`.

The candidate is intentionally isolated for an online measurement.  Its local
gain is much broader than the saturated K quotient refinement, while its
measured cost is small relative to the 70-second server headroom.  It is not
assumed to close the full 1818-point gap by itself.

Revision `06250ad` changes the acceptance unit of the asymmetric Activation
pass from each independent 64-channel block to the complete token.  The legal
mantissa coordinate search is still block-local and still runs exactly once,
but a token is accepted only when its aggregate asymmetric objective improves.
This admits compensating changes that the old 1% per-block plain-MSE cap
discarded.  A 15% token-level plain-MSE cap remains as a distribution guard;
the asymmetric coupling is `7/32`, selected on generic distributions rather
than the public or captured-model tensors.

- Across eight independent nine-family synthetic seeds, every aggregate
  improves.  The prototype mean rises from `0.061--0.085` to `0.233--0.252`,
  a per-seed gain of approximately `+0.158` to `+0.174`; all tested families
  improve, including heavy-tail, sparse-outlier and low-rank cases.
- Non-Qwen public Linear changes `0.78208 -> 0.82433`.
- Captured Qwen Linear mean changes `0.45383 -> 0.46047`; all three layers
  improve, and Qwen remains confirmation-only.
- Same-process public timing is `25.45 -> 25.55 s`.  No extra coordinate
  sweep, scale candidate, factorization or matrix product was added.
- Weight is bitwise unchanged and every Attention Q/K/V output is bitwise
  unchanged relative to `8d88fbc`; the official output-format check remains
  `22/22`.

This is a substantially stronger, effectively runtime-neutral successor to
`0ff81b5`.  The analytic V prefix candidate remains separate so the next
server result can be attributed to the Linear acceptance change.

The user reported `3e93bd2` at 18417 points in 232 seconds.  Relative to
`8d88fbc`, the isolated token-guarded asymmetric Activation path gains 235
server points for 2 seconds.  This promotes it to the online baseline, leaving
1583 points to 20000 and 68 seconds below the hard timeout.  The gain confirms
that quantized-opponent reconstruction transfers, but its much larger local
Linear improvement maps to modest server weight; further Linear coordinate
sweeps or covariance sketches are therefore stopped.  The next candidate uses
the remaining budget on V/Attention and remains isolated from Linear changes.

The user then reported `c11d987` at 18471 points in 236 seconds. Its random
trend estimate had been about 20000--20800 depending on the fitted snapshot,
so the actual gain of only 54 points establishes that the synthetic affine
score is no longer a usable point predictor. It remains useful only for
rejecting broad regressions. All post-`c11d987` work must preserve separately
submittable online probes; bundling unrelated local gains loses the only signal
available from the hidden evaluator.

### Post-c11 black-box probe family

Four changes were first built as independent probes from the common
`c11d987` parent:

1. raise only the short-sequence analytic V-prefix cap from `0.025` to `0.2`;
2. change only the outer V exchange group from 16 to 32 tokens;
3. add only the calibration-selected Q/K family (rich, direct,
   Hadamard-only, Smooth-only, and Smooth+Hadamard);
4. add only activation-weighted direct Linear fallback quantization;
5. combine accepted effects only after their main effects are known.

For a probe score `S_i` and baseline `B=18471`, `S_i-B` is the corresponding
online main effect. The interaction of the full candidate is its gain minus
the sum of those four effects. This cannot reconstruct the hidden tensors, but
it identifies the score-weighted sensitivity to short-sequence V, wider token
coupling, alternate Q/K transforms, and direct Linear cases. It is a much more
sample-efficient form of online adaptation than tuning another synthetic
score mapping.

Audit then quarantined probes 2 and 4 from the main candidate. The 32-token V
prototype applied a coefficient calibrated on 16-token groups and halved the
update density. The direct Linear prototype replaced the algorithm after the
existing selector had evaluated a different direct path, mishandled arbitrary
prefix dimensions, and added about 11 seconds on an archived `8192x2048`
Weight when activated. They remain useful online diagnostics but are not safe
defaults.

The corrected main worktree therefore contains only the stronger decayed
V-prefix cap and Q/K multibranch selection. It reuses the inner rich/direct
losses and computes only the three genuinely new Q/K alternatives. The full
format check passes `22/22` in `32.67 s`. Relative to `c11d987`, random
Attention moves `0.26475 -> 0.27616` and Linear remains `0.22338`. Public
Attention is effectively flat (`0.37578 -> 0.37531`), while the six-profile
robust mean moves `0.49933/0.39337 -> 0.49952/0.39541` and the worst case moves
`0.19096 -> 0.19903`. The old frozen trend fit would place this candidate over
21000, but the refit that includes the disappointing server result gives only
an illustrative `19613`; neither value is treated as a server forecast. A
same-environment full-check rerun measured `32.569 s` for `c11d987` and
`32.671 s` for the candidate, only 0.31% slower. Scaling the measured 236-second
server baseline would give about 237 seconds, with the usual server variance.

### Random exam-like trend dataset

The hidden exam tensors may be generated rather than copied from an open model.
The supplied sparse sample supports that possibility: it uses the same five
widely separated sequence lengths for calibration and test, has negligible
adjacent-channel correlation, and combines log-scale channel variation with
rare heavy tails.  Qwen is therefore removed from candidate selection and kept
only as an optional catastrophic-regression guard.

`random_exam_dataset.py` freezes a reproducible seed recipe and materializes
the organizer's exact `linear.pt` / `attn.pt` interfaces.  The current set has
five Linear groups and four Attention groups, independent derived seeds, and
the following coverage:

- IID, lognormal channel scaling, AR channel correlation, sparse heavy tails,
  and a public-like extreme sparse-activation family;
- GQA, MHA and MQA with head dimensions 128 and 256;
- calibration and test lengths from 16 through 1024 tokens;
- approximately 95 MB of generated NVFP4 carriers, excluded from Git because
  the tracked generator and manifest reproduce them exactly.

`online_trend_dataset.py` replayed eleven measured revisions from `d75e03a`
through `3e93bd2`.  Only a two-source standardized mixture was fitted; there
are no per-revision, per-profile or Qwen weights.  The frozen mixture is
Attention `0.6000` and Linear `0.4000`.  It reproduces the known online order
with Spearman correlation `0.9909` and `98.18%` pairwise accuracy.  The single
reversal is `5b922c8` versus `a649209`; those two have identical random
Attention outputs and their random Linear proxy moves opposite to the server.
The affine point fit has mean absolute error about 282 points, so the dataset
is a trend/ranking gate, not an absolute leaderboard predictor.

The generated dataset passes the organizer interface validator `94/94` on the
current worktree in 19.23 seconds.  Future candidates should first improve the
random trend score, then pass public/Qwen regression guards and a separate
runtime gate.  A local gain on Qwen alone is no longer promotion evidence.

### Calibration-guarded direct fallback (local 20000 gate passed)

The random set exposed a distribution-selection failure rather than a need for
another unconditional Hessian pass.  On `3e93bd2`, Linear scores `-0.0436` on
the IID family and `-0.5715` on the public-like extreme-tail family, while the
paired Q/K transform is negative on the correlated Attention family.  The new
candidate retains the richer algorithms only when small deterministic
calibration projections support them; otherwise both operands use the refined
direct converter.

- Linear evaluates three calibration samples, at most 16 tokens and 128
  output rows.  It requires 2% lower mean residual and agreement on alternating
  folds.  The output residual is expanded from activation and weight errors;
  no calibration `A @ W` target is materialized.
- Q/K evaluates three samples with at most 64 tokens and requires 5% lower
  aggregate full-plus-causal Attention output error.  A rejected first version
  also required alternating-fold agreement; that rule incorrectly rejected a
  strong sparse-tail path on an independent seed because the folds had
  different sequence lengths.
- V adds two fixed-scale analytic full/causal prefix updates after the existing
  nested 4/8/16-token exchange.  In isolation this is only a small signal
  (`18872 -> 18922`); the calibration guards provide the material gain.

Final random-trend results:

| dataset seed | `3e93bd2` local estimate | candidate local estimate |
|---|---:|---:|
| fitted `20260908` | 18872 | **20818** |
| holdout `20261017` | 18903 | **20784** |
| holdout `20261129` | 19217 | **20645** |

On the fitted set, Attention moves `0.229892 -> 0.264748` and Linear moves
`0.100349 -> 0.223376`.  The first holdout moves `0.237097/0.086216 ->
0.264399/0.220593`; the second moves `0.232304/0.131273 ->
0.258739/0.219474`.  These point values use the frozen affine trend mapping and
are not claims about the server result.

The public/Qwen guard matrix remains non-regressive: candidate values are
Attention `0.37578/0.92816` and Linear `0.82433/0.46044`; the corresponding
`3e93bd2` values are `0.37382/0.92800` and `0.82433/0.46044`.  Six independent
Attention structures covering MHA/GQA/MQA and dimensions 64/128/256 improve
from `0.46334/0.36599` to `0.46373/0.36882` for full/causal means.

Same-process public API timing is `37.288 s` for `3e93bd2` and `38.472 s` for
the candidate, a 3.17% increase.  Scaling the measured 232-second server
baseline by that ratio gives an illustrative 239-second runtime, leaving about
61 seconds below the hard 300-second cutoff.  The output-format checks pass
`22/22` on the organizer sample and `94/94` on the generated random set.

The user reported `fe4b879` at 17675 points. It confirms a 167-point gain from
removing the 0.25 damping while leaving the operation count unchanged. Server
time has not yet been reported. The remaining gap to 20000 is 2325 points.

The user reported `be6ffae` at 17508 points and 254 seconds. The isolated
Linear mechanism gains 68 points for 2 seconds. Its small score contribution,
despite measurable local Linear gains, makes further Linear complexity a low
priority. Current scoring weights remain unknown; do not infer exact weights
from this single ablation. The remaining gap to 20000 is 2492 points.

The user reported `988385e` at 17440 points and 252 seconds. Relative to its
parent, the isolated V mechanism gains 449 points for 2 seconds. It is promoted
as the best verified source; 2560 points remain to the 20000 minimum target.

The user reported the `3c40705` result on 2026-09-06. It gains 216 points
and takes 8 seconds more than `def4524`; 3009 points remain to the target.
Centering and mantissa refinement were submitted together, so their separate
server contributions are unknown. Local proxy gains are not server points.

### Undamped V coupling (server gain confirmed)

The V exchange objective already estimates its permutation-invariant token
coupling from calibration attention probabilities. The candidate changes only
the final damping factor from 0.25 to 1.0, using the full measured coefficient.
It performs exactly the same four legal-mantissa updates and adds no dynamic
operation relative to `be6ffae`.

- Public Attention full/causal NMSE: `0.00393719/0.00459384` to
  `0.00390828/0.00456076`.
- Across three independent six-profile synthetic matrices, both full and
  causal means improved in every matrix. The original/candidate means were
  `0.48953/0.36484 -> 0.49526/0.37461`,
  `0.41013/0.34310 -> 0.41654/0.35334`, and
  `0.41108/0.34841 -> 0.41751/0.35457`.
- Captured Qwen full/causal means improved from `0.9366/0.9135` to
  `0.9382/0.9150`; this is a confirmation set, not the selection set.
- A coupling of 2.0 was worse than 1.0 on the robust matrix, providing a clear
  local optimum rather than an unchecked stronger-is-better trend.
- Official format check: 22/22. Runtime code paths and loop counts are
  unchanged, so expected server time remains near the 254-second baseline.

Server result: 17675 points, a confirmed gain of 167 over `be6ffae`. Runtime is
still awaiting measurement.

### Current candidate: nested 8/16-token V coupling

The confirmed 16-token exchangeable V objective is decomposed into a shared
16-token term and two nested 8-token terms. Both coefficients are estimated in
the same calibration softmax pass. Dynamic rounding keeps the same 16-token
shape, four update rounds, and one scatter per round; it adds only one small
sum and cost term. The candidate is stacked on the output-identical performance
commit, so it remains below the measured runtime of `fe4b879` locally.

- Four independent robust seeds all improve the full+causal aggregate. Mean
  full changes `0.463125 -> 0.462486`, while the larger causal gain is
  `0.384078 -> 0.387364`.
- Public full/causal NMSE improves from `0.00390828/0.00456076` to
  `0.00388800/0.00453707`.
- Captured Qwen means improve slightly, and every layer improves on the
  full+causal aggregate; Qwen was confirmation-only.
- Public V-kernel timing is unchanged within noise, while end-to-end public
  Attention measured `8.85 -> 7.90 s` after the output-identical speedups.
- Official format check: 22/22.

This is a pending online candidate. Do not infer server points from the local
gain; its purpose is to test whether a more faithful V covariance structure
continues the server-confirmed V direction.

### Current candidate: dynamic quotient-space K centering

Revision `a2b0de7` replaces the calibration-fitted K centering guard with a
test-time choice between the original K and a per-head mean-centered K.  For a
given head, subtracting the same vector from every K token changes each query's
logits only by one row-wise constant.  Softmax removes that constant exactly,
so this is a true model symmetry rather than a Qwen-specific approximation.

Both legal HiF4 candidates are scored after removing the token-constant error,
using the already stored rank-8 Q covariance.  Mean centering is selected per
KV head only when this quotient-space loss improves by at least 1%.  The final
K Hessian sweep and mantissa refinement run only once on the selected candidate.

- Four-seed combined regression on top of `04a23c1`: robust full/causal
  relative score `0.46923/0.38963 -> 0.47278/0.39121`; worst case
  `0.07652 -> 0.11949`.
- Public Attention relative score: `0.34223/0.33474 -> 0.37383/0.37176`.
- All three captured Qwen layers are numerically unchanged in the end-to-end
  score.  Qwen served only as a rejection set.
- An eight-seed test against the online `fe4b879` parent improved robust means
  by `+0.00666` full and `+0.00321` causal, and raised the worst score by
  `+0.05682`; five of eight seed aggregates improved.
- The old guard quantized sampled Q/V and two K candidates and evaluated both
  full and causal attention during calibration.  Removing it more than covers
  the extra basic K candidate: the four-seed/public/model matrix measured
  `49.97 -> 34.00 s`, public end-to-end measured `16.54 -> 16.17 s`, and the
  official output-format check passed 22/22 in 24.15 s.

The candidate is structurally better motivated and locally faster, but its
score and runtime remain unconfirmed until an exam-server submission.

### Current candidate: nested 4/8/16-token V coupling

Revision `92ed4fe` adds a four-token level to the existing nested V objective.
The measured average within-group correlations at lengths 4, 8 and 16 are
decomposed into non-negative hierarchical coefficients.  Dynamic rounding adds
one four-token reduction and one cost term, while retaining exactly four update
rounds and one scatter per round.

- Across four independent robust seeds, every full+causal aggregate improves;
  causal improves in all four and full improves in three.  Mean full/causal
  changes `0.463647/0.388722 -> 0.463849/0.389570`.
- Public full/causal relative score changes
  `0.373832/0.371760 -> 0.375203/0.373196`.
- Captured Qwen mean changes `0.953928/0.939556 -> 0.953943/0.939725`.
  One layer's full score moves down by only `0.000026`, while its causal score
  and the three-layer aggregate improve.
- Public V-kernel timing is unchanged within noise (`0.2580 -> 0.2492 s`).
  The official format check passes 22/22, and the integrated public V outputs
  match the independent prototype across all 25 parameter tensors.

This is an isolated, near-zero-cost extension on top of the larger dynamic K
candidate.  Its online contribution must be judged separately from `a2b0de7`.

### Current candidate: appended quotient-space K mantissa refinement

Revision `9fa93f2` retains the existing K Hessian and mantissa result, then
performs one additional fixed-scale coordinate pass.  Its loss subtracts the
mean reconstruction error over tokens before applying the already stored
rank-8 Q covariance.  This exactly removes the K-error component that changes
each softmax row only by a constant.  The pass does not add a factorization,
increase the Hessian rank, or search new global/local scales; each block is
replaced only after at least 1% improvement in the quotient-space objective.

- Four-seed robust full/causal relative score changes
  `0.472565/0.391965 -> 0.478234/0.394402`; the worst case changes
  `0.124237 -> 0.143826`.  Both full and causal means improve in every seed.
- Public changes `0.375203/0.373196 -> 0.373803/0.373821`: full regresses
  `0.001400`, while causal improves `0.000625`.
- All three Qwen layers improve in both full and causal evaluation.  The
  combined public/Qwen mean changes
  `0.797647/0.779994 -> 0.798219/0.780683`.
- The 4-seed/public/model matrix takes `36.61 -> 39.36 s`, an increase of
  2.76 seconds.  This remains well below the time removed by dynamic K's old
  calibration proxy.  The official format check passes 22/22 in 27.2 seconds.
- The integrated implementation matches the independent stacked prototype on
  all 60 selected K parameter tensors and their end-to-end scores.

The public full regression made this less certain than the dynamic centering
candidate.  The submitted `8d88fbc` result is `18182/230 s` against the clean
`768a670 = 18178/233 s` parent: only +4 points.  Do not spend more runtime on
this refinement family; the next algorithmic work must target a different
Attention/V bottleneck.

The user clarified that 20000 is the minimum competitive algorithm target,
motivated by another entrant reportedly scoring 22000. There is no known
20000-point source revision or official standard-converter score. Research must
seek a stronger algorithmic approach, not assume small gains near 18000 suffice.

`13a718a` is the explicit source rollback from rejected `097c2e5` to the
`def4524` numerical path. The rejected implementation remains recoverable from
Git and must not silently return in a later candidate.

## Conclusions supported by the current server

### Linear output-error rounding (server gain confirmed)

The next isolated candidate starts from `988385e` (17440 / 252 s) and changes
only Linear weight mantissas. It preserves the existing scales and estimates
off-diagonal covariance reliability from calibration second/fourth moments.
Sampling uncertainty suppresses the additional correlation term; the retained
strength is at most 0.25. One coordinate pass must improve the approximate
block objective by at least 1%. No Linear A@W calibration target is computed.

- Public Linear mean output NMSE: -7.7452%; eight real-model layers: -4.2177%.
- Nine-family synthetic development/holdout: -3.1458% / -2.7979%.
- No regression among the 27 configurations or 91 individual tests tested.
  These changes cannot be converted directly to server points.
- The earlier fixed-strength candidate slightly regressed some synthetic
  families. Estimating sampling uncertainty preserves the independent-noise
  cases and retains gains on correlated cases.
- The final kernel skips stationary blocks using a conservative gradient
  bound. It exactly matches the unpruned prototype on public, real-model and
  synthetic holdout data. All previous functions except Linear calibration
  are unchanged, including Q/K/V and dynamic activation.
- Official format check 22/22. Same-process four-thread times: control 23.5368
  s, candidate 24.9988 s, control 23.1877 s. Relative to the control mean the
  increase is 7.00%; multiplying 252 s gives a rough 269.7 s estimate, not a
  guaranteed server runtime. The hard limit remains 300 s.
- Actual server result: 17508 / 254 s. The local 269.7 s extrapolation was
  conservative; runtime scaling is workload-dependent and is not a guarantee.

Two V extensions were not promoted: a hierarchy/global-scale coordinate pass
improved public output NMSE by only about 0.03% while markedly increasing V
runtime; adding cross-segment error coupling yielded only about 0.1% there.
Their implementations remain local research artifacts, outside the submission.

### 2026-09-06: V output-error research (server gain confirmed)

The next candidate changes only V mantissa rounding on top of `3c40705`.
It minimizes a coupled 16-token loss, sum(e^2) + beta * sum(e)^2, with one
calibration-derived coefficient per KV head. The coefficient retains average
probability coupling rather than learned position-specific entries. Four
greedy legal-mantissa updates each strictly lower this approximate objective;
the existing global/lv2/lv3 scales are unchanged.

- Public Attention output NMSE: full -3.8153%, causal -3.4215%.
- Four-model/eight-layer mean relative NMSE: -10.3057% (full/causal separately).
- Generic RoPE stress: -8.4373%; 12 synthetic holdout configurations: -3.2461%;
  three long-sequence configurations: -3.6892%.
- All 64 configuration/mask averages improve; the worst individual sample
  regresses 0.0488%. These are local errors, not estimated server points.
- A learned 16x16 positional metric was rejected: one model-layer full error
  increased about 140%. Sequence-wide dual rounding was also not selected.
- Independent FP32 Golden, pinned parent `3c40705`, 18 objective/grid/shape
  checks, kernel parity over 20 configurations, and complete public API
  integration checks passed. Q/K states and outputs remain unchanged.
- Official format check: 22/22. Same-process times, four CPU threads: parent
  22.52 s, candidate 22.57 s, parent 22.99 s. Server result: 17440 / 252 s.

This is an isolated direction-validation candidate. It must not be described
as a 20000-point solution before server evaluation.

1. The hard timeout is 300 seconds. A local gain is irrelevant if the server
   run crosses that boundary.
2. K-only, calibration-prefactored Hessian refinement is useful: the move to
   `def4524` increased the score by 775 while reducing measured server time.
3. Dynamic Q Hessian refinement is not worth its cost. It contributed to the
   `237b142` timeout and Q-only local ablation did not improve the real-model
   proxy.
4. Increasing Linear rank from 32 to 40, adding a second sweep, and adding
   position/output-aware K curvature did not change the online score. These are
   not promoted even though selected local metrics improved.
5. V importance and adaptive Q/K smoothing have not shown an isolated online
   gain. Treat them as unproven rather than automatically stacking more logic.
6. Only `solution.py` is required in the contest archive, at the ZIP root.

## Validation-set bias discovered

The captured Qwen2.5-0.5B data has 14 Q heads, 2 KV heads, head dimension 64,
and calibration sequence lengths 16--80. The organizer public Attention sample
has 16 Q heads, 2 KV heads, head dimension 256, and lengths up to 1024.

The rejected `097c2e5` candidate illustrates the mismatch:

- Qwen Linear proxy: `0.41823 -> 0.42704`;
- Qwen Attention full/causal: `0.71723/0.66891 -> 0.72019/0.67516`;
- public full-Attention proxy: `0.30040 -> 0.30005` (slight regression);
- server: `16775/242 s -> 16775/244 s`.

Therefore Qwen is a diagnostic case, not a tuning target. The public mini
sample is also a diagnostic case, not an online-score estimator.

## Promotion gates for future candidates

A candidate may be pushed for server evaluation only when all of the following
hold:

1. It changes one attributable mechanism, or its components have local
   ablations.
2. It passes the official output-format check (`22/22`).
3. It is tested across head dimensions 64, 128, and 256; MHA, GQA, and MQA;
   causal and full Attention; and short and long sequences.
4. It does not rely on a Qwen-only average improvement. No distribution family
   may suffer a large regression to obtain a better overall mean.
5. Its runtime is compared with `def4524` in the same process and environment.
   The target is at most 270 seconds extrapolated server time, leaving at least
   30 seconds for variance.
6. A no-score-change server result is rejected even when local proxies improve.

## High-risk ideas not to repeat blindly

- Full dynamic Q+K Hessian refinement: timed out.
- More Linear Hessian rank or sweeps: consumed time without an online gain.
- Fine positional K buckets: short-sequence Qwen tuning did not transfer.
- Calibration-trained dense affine rotations or seed searches: historical
  organizer experiments showed catastrophic hidden-set sensitivity.
- Bundling several unablated changes: a single aggregate score cannot identify
  which component helped or hurt.

## Local candidates rejected after the baseline rollback

- Per-KV-head Smooth-QK alpha selection used three conservative candidates and
  a calibration logit guard. At a 7.5% replacement threshold it preserved the
  public sample and five of six synthetic structure families, but regressed two
  of three Qwen layers severely after K-Hessian refinement. Root cause: the
  selector optimized direct Q/K quantization while the final K path used a
  different Hessian objective. The implementation was removed.
- Diagonal Linear Weight/Activation compensation tested strengths 0.25, 0.5,
  0.75, and 1.0. Mean Qwen changes were below 0.0001 and individual layers
  disagreed in direction. The extra state and multiply were removed.
- Cross-block Linear low-rank selection was tested in two forms. Choosing
  between the rank-32 result and its plain-MSE predecessor improved the public
  combined proxy by only `0.000014` and added about 0.34 local seconds. A wider
  coordinate sweep over local hierarchy choices added roughly 2--4 seconds;
  strict row guards rejected nearly all changes, while relaxed guards reduced
  the Qwen Linear proxy from `0.41823` to `0.35103`. Both forms were removed.
- A non-Hessian two-stage fixed butterfly rotation was tested on Linear while
  leaving Attention unchanged. It preserved the floating-point operator and
  added only about 0.6 local seconds, but all three Qwen layers regressed and
  the mean proxy fell from `0.4182` to `0.4132`. One signed Hadamard stage is
  retained; adding more distribution mixing is not assumed to be beneficial.
- A calibration-derived lane permutation improved all three captured Qwen
  Linear layers but reduced the nine-family synthetic mean and strongly hurt
  correlated/low-rank families. It was rejected as a direct example of Qwen
  overfitting.
- A full-width power-of-two Linear Hadamard (H64 followed by block-axis
  mixing) reduced the non-Qwen public Linear proxy from `0.75724` to `0.74174`
  and regressed the multi-distribution matrix. HiF4's 64-value global-scale
  boundary should remain aligned with the rotation boundary.
- Rank-1/rank-2 calibration PCA Householder rotations were tested as a
  non-Hessian replacement. They failed on heavy-tail and sparse-outlier
  families because a few learned directions do not spread general outliers as
  reliably as H64. They were removed.
- A second fixed signed-Hadamard stage for Q/K preserved the exact floating-
  point attention logits but reduced the robust Attention mean from
  `0.46303/0.32948` to `0.39043/0.26784`; the worst case became negative. It
  was removed without server submission.
- Replacing the 12+5 scale search with one 17-point grid slightly improved the
  public Linear proxy (`0.75724 -> 0.75755`) but reduced robust full Attention
  (`0.46303 -> 0.44879`) and was slower locally. A one-candidate closed-form
  least-squares scale update was faster on public Linear (`23.58 -> 22.73 s`)
  but regressed both public Linear (`0.75724 -> 0.75677`) and robust Attention
  (`0.46303/0.32948 -> 0.45821/0.32812`). Both scale-search replacements were
  removed.

## Distribution-level Linear ablation

`robust_linear_benchmark.py` covers iid, heavy-tail, one-sided heavy-tail,
correlated, correlated-heavy-weight, channel-scale, sparse-outlier, and
low-rank families at widths 256--1024. It can also consume the public Linear
file and captured real-model bundles. The main finding is that no single
mechanism dominates every distribution: diagonal Weight/Activation weighting
is the safest generic path, Smooth handles channel heterogeneity, Hadamard
handles unstructured tails, and low-rank Hessian refinement is valuable on
model-like correlated data. The full current route remains best on both the
non-Qwen public matrix (`0.75724`) and captured Qwen (`0.41823`), so Hessian is
retained as an evidence-backed component rather than treated as the only
research direction.

## Rejected cross-validation candidate

K refinement now searches against the full calibration Q covariance but
accepts a changed block only when it also improves a rank-8 covariance built
from alternating calibration samples. Validation is evaluated only for blocks
that pass the primary guard. This permits lowering the primary improvement
threshold from 10% to 5% without selecting calibration-fragile changes.

- robust six-profile Attention: full `0.46303 -> 0.46463`, causal
  `0.32948 -> 0.33090`, worst `0.03461 -> 0.06076`;
- non-Qwen public Attention: full `0.30040 -> 0.30109`, causal
  `0.29539 -> 0.29636`;
- captured Qwen Attention: full `0.71723 -> 0.71675`, causal
  `0.66891 -> 0.66870`.

This candidate was intentionally selected by non-Qwen gains. The official
check passed `22/22` in 26.8 s.

The same cross-validation framework enables a bounded per-KV-head Smooth-QK
selector over alpha `{0.25, 0.34375, 0.4375}`. It scores at most three 32-token
calibration samples with the final Q/K quantizers, requires improvement on
both alternating halves, and applies a logit guard. Unlike the rejected direct
selector, its calibration objective includes the final K refinement.

- robust six-profile Attention with selection: full `0.47213`, causal
  `0.33617`, worst `0.06076`;
- non-Qwen public Attention remains `0.30109/0.29636`;
- captured Qwen improves slightly to `0.7181/0.6711`.

The public Attention path added roughly 7--8 local seconds during calibration;
the official full check remained inside the prior 25--29 second range. Online,
however, `e0a19b0` regressed from `16775/242 s` to `16700/250 s`. The runtime
increase matched the local warning, while every local quality proxy predicted
the wrong ordering. The candidate is therefore rejected and `solution.py` is
restored to the `def4524` numerical path. Do not resume alpha selection or
K-validation threshold tuning without a new server-correlated objective.

## Next research direction

Keep the `def4524` runtime path and search for a structural improvement that is
nearly free dynamically. The next target must change the actual HiF4 coding or
scale allocation rather than selecting among more calibration-fitted Attention
paths. Prefer fixed transforms, codebook/scale phase choices, and mechanisms
whose dynamic overhead is negligible. Treat all local Qwen, public, and
synthetic scores as rejection tests only until a proxy is shown to preserve
the ordering of multiple online submissions.

## Numerically identical runtime tuning

The generic scale-search chunk was swept on the public Linear case without
changing any candidate, tie break, or output tensor. Local times were about
`22.54 s` at 1024 blocks, `22.33 s` at 2048, `23.5 s` at the old 4096,
`21.69--21.80 s` at 8192, and `23.52 s` at 16384. The search chunk is therefore
8192; the Hessian chunk remains 8192. This is a performance-only change, so its
expected score is exactly the `def4524` baseline while freeing server headroom.

After the confirmed `fe4b879` result, three additional implementation changes
preserve every HiF4 output bit: candidate loss now reuses its temporary tensor
in place; the redundant refinement multiplier 1.0 is omitted from the fast
path; and K/Linear mantissa refinement keeps Hessian factors grouped by channel
block instead of copying them once per row. Public Weight, five Activations,
and all five Q/K/V test outputs matched `fe4b879` element-for-element. The
generic quantizer microbenchmark improved by roughly 9--12.5%; grouped Linear
rounding improved by roughly 35% and removed a large repeated-factor buffer.
The expected server saving is about 15--25 seconds, but only an online timing
can confirm that estimate.

V rejection tests after `fe4b879`: standalone token groups 8, 32, and 64 did
not beat group 16 on the full/causal aggregate; converting the measured
correlation with `r/(1-r)` slightly regressed multi-seed results because the
equal-diagonal approximation is imperfect. K midrange centering was also
strongly worse than mean centering when forced on. None of these changes is in
the submission path.

A low-cost Q candidate weighted its diagonal quantization objective with the
calibration K second moment.  Its best square-root weighting improved only two
of four robust seeds, slightly reduced the causal mean and worst case, reduced
Qwen layer-0 full attention, and added about 6.8% in its same-batch timing.  It
is rejected; dynamic Q Hessian or additional Q weighting should not be restored
without a different generalization argument.

Two extensions to the dynamic K selector were also rejected.  Re-centering
from the direct candidate's mean residual hurt causal attention immediately.
One fixed-point step after mean centering improved the four-seed mean, but only
two seeds won under its conservative gate, its gain was concentrated in the
short d64 profile, q-heavy causal attention regressed, and it added about 6.1%
to the matrix runtime.  The submitted selector therefore remains exactly
`none/mean`, with one final Hessian refinement.

Using the exact coordinate gradient instead of the 16-token residual sum for V
rounding changed robust means by only a few millionths and improved the public
score by about `0.0001`; it is below the promotion threshold.  A bit-exact
low-rank Hessian implementation eliminated a 64 MiB repeated-factor temporary,
but the full CPU kernel was 2.5% slower despite a faster isolated projection.
Neither change is in the submission path.

Extending the V hierarchy once more to 2/4/8/16 regressed both public metrics
(`0.375203/0.373196 -> 0.374927/0.373030`) and was stopped before a larger
sweep.  Replacing the existing K mantissa pass with the quotient-space pass was
also rejected: its one-seed robust causal mean fell
`0.389681 -> 0.387343`.  Only the appended, guarded form in `9fa93f2` passed
the multi-seed gate.

Two later V experiments were also stopped.  Adding shifted 4/8/16-token
partitions produced only a tiny public causal gain while reducing the first
robust seed's full score by `0.0021`.  Replacing the exchangeable hierarchy
with calibration-fitted 16-token position Hessians was much worse: even the
mildest tested off-diagonal strengths reduced both public full and causal
scores, and stronger variants became negative.  Position-specific calibration
correlations therefore do not transfer reliably; keep the permutation-
invariant V hierarchy unless a new independent signal is available.

A one-round V local-scale pass under the existing nested objective changed the
public full/causal score only `0.373803/0.373821 -> 0.375184/0.375064` while
roughly doubling the isolated refinement-kernel time.  A second round and
nearby global-scale candidates added negligible quality.  This is below the
promotion threshold after the server showed that much larger K proxy changes
were worth only four points, so no V scale pass is included in `0ff81b5`.
