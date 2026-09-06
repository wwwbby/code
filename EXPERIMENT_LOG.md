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
| `be6ffae` | **17508** | **254 s** | **online baseline** | Linear mantissa refinement with covariance reliability |

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

The user clarified that 20000 is the minimum competitive algorithm target,
motivated by another entrant reportedly scoring 22000. There is no known
20000-point source revision or official standard-converter score. Research must
seek a stronger algorithmic approach, not assume small gains near 17000 suffice.

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
