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
| `def4524` | **16775** | **242 s** | **online baseline** | Rank-32 Linear plus pre-factored K-only Hessian |
| `097c2e5` | 16775 | 244 s | rejected | Rank-40/two sweeps plus position-aware output Hessian |

`13a718a` is the explicit source rollback from rejected `097c2e5` to the
`def4524` numerical path. The rejected implementation remains recoverable from
Git and must not silently return in a later candidate.

## Conclusions supported by the current server

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

## Next research direction

Keep the `def4524` runtime path and search for a structural improvement that is
nearly free dynamically. The first target is a calibration-only, distribution-
robust Q/K objective evaluated on a synthetic shape matrix. If it cannot beat
the baseline across that matrix, move to low-rank cross-block Linear error
compensation with strict row-level guards rather than increasing the existing
within-block Hessian budget.
