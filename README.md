# HiF4 quantization demo

Huawei algorithm contest demo for converting NVFP4 carrier + block scales into
legal HiF4 parameters for Linear and Attention workloads.

## Files

- `solution.py`: the six public contest APIs and quantization implementation.
- `local_proxy_benchmark.py`: reproducible synthetic data generator and proxy
  scorer. It compares the demo with the paper-style peak conversion baseline.

## Algorithm

For every 64-value HiF4 block, the implementation:

1. decodes the NVFP4 carrier in FP32;
2. starts from the paper-style `max(abs(x)) / 7` E6M2 scale;
3. searches 12 broad and 5 guarded refinement scale candidates;
4. exactly minimizes elementwise squared error over legal level-2 and level-3
   micro-scale choices;
5. rounds and clamps the 64 leaves to legal S1P2 values.

Linear calibration applies reciprocal SmoothQuant-style scaling and a shared
signed H64 transform to weights and activations. Weight scale selection uses
activation second moments, then a guarded two-round coordinate search minimizes
the full damped 64x64 calibration Hessian. Dynamic activation search is weighted
by the transformed weight's output sensitivity.

Q and K use reciprocal per-channel smoothing followed by the same signed H64
transform inside every attention head. V stays in its original basis because
the interface has no inverse-transform hook; its quantization instead uses a
conservative calibration-derived attention-usage weighting with plain-MSE and
weighted-MSE guards.

The paired Linear and Q/K transforms are algebraically cancelling, so they
preserve the unquantized Linear output and attention logits exactly apart from
floating-point roundoff.

The submission code performs no file I/O and never computes the prohibited
Linear `A @ W` calibration target.

## Local results

On the organizer's public mini sample:

- official output-format checks: `22/22` passed;
- Linear output NMSE cases: `0.00022815`, `0.00028119`, `0.00025422`,
  `0.00028406`, `0.00025762` (mean `0.00026105`);
- mixed causal/full Attention output NMSE: `0.00505861`;
- full self-check runtime on the development machine: about `127 s`.

The numerical path matches the public implementation associated with a reported
score above 23,000 on these sample regressions, but only a fresh hidden-set
submission can confirm the score for this repository revision.

## Run the proxy benchmark

```bash
python local_proxy_benchmark.py --solution_dir .
```

## Run the organizer self-check

Place the organizer's `self_check.py` and `mini_sample` directory next to this
repository, then run:

```bash
python ../self_check.py --solution_dir . --datasets_dir ../mini_sample
```

Create the submission archive with `solution.py` at the ZIP root:

```bash
python -m zipfile -c solution.zip solution.py
```
