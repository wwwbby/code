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
3. searches the five nearest legal E6M2 candidates;
4. exactly minimizes elementwise squared error over legal level-2 and level-3
   micro-scale choices;
5. rounds and clamps the 64 leaves to legal S1P2 values.

Before quantization, Linear inputs and weights receive the same deterministic
signed H4 rotation inside each four-value level-3 group. Q and K receive the
same signed H8 rotation per attention head inside each eight-value level-2
group. These transforms are orthogonal, so they preserve the unquantized
Linear output and Q/K logits while spreading local outliers. V is not rotated
because the public interface has no post-attention inverse-transform hook.

The submission code performs no file I/O and never computes the prohibited
Linear `A @ W` calibration target.

## Local results

Using the synthetic benchmark committed here:

- Linear score mean: `0.459692`
- Attention score mean: `0.435800`
- Combined score sum: `4.477461`
- Combined score mean: `0.447746`

The rotation states contain only CPU tensors and primitive values, and the
generated output shapes and HiF4 values pass the local contract checks. Run the
organizer's `self_check.py` against the official mini sample before submission.

CPU stress test on the development machine:

- `1024 x 4096` tensor: about `0.47-0.65 s`
- `128 x 4096` tensor: about `0.06-0.08 s`

These are proxy results, not the score of the contest's hidden evaluation set.

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
