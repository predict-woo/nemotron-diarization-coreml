# Findings

What we learned converting Nemotron-3-Diarization to CoreML and optimizing it for Apple silicon (M2 Pro,
macOS 27, CoreML via coremltools 9). The timing and memory experiments behind each point are in
[`experiments/`](../experiments).

## The model and the original

* **Streaming step.** Each step encodes [speaker cache (≤264), FIFO (≤264), chunk + look-ahead (13)] frames of
  80 ms. The encoder is a 31-layer pre-LN Transformer with RoPE, with positions restarting at 0 every step. The
  encoder is bidirectional, so nothing carries over between steps: every step recomputes every position. Once
  the cache is full, the step length cycles between 322 and 541 positions.
* **The speaker cache stores input embeddings,** the feature-stacking projections, which never depend on the
  encoder's precision. Encoder outputs enter the cache only through probabilities. Those are read only at "pop"
  steps (when the FIFO overflows into the cache, every ~18 s of audio) to decide which frames compression keeps.
* **The two official implementations disagree.** NeMo's `.nemo` checkpoint stores the STFT window and mel
  filterbank as bfloat16-rounded values, while the HF Transformers port uses exact float32. With identical
  features the two agree to 1e-5 in logits, but end to end they differ by up to 7.6e-3 in probability. This
  project matches HF Transformers. `scripts/build_bundle.py --frontend nemo` uses NeMo's constants instead, and
  with them the runtime matches NeMo to 8e-7. NeMo also chunks the last ~12 frames of a stream differently.

## Exactness (fp32 builds)

* **Cache compression depends on top-k tie-breaking.** Confident single-speaker frames saturate the sigmoid to
  exactly 1.0, so many frames get exactly equal scores. Which of the tied frames survive is decided by PyTorch's
  CPU `topk`, which is libc++ `std::nth_element` over (value, index) pairs. A reasonable stable top-k diverges on
  a 3-minute recording and flips 13 frames. [`native/torch_topk.cpp`](../native/torch_topk.cpp) reproduces torch's
  algorithm: it matches on 4000 adversarial cases, and the Swift runtime links it.
* **Summation order matters for the same reason.** The ports sum the 8 speaker scores sequentially and average
  probabilities with a sequential sum then division, as torch does. NumPy's default `sum` is pairwise.
* **Compression is inherently sensitive.** Even the original's own fp32 and fp64 runs keep different frames on
  one recording. The fp32 CoreML builds reproduce the fp32 original's choices exactly.

## Neural Engine

1. **The ANE computes only in fp16.** It accumulates products at higher precision but rounds inputs and outputs
   to fp16, and matmul outputs saturate at 32,768. GELU, exp and similar functions are 33-point
   piecewise-linear lookup tables, with GELU error up to 0.006
   ([ane-guide](https://ane-guide.readthedocs.io/en/latest/part-1-machine/03-numerics.html)).
   * On identical inputs, the ANE's logit error is 2.8x that of the same fp16 model on the GPU, so the lookup
     tables account for most of it.
   * Computing GELU from multiplies and adds in fp16 doesn't help: a Chebyshev/Clenshaw evaluation errs by 3.8e-3,
     against the table's 5.9e-3.
2. **LayerNorm overflows in fp16 unless its input is scaled down.** The residual stream grows to ~1000, and the
   sum of squared deviations reaches 5.7M, against fp16's maximum of 65,504. LayerNorm is scale-invariant, so
   inputs are divided by 32 (and eps by 32²) first. A naive fp16 conversion without this drifts far more.
3. **Layout.** The model follows apple/ml-ane-transformers: (1, C, 1, S) tensors, 1×1 convolutions, and per-head
   einsum attention. All 1,936 operations of every bucket function land on the ANE. Alternatives that were slower
   at 544 positions (23.2 ms baseline):
   * Batched-head attention: 44.9 ms.
   * RoPE through extra projection weights instead of split/concat: 24.7 ms.
4. **Where ANE time goes at 544 positions:**

   | Part | Time | Work |
   | --- | --- | --- |
   | Attention core (incl. RoPE 2.4 ms) | 10.1 ms (44%) | ~19 GFLOP |
   | MLP | 8.2 ms | ~70 GFLOP |

   The attention core is far less efficient per FLOP.
5. **Length buckets are nearly free on the ANE.** Five functions loaded together need 0.6 GB resident: compiled
   ANE programs keep their weights outside the process. On the GPU, each function costs ~400 MB.
6. **The ANE's cost curve has plateaus,** so the buckets sit at their ends (192, 288, 384, 480, 544):

   | Length | 192 | 288 | 320 | 352 | 384 | 416 | 448 | 480 | 512 | 544 | 576 |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | Step (ms) | 6.6 | 10.3 | 11.3 | 11.9 | 12.7 | 16.7 | 17.7 | 18.3 | 21.7 | 23.0 | 23.7 |

   All eight multiples of 32 would be 3% faster, at ~15 s more first-launch compile per function. The first
   launch compiles every function (~75 s for five); macOS caches the result.
7. **Where the remaining disagreement comes from.**
   * 90% of the frames where the ANE build departs from the original trace to the ~4% of steps that compress
     the speaker cache. There, fp16 probabilities keep different, equally good frames.
   * Running only those steps in fp32 cuts the disagreement from 0.22% to 0.02% of frames. That needs the CPU or
     GPU, so it exists only as a diagnostic (`scripts/build_bundle.py --exact-model`).
   * Diarization error against ground truth is unchanged (8.27% vs 8.23%).
8. **Not available here: Core AI.** Apple's Core AI framework (macOS 27) can run one dynamic-length function, but
   it needs Xcode 27.

## GPU (fp32)

1. **Where GPU time goes.** Of a 36 ms step at full length:
   * MLP: 14.3 ms (~4.9 TFLOPS, 72% of peak).
   * Attention projections: 8.7 ms.
   * Attention core: 9.1 ms, dominated by two batched matmuls with an inner dimension of 64.
   * RoPE: ~1.2 ms.
   * Norms and head: ~4 ms.

   Lengths that are multiples of 64 are cheapest per position (~64 µs), and 541 positions cost as much as 576.
2. **Shapes.** Padding every step to 541 wastes ~20% on average.
   * **Multifunction buckets (used).** Fixed-length functions at 192/384/448/512/576 run at fixed-shape speed.
     But each function holds its own GPU copy of the weights, which the GPU backend writes to temporary
     `payload-*.bin` files and maps: ~400 MB each, file-backed.
   * **Flexible (RangeDim).** The dynamic GPU graph recompiles on every load (~30 s) and runs ~10% slower per
     position. With `reshapeFrequency = .infrequent` it loads fast, but every new length gets its own weight
     copy, and the GPU ran out of memory after a few dozen lengths.
   * **Enumerated shapes.** The GPU keeps only the most recently used shape specialized, so every switch costs
     ~330 ms.
   * **MIL `cond` branches over shared weights.** About 6x slower, and still a weight copy per branch.
   * **Weights as model inputs.** Copied on every call, +8.5 ms for 260 MB even from page-aligned memory.
3. **Concurrency.** Only pop steps depend on earlier outputs, and which steps pop is fixed by the chunk schedule.
   So whole-file mode runs the other steps concurrently:
   * Two predictions on one `MLModel` give 13–20% more throughput.
   * The CPU takes a share of the steps alongside the GPU.
   * In Python, coremltools holds the interpreter lock during `predict`, which hides all of this.
4. **fp16 is not identical,** on the GPU or the ANE: the speaker cache diverges at the first compression, and the
   GPU's fp16 throughput is no higher than its fp32 throughput (33.7 vs 36.7 ms per step).
5. **Tried, no gain:**
   * The SDPA op and the iOS 26 target.
   * Linear layers as 1×1 convolutions.
   * A manual softmax.
   * Batching independent steps: the GPU is compute-bound.
   * `.all` compute units and the `fastPrediction` hint.
   * Computing only the emitted frames in the last layer (≤3%).
   * A hand-written Accelerate CPU encoder: Accelerate's sgemm tops out at 1.1–1.7 TFLOPS.
6. **Loading.** Bucket functions load ~10x faster in parallel than one after another.
