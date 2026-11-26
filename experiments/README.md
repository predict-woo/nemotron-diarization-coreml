# Experiments

Measurements behind the README's findings (timing-only variants are not exact and are not used by the runtime).

| File | Measures |
| --- | --- |
| `ablate.py`, `ablate2.py`, `ablate3.py`, `ablate4.py` | GPU time of attention parts, GELU, 1x1-conv MLP layout, head, per-layer cost |
| `length_sweep.py` | step latency vs fixed model length (alignment: multiples of 64) |
| `hints.py` | CoreML optimization hints (reshape frequency, fastPrediction) on fixed / flexible models |
| `weights_as_inputs.py`, `memtest/winput.swift` | per-call copy of weights passed as inputs |
| `memtest/main.swift`, `mf.swift`, `hold.swift`, `spec.swift`, `cpumem.swift` | memory of bucket models and shape specializations |
| `memtest/alt.swift` | latency when alternating bucket functions in the steady-state length cycle |
| `memtest/concurrent.swift`, `concurrent2.swift`, `shared.swift` | CPU+GPU and multi-stream concurrency |
| `memtest/sgemm.swift` | Accelerate sgemm throughput (CPU ceiling) |
| `memtest/units.swift`, `loadtime.swift` | compute units / hints, sequential vs parallel model loading |
| `cond_toy.py`, `memtest/condtest.swift` | MIL `cond` branches sharing weights (speed, memory) |
| `memtest/rangeload.swift` | load time / memory of flexible-length models |
| `ane_ablate.py` | Neural Engine time of attention core, RoPE, MLP; attention / RoPE reformulations |
| `ane_error.py` | teacher-forced fp16 error: Neural Engine vs GPU fp16 vs fp32 |
| `memtest/anesweep.swift` | Neural Engine step latency and memory per bucket length |

Run the Python ones from the repository root (`python experiments/ablate.py`), compile the Swift ones with
`swiftc -O <file>.swift` (with `SDKROOT` set as in the main README).
