# Phase 6: model speed on laptop-ryzen5-5600h

AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 12 logical CPUs, Windows AMD64; onnxruntime 1.31.1, CPU execution provider, batch 1, 20 runs after 3 warm-up.
Not a Raspberry Pi: rerun this script on the device for its numbers.

| Model | MB on disk | Threads | Median ms | p95 ms | Calls per second | Load s | Memory MB |
|---|---|---|---|---|---|---|---|
| detector_fp32 | 27.7 | 1 | 539 | 546 | 1.86 | 0.9 | 117 |
| detector_fp32 | 27.7 | 2 | 291 | 320 | 3.43 | 0.8 | 105 |
| detector_fp32 | 27.7 | 4 | 187 | 208 | 5.36 | 0.8 | 106 |
| detector_int8 | 12.2 | 1 | 717 | 724 | 1.40 | 3.1 | 134 |
| detector_int8 | 12.2 | 2 | 406 | 431 | 2.46 | 3.0 | 134 |
| detector_int8 | 12.2 | 4 | 278 | 290 | 3.60 | 2.9 | 133 |
| species_fp32 | 201.4 | 1 | 1010 | 1016 | 0.99 | 2.1 | 264 |
| species_fp32 | 201.4 | 2 | 521 | 577 | 1.92 | 1.9 | 263 |
| species_fp32 | 201.4 | 4 | 298 | 348 | 3.35 | 1.9 | 265 |
| reid_fp32 | 83.8 | 1 | 538 | 545 | 1.86 | 0.6 | 132 |
| reid_fp32 | 83.8 | 2 | 276 | 285 | 3.63 | 0.5 | 131 |
| reid_fp32 | 83.8 | 4 | 178 | 186 | 5.62 | 0.5 | 131 |
| reid_int8 | 22.7 | 1 | 355 | 360 | 2.81 | 0.3 | 70 |
| reid_int8 | 22.7 | 2 | 182 | 192 | 5.50 | 0.3 | 69 |
| reid_int8 | 22.7 | 4 | 117 | 134 | 8.53 | 0.2 | 69 |
