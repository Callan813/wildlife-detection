# Phase 1: filter experiments

1,200 events. Frames per event: median 3, 26% are a single frame. Events per camera: median 2, max 8.

| Rule | Target recall | Threshold | Recall | Empty events removed |
|---|---|---|---|---|
| Max score (baseline) | 98% | 0.06 | 98.8% | 6.2% |
| Max score (baseline) | 95% | 0.08 | 95.2% | 30.3% |
| Max score (baseline) | 90% | 0.10 | 90.8% | 55.2% |
| Burst: 2 frames agree | 98% | 0.06 | 98.0% | 8.7% |
| Burst: 2 frames agree | 95% | 0.06 | 98.0% | 8.7% |
| Burst: 2 frames agree | 90% | 0.08 | 90.5% | 36.3% |
| Burst: 3 frames agree | 98% | 0.05 | 99.2% | 2.8% |
| Burst: 3 frames agree | 95% | 0.06 | 96.2% | 13.3% |
| Burst: 3 frames agree | 90% | 0.07 | 91.5% | 25.7% |
| Burst: same box in 2 frames | 98% | 0.01 | 85.8% (misses target) | 27.3% |
| Burst: same box in 2 frames | 95% | 0.01 | 85.8% (misses target) | 27.3% |
| Burst: same box in 2 frames | 90% | 0.01 | 85.8% (misses target) | 27.3% |
| Burst: same box in 3 frames | 98% | 0.01 | 76.7% (misses target) | 33.3% |
| Burst: same box in 3 frames | 95% | 0.01 | 76.7% (misses target) | 33.3% |
| Burst: same box in 3 frames | 90% | 0.01 | 76.7% (misses target) | 33.3% |
| Camera background (IoU 0.5) | 98% | 0.06 | 98.8% | 6.2% |
| Camera background (IoU 0.5) | 95% | 0.07 | 97.2% | 16.7% |
| Camera background (IoU 0.5) | 90% | 0.09 | 91.5% | 45.3% |
| Camera background (IoU 0.7) | 98% | 0.06 | 98.8% | 6.2% |
| Camera background (IoU 0.7) | 95% | 0.07 | 97.2% | 16.3% |
| Camera background (IoU 0.7) | 90% | 0.09 | 92.2% | 44.7% |
