# Phase 1: filter experiments

1,200 events. Frames per event: median 3, 26% are a single frame. Events per camera: median 2, max 8.

| Rule | Target recall | Threshold | Recall | Empty events removed |
|---|---|---|---|---|
| Max score (baseline) | 98% | 0.02 | 98.7% | 8.7% |
| Max score (baseline) | 95% | 0.05 | 95.7% | 24.0% |
| Max score (baseline) | 90% | 0.15 | 90.0% | 53.0% |
| Burst: 2 frames agree | 98% | 0.01 | 98.8% | 5.2% |
| Burst: 2 frames agree | 95% | 0.02 | 96.2% | 15.8% |
| Burst: 2 frames agree | 90% | 0.02 | 96.2% | 15.8% |
| Burst: 3 frames agree | 98% | 0.01 | 96.7% (misses target) | 12.5% |
| Burst: 3 frames agree | 95% | 0.01 | 96.7% | 12.5% |
| Burst: 3 frames agree | 90% | 0.02 | 91.0% | 27.3% |
| Burst: same box in 2 frames | 98% | 0.01 | 92.2% (misses target) | 10.0% |
| Burst: same box in 2 frames | 95% | 0.01 | 92.2% (misses target) | 10.0% |
| Burst: same box in 2 frames | 90% | 0.01 | 92.2% | 10.0% |
| Burst: same box in 3 frames | 98% | 0.01 | 82.5% (misses target) | 25.5% |
| Burst: same box in 3 frames | 95% | 0.01 | 82.5% (misses target) | 25.5% |
| Burst: same box in 3 frames | 90% | 0.01 | 82.5% (misses target) | 25.5% |
| Camera background (IoU 0.5) | 98% | 0.01 | 98.7% | 4.3% |
| Camera background (IoU 0.5) | 95% | 0.02 | 97.3% | 11.7% |
| Camera background (IoU 0.5) | 90% | 0.08 | 90.7% | 42.2% |
| Camera background (IoU 0.7) | 98% | 0.02 | 98.0% | 10.0% |
| Camera background (IoU 0.7) | 95% | 0.02 | 98.0% | 10.0% |
| Camera background (IoU 0.7) | 90% | 0.10 | 91.0% | 45.2% |
