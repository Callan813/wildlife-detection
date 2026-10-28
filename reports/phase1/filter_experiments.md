# Phase 1: filter experiments

1,200 events. Frames per event: median 3, 26% are a single frame. Events per camera: median 2, max 8.

| Rule | Target recall | Threshold | Recall | Empty events removed |
|---|---|---|---|---|
| Max score (baseline) | 98% | 0.01 | 96.5% (misses target) | 28.0% |
| Max score (baseline) | 95% | 0.01 | 96.5% | 28.0% |
| Max score (baseline) | 90% | 0.06 | 90.2% | 65.0% |
| Burst: 2 frames agree | 98% | 0.01 | 94.0% (misses target) | 35.2% |
| Burst: 2 frames agree | 95% | 0.01 | 94.0% (misses target) | 35.2% |
| Burst: 2 frames agree | 90% | 0.01 | 94.0% | 35.2% |
| Burst: 3 frames agree | 98% | 0.01 | 90.3% (misses target) | 40.2% |
| Burst: 3 frames agree | 95% | 0.01 | 90.3% (misses target) | 40.2% |
| Burst: 3 frames agree | 90% | 0.01 | 90.3% | 40.2% |
| Burst: same box in 2 frames | 98% | 0.01 | 84.7% (misses target) | 37.8% |
| Burst: same box in 2 frames | 95% | 0.01 | 84.7% (misses target) | 37.8% |
| Burst: same box in 2 frames | 90% | 0.01 | 84.7% (misses target) | 37.8% |
| Burst: same box in 3 frames | 98% | 0.01 | 75.5% (misses target) | 44.3% |
| Burst: same box in 3 frames | 95% | 0.01 | 75.5% (misses target) | 44.3% |
| Burst: same box in 3 frames | 90% | 0.01 | 75.5% (misses target) | 44.3% |
| Camera background (IoU 0.5) | 98% | 0.01 | 95.0% (misses target) | 29.3% |
| Camera background (IoU 0.5) | 95% | 0.01 | 95.0% | 29.3% |
| Camera background (IoU 0.5) | 90% | 0.03 | 91.3% | 53.5% |
| Camera background (IoU 0.7) | 98% | 0.01 | 95.5% (misses target) | 28.5% |
| Camera background (IoU 0.7) | 95% | 0.01 | 95.5% | 28.5% |
| Camera background (IoU 0.7) | 90% | 0.04 | 91.2% | 58.8% |
