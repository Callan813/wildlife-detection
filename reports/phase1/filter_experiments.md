# Phase 1: filter experiments

1,200 events. Frames per event: median 3, 26% are a single frame. Events per camera: median 2, max 8.

| Rule | Target recall | Threshold | Recall | Empty events removed |
|---|---|---|---|---|
| Max score (baseline) | 98% | 0.01 | 95.5% (misses target) | 31.0% |
| Max score (baseline) | 95% | 0.01 | 95.5% | 31.0% |
| Max score (baseline) | 90% | 0.05 | 90.0% | 66.0% |
| Burst: 2 frames agree | 98% | 0.01 | 91.8% (misses target) | 40.2% |
| Burst: 2 frames agree | 95% | 0.01 | 91.8% (misses target) | 40.2% |
| Burst: 2 frames agree | 90% | 0.01 | 91.8% | 40.2% |
| Burst: 3 frames agree | 98% | 0.01 | 87.8% (misses target) | 46.3% |
| Burst: 3 frames agree | 95% | 0.01 | 87.8% (misses target) | 46.3% |
| Burst: 3 frames agree | 90% | 0.01 | 87.8% (misses target) | 46.3% |
| Burst: same box in 2 frames | 98% | 0.01 | 81.5% (misses target) | 43.0% |
| Burst: same box in 2 frames | 95% | 0.01 | 81.5% (misses target) | 43.0% |
| Burst: same box in 2 frames | 90% | 0.01 | 81.5% (misses target) | 43.0% |
| Burst: same box in 3 frames | 98% | 0.01 | 72.3% (misses target) | 50.5% |
| Burst: same box in 3 frames | 95% | 0.01 | 72.3% (misses target) | 50.5% |
| Burst: same box in 3 frames | 90% | 0.01 | 72.3% (misses target) | 50.5% |
| Camera background (IoU 0.5) | 98% | 0.01 | 93.8% (misses target) | 32.5% |
| Camera background (IoU 0.5) | 95% | 0.01 | 93.8% (misses target) | 32.5% |
| Camera background (IoU 0.5) | 90% | 0.02 | 90.7% | 47.2% |
| Camera background (IoU 0.7) | 98% | 0.01 | 94.5% (misses target) | 31.5% |
| Camera background (IoU 0.7) | 95% | 0.01 | 94.5% (misses target) | 31.5% |
| Camera background (IoU 0.7) | 90% | 0.04 | 90.0% | 62.7% |
