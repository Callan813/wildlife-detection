# Phase 1 results: empty-frame filtering

Model: `MegaDetectorV6MIT/MDV6-mit-yolov9-c`
Data: 5,726 frames in 1,200 trigger events (600 animal, 600 empty) from 507 camera locations.

## Operating point (target event recall 98%)

The target is out of reach: even at the lowest threshold, 27 animal events have no detector box at all. The row below is the lowest threshold.

| Metric | Value |
|---|---|
| Confidence threshold | 0.01 |
| Animal events kept (recall) | 95.5% |
| Precision of kept events | 58.1% |
| Empty events removed | 31.0% |
| Review workload removed | 17.8% |
| Animal events missed | 27 of 600 |

## Recall by species group at that threshold (worst first)

| Group | Events | Recall |
|---|---|---|
| pangolin | 41 | 87.8% |
| rodent | 41 | 92.7% |
| macaque | 42 | 92.9% |
| marten | 32 | 93.8% |
| junglefowl | 36 | 94.4% |
| small_cat | 41 | 95.1% |
| civet | 42 | 95.2% |
| squirrel | 42 | 95.2% |
| wild_boar | 43 | 95.3% |
| mouse_deer | 29 | 96.6% |
| porcupine | 41 | 97.6% |
| barking_deer | 43 | 97.7% |
| sambar | 43 | 97.7% |
| bear | 42 | 100.0% |
| mongoose | 42 | 100.0% |

## Latency (this machine, per image, includes image decode; batched runs report batch time / batch size)

Mean 30 ms, median 28 ms, p95 40 ms.

## Files

- `event_sweep.csv`, `image_sweep.csv`: metrics at every threshold
- `recall_by_group.csv`: per-group recall at the operating point
- `tradeoff.png`: recall vs. review workload removed

Frame-level numbers include label noise (sequence-level labels on frames where the animal has left the view), so the event-level row is the headline result.
