# Phase 1 results: empty-frame filtering

Model: `MegaDetectorV6MIT/MDV6-mit-yolov9-c`
Data: 5,726 frames in 1,200 trigger events (600 animal, 600 empty) from 507 camera locations.

## Operating point (target event recall 98%)

| Metric | Value |
|---|---|
| Confidence threshold | 0.02 |
| Animal events kept (recall) | 98.7% |
| Precision of kept events | 51.9% |
| Empty events removed | 8.7% |
| Review workload removed | 5.0% |
| Animal events missed | 8 of 600 |

## Recall by species group at that threshold (worst first)

| Group | Events | Recall |
|---|---|---|
| porcupine | 41 | 95.1% |
| marten | 32 | 96.9% |
| junglefowl | 36 | 97.2% |
| small_cat | 41 | 97.6% |
| civet | 42 | 97.6% |
| mongoose | 42 | 97.6% |
| squirrel | 42 | 97.6% |
| barking_deer | 43 | 100.0% |
| sambar | 43 | 100.0% |
| wild_boar | 43 | 100.0% |
| bear | 42 | 100.0% |
| macaque | 42 | 100.0% |
| pangolin | 41 | 100.0% |
| rodent | 41 | 100.0% |
| mouse_deer | 29 | 100.0% |

## Latency (this machine, single image, includes image decode)

Mean 2705 ms, median 826 ms, p95 1081 ms.

## Files

- `event_sweep.csv`, `image_sweep.csv`: metrics at every threshold
- `recall_by_group.csv`: per-group recall at the operating point
- `tradeoff.png`: recall vs. review workload removed

Frame-level numbers include label noise (sequence-level labels on frames where the animal has left the view), so the event-level row is the headline result.
