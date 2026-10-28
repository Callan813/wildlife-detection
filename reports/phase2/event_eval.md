# Phase 2: empty-event filtering with a crop classifier

Test: 566 events (416 animal, 150 empty) from 114 cameras not used in training.
Each threshold is the highest that keeps the target share of animal events on val, applied unchanged to test.

25 of 416 animal test events have no detector box at or above the crop floor, so no method can keep more than 94.0% of them. Targets above what val allows are marked out of reach.

Cells: share of empty test events removed (test recall of animal events in brackets).
Thresholds, workload and missed counts are in event_eval.csv.

| method, events scored by | 98% target | 95% target | 90% target |
|---|---|---|---|
| detector, best box | out of reach | out of reach | 54.7% (92.3%) |
| detector, 2nd box | out of reach | out of reach | 49.3% (89.9%) |
| ours, best box | out of reach | out of reach | 69.3% (89.9%) |
| ours, 2nd box | out of reach | out of reach | 56.7% (89.9%) |
| ours x detector, best box | out of reach | out of reach | 60.0% (92.1%) |
| ours x detector, 2nd box | out of reach | out of reach | 55.3% (89.4%) |

## Test recall by species group at the 90% threshold

| group | detector, best box | detector, 2nd box | ours, best box | ours, 2nd box |
|---|---|---|---|---|
| barking_deer | 100% | 94% | 94% | 94% |
| bear | 100% | 100% | 100% | 100% |
| civet | 95% | 87% | 95% | 95% |
| junglefowl | 96% | 92% | 96% | 96% |
| macaque | 90% | 90% | 90% | 90% |
| marten | 85% | 85% | 74% | 78% |
| mongoose | 96% | 96% | 93% | 96% |
| mouse_deer | 98% | 100% | 98% | 100% |
| pangolin | 90% | 90% | 90% | 70% |
| porcupine | 96% | 93% | 93% | 89% |
| rodent | 82% | 79% | 85% | 82% |
| sambar | 100% | 97% | 100% | 100% |
| small_cat | 77% | 69% | 77% | 77% |
| squirrel | 77% | 69% | 62% | 65% |
| wild_boar | 92% | 94% | 92% | 92% |

## Crop-level F1 on test, classes every model can name (2,383 crops)

| class | ours |
|---|---|
| background | 0.84 |
| barking_deer | 0.81 |
| bear | 0.90 |
| civet | 0.78 |
| junglefowl | 0.93 |
| macaque | 0.88 |
| marten | 0.72 |
| mongoose | 0.49 |
| mouse_deer | 0.87 |
| pangolin | 0.84 |
| porcupine | 0.80 |
| rodent | 0.70 |
| sambar | 0.77 |
| small_cat | 0.70 |
| squirrel | 0.46 |
| wild_boar | 0.91 |

Macro F1: ours 0.776
