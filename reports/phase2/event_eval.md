# Phase 2: empty-event filtering with a crop classifier

Test: 566 events (416 animal, 150 empty) from 114 cameras not used in training.
Each threshold is the highest that keeps the target share of animal events on val, applied unchanged to test.

Cells: share of empty test events removed (test recall of animal events in brackets).
Thresholds, workload and missed counts are in event_eval.csv.

| method, events scored by | 98% target | 95% target | 90% target |
|---|---|---|---|
| detector, best box | 13.3% (97.6%) | 29.3% (94.0%) | 54.7% (90.4%) |
| detector, 2nd box | 2.7% (99.5%) | 12.7% (97.4%) | 24.7% (88.2%) |
| ours, best box | 18.7% (97.8%) | 43.3% (95.2%) | 75.3% (89.7%) |
| ours, 2nd box | 2.7% (97.1%) | 17.3% (92.5%) | 24.7% (89.2%) |
| ours x detector, best box | 12.7% (98.1%) | 26.7% (95.7%) | 65.3% (90.4%) |
| ours x detector, 2nd box | 2.7% (99.0%) | 8.0% (96.6%) | 27.3% (87.3%) |
| ours-nols, best box | 14.0% (98.6%) | 44.0% (93.8%) | 67.3% (90.4%) |
| ours-nols, 2nd box | 5.3% (96.6%) | 10.7% (94.7%) | 22.0% (88.5%) |
| ours-nols x detector, best box | 14.7% (97.6%) | 44.7% (93.8%) | 70.0% (89.7%) |
| ours-nols x detector, 2nd box | 2.7% (98.6%) | 15.3% (94.2%) | 26.0% (90.4%) |

## Test recall by species group at the 98% threshold

| group | detector, best box | detector, 2nd box | ours, best box | ours, 2nd box | ours-nols, best box | ours-nols, 2nd box |
|---|---|---|---|---|---|---|
| barking_deer | 100% | 100% | 97% | 94% | 100% | 91% |
| bear | 100% | 100% | 100% | 92% | 100% | 92% |
| civet | 100% | 100% | 100% | 100% | 100% | 100% |
| junglefowl | 96% | 100% | 100% | 100% | 100% | 100% |
| macaque | 97% | 100% | 97% | 100% | 97% | 100% |
| marten | 93% | 100% | 96% | 100% | 93% | 100% |
| mongoose | 96% | 100% | 100% | 100% | 100% | 100% |
| mouse_deer | 100% | 100% | 100% | 90% | 100% | 93% |
| pangolin | 100% | 100% | 90% | 90% | 100% | 90% |
| porcupine | 100% | 100% | 96% | 100% | 100% | 96% |
| rodent | 91% | 100% | 91% | 94% | 94% | 91% |
| sambar | 100% | 97% | 100% | 100% | 100% | 97% |
| small_cat | 92% | 100% | 100% | 100% | 92% | 100% |
| squirrel | 100% | 96% | 96% | 100% | 100% | 100% |
| wild_boar | 97% | 100% | 100% | 94% | 100% | 97% |

## Crop-level F1 on test, classes every model can name (2,936 crops)

| class | ours | ours-nols |
|---|---|---|
| background | 0.93 | 0.94 |
| barking_deer | 0.79 | 0.80 |
| bear | 0.86 | 0.93 |
| civet | 0.76 | 0.78 |
| junglefowl | 0.91 | 0.91 |
| macaque | 0.88 | 0.88 |
| marten | 0.70 | 0.69 |
| mongoose | 0.56 | 0.58 |
| mouse_deer | 0.89 | 0.85 |
| pangolin | 0.68 | 0.77 |
| porcupine | 0.82 | 0.80 |
| rodent | 0.63 | 0.77 |
| sambar | 0.75 | 0.75 |
| small_cat | 0.72 | 0.63 |
| squirrel | 0.50 | 0.50 |
| wild_boar | 0.89 | 0.90 |

Macro F1: ours 0.767, ours-nols 0.781
