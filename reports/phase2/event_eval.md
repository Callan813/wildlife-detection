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
| speciesnet, best box | out of reach | out of reach | 76.7% (91.3%) |
| speciesnet, 2nd box | out of reach | out of reach | 66.7% (90.1%) |
| speciesnet x detector, best box | out of reach | out of reach | 69.3% (91.8%) |
| speciesnet x detector, 2nd box | out of reach | out of reach | 64.7% (90.9%) |
| ind-ads, best box | out of reach | out of reach | 63.3% (90.9%) |
| ind-ads, 2nd box | out of reach | out of reach | 46.0% (92.1%) |
| ind-ads x detector, best box | out of reach | out of reach | 60.7% (91.8%) |
| ind-ads x detector, 2nd box | out of reach | out of reach | 49.3% (90.9%) |

## Test recall by species group at the 90% threshold

| group | detector, best box | detector, 2nd box | ours, best box | ours, 2nd box | speciesnet, best box | speciesnet, 2nd box | ind-ads, best box | ind-ads, 2nd box |
|---|---|---|---|---|---|---|---|---|
| barking_deer | 100% | 94% | 94% | 94% | 97% | 97% | 97% | 100% |
| bear | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| civet | 95% | 87% | 95% | 95% | 95% | 92% | 95% | 95% |
| junglefowl | 96% | 92% | 96% | 96% | 96% | 96% | 96% | 96% |
| macaque | 90% | 90% | 90% | 90% | 90% | 90% | 90% | 90% |
| marten | 85% | 85% | 74% | 78% | 81% | 81% | 78% | 78% |
| mongoose | 96% | 96% | 93% | 96% | 96% | 96% | 100% | 89% |
| mouse_deer | 98% | 100% | 98% | 100% | 100% | 95% | 98% | 100% |
| pangolin | 90% | 90% | 90% | 70% | 90% | 80% | 90% | 90% |
| porcupine | 96% | 93% | 93% | 89% | 96% | 89% | 93% | 96% |
| rodent | 82% | 79% | 85% | 82% | 85% | 82% | 82% | 85% |
| sambar | 100% | 97% | 100% | 100% | 100% | 100% | 100% | 100% |
| small_cat | 77% | 69% | 77% | 77% | 77% | 77% | 77% | 77% |
| squirrel | 77% | 69% | 62% | 65% | 65% | 77% | 69% | 85% |
| wild_boar | 92% | 94% | 92% | 92% | 89% | 86% | 89% | 89% |

## Crop-level F1 on test, classes every model can name (2,139 crops)

| class | ours | speciesnet | ind-ads |
|---|---|---|---|
| background | 0.86 | 0.91 | 0.89 |
| barking_deer | 0.82 | 0.95 | 0.87 |
| bear | 0.90 | 0.88 | 0.55 |
| civet | 0.78 | 0.86 | 0.55 |
| junglefowl | 0.94 | 0.98 | 0.92 |
| macaque | 0.88 | 0.96 | 0.36 |
| mongoose | 0.51 | 0.94 | 0.41 |
| pangolin | 0.84 | 0.86 | 0.51 |
| porcupine | 0.82 | 0.89 | 0.65 |
| sambar | 0.77 | 0.99 | 0.87 |
| small_cat | 0.70 | 0.63 | 0.25 |
| wild_boar | 0.91 | 0.98 | 0.84 |

Macro F1: ours 0.812, speciesnet 0.903, ind-ads 0.641
