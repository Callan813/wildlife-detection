# Phase 2: empty-event filtering with a crop classifier

Test: 566 events (416 animal, 150 empty) from 114 cameras not used in training.
Each threshold is the highest that keeps the target share of animal events on val, applied unchanged to test.

29 of 416 animal test events have no detector box at or above the crop floor, so no method can keep more than 93.0% of them. Targets above what val allows are marked out of reach.

Cells: share of empty test events removed (test recall of animal events in brackets).
Thresholds, workload and missed counts are in event_eval.csv.

| method, events scored by | 98% target | 95% target | 90% target |
|---|---|---|---|
| detector, best box | out of reach | out of reach | 55.3% (91.1%) |
| detector, 2nd box | out of reach | out of reach | 52.7% (90.6%) |
| ours, best box | out of reach | out of reach | 70.7% (89.9%) |
| ours, 2nd box | out of reach | out of reach | 58.7% (90.4%) |
| ours x detector, best box | out of reach | out of reach | 63.3% (91.1%) |
| ours x detector, 2nd box | out of reach | out of reach | 57.3% (89.4%) |
| probe, best box | out of reach | out of reach | 74.0% (91.3%) |
| probe, 2nd box | out of reach | out of reach | 54.0% (92.1%) |
| probe x detector, best box | out of reach | out of reach | 67.3% (91.6%) |
| probe x detector, 2nd box | out of reach | out of reach | 55.3% (91.3%) |
| speciesnet, best box | out of reach | out of reach | 80.7% (90.9%) |
| speciesnet, 2nd box | out of reach | out of reach | 64.7% (91.3%) |
| speciesnet x detector, best box | out of reach | out of reach | 73.3% (91.3%) |
| speciesnet x detector, 2nd box | out of reach | out of reach | 65.3% (90.6%) |
| ind-ads, best box | out of reach | out of reach | 59.3% (91.1%) |
| ind-ads, 2nd box | out of reach | out of reach | 50.7% (91.8%) |
| ind-ads x detector, best box | out of reach | out of reach | 62.0% (91.8%) |
| ind-ads x detector, 2nd box | out of reach | out of reach | 52.7% (90.6%) |

## Test recall by species group at the 90% threshold

| group | detector, best box | detector, 2nd box | ours, best box | ours, 2nd box | probe, best box | probe, 2nd box | speciesnet, best box | speciesnet, 2nd box | ind-ads, best box | ind-ads, 2nd box |
|---|---|---|---|---|---|---|---|---|---|---|
| barking_deer | 100% | 97% | 94% | 97% | 97% | 100% | 97% | 100% | 100% | 100% |
| bear | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| civet | 95% | 92% | 95% | 95% | 95% | 95% | 95% | 95% | 95% | 95% |
| junglefowl | 96% | 92% | 96% | 96% | 96% | 96% | 96% | 96% | 96% | 96% |
| macaque | 87% | 87% | 90% | 90% | 90% | 90% | 90% | 90% | 90% | 90% |
| marten | 78% | 78% | 74% | 74% | 78% | 78% | 74% | 78% | 78% | 74% |
| mongoose | 96% | 100% | 93% | 96% | 96% | 100% | 96% | 96% | 100% | 93% |
| mouse_deer | 98% | 100% | 98% | 100% | 98% | 100% | 100% | 95% | 98% | 100% |
| pangolin | 90% | 90% | 90% | 90% | 90% | 90% | 90% | 90% | 90% | 90% |
| porcupine | 96% | 96% | 93% | 93% | 96% | 96% | 96% | 96% | 93% | 96% |
| rodent | 82% | 82% | 85% | 82% | 85% | 82% | 85% | 82% | 85% | 85% |
| sambar | 100% | 97% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| small_cat | 77% | 69% | 77% | 77% | 77% | 77% | 77% | 77% | 77% | 77% |
| squirrel | 73% | 73% | 62% | 65% | 69% | 77% | 65% | 77% | 65% | 81% |
| wild_boar | 89% | 92% | 92% | 89% | 92% | 89% | 89% | 89% | 89% | 89% |

## Crop-level F1 on test, classes every model can name (2,094 crops)

| class | ours | probe | speciesnet | ind-ads |
|---|---|---|---|---|
| background | 0.85 | 0.92 | 0.90 | 0.88 |
| barking_deer | 0.82 | 0.93 | 0.95 | 0.87 |
| bear | 0.90 | 0.95 | 0.88 | 0.55 |
| civet | 0.78 | 0.87 | 0.86 | 0.55 |
| junglefowl | 0.94 | 0.98 | 0.98 | 0.92 |
| macaque | 0.88 | 0.94 | 0.96 | 0.36 |
| mongoose | 0.51 | 0.93 | 0.94 | 0.41 |
| pangolin | 0.84 | 0.90 | 0.86 | 0.51 |
| porcupine | 0.82 | 0.93 | 0.89 | 0.65 |
| sambar | 0.77 | 0.95 | 0.99 | 0.88 |
| small_cat | 0.70 | 0.77 | 0.63 | 0.25 |
| wild_boar | 0.91 | 0.96 | 0.98 | 0.84 |

Macro F1: ours 0.810, probe 0.920, speciesnet 0.902, ind-ads 0.641
