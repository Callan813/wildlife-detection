# Phase 2: empty-event filtering with a crop classifier

Test: 566 events (416 animal, 150 empty) from 114 cameras not used in training.
Each threshold is the highest that keeps the target share of animal events on val, applied unchanged to test.

| Val target | Method | Threshold | Test recall | Empty events removed | Review workload removed | Animals missed |
|---|---|---|---|---|---|---|
| 98% | detector | 0.068 | 97.6% | 13.3% | 5.3% | 10 |
| 98% | ours | 0.489 | 97.8% | 18.7% | 6.5% | 9 |
| 98% | ours x detector | 0.031 | 98.1% | 12.7% | 4.8% | 8 |
| 95% | detector | 0.080 | 94.0% | 29.3% | 12.2% | 25 |
| 95% | ours | 0.572 | 95.2% | 43.3% | 15.0% | 20 |
| 95% | ours x detector | 0.039 | 95.7% | 26.7% | 10.2% | 18 |
| 90% | detector | 0.098 | 90.4% | 54.7% | 21.6% | 40 |
| 90% | ours | 0.723 | 89.7% | 75.3% | 27.6% | 43 |
| 90% | ours x detector | 0.058 | 90.4% | 65.3% | 24.4% | 40 |

## Test recall by species group at the 98% threshold

| group | detector | ours | ours x detector |
|---|---|---|---|
| barking_deer | 100% | 97% | 97% |
| bear | 100% | 100% | 100% |
| civet | 100% | 100% | 100% |
| junglefowl | 96% | 100% | 96% |
| macaque | 97% | 97% | 97% |
| marten | 93% | 96% | 96% |
| mongoose | 96% | 100% | 96% |
| mouse_deer | 100% | 100% | 100% |
| pangolin | 100% | 90% | 100% |
| porcupine | 100% | 96% | 100% |
| rodent | 91% | 91% | 94% |
| sambar | 100% | 100% | 100% |
| small_cat | 92% | 100% | 100% |
| squirrel | 100% | 96% | 100% |
| wild_boar | 97% | 100% | 97% |

## Crop-level F1 on test, classes every model can name (2,936 crops)

| class | ours |
|---|---|
| background | 0.93 |
| barking_deer | 0.79 |
| bear | 0.86 |
| civet | 0.76 |
| junglefowl | 0.91 |
| macaque | 0.88 |
| marten | 0.70 |
| mongoose | 0.56 |
| mouse_deer | 0.89 |
| pangolin | 0.68 |
| porcupine | 0.82 |
| rodent | 0.63 |
| sambar | 0.75 |
| small_cat | 0.72 |
| squirrel | 0.50 |
| wild_boar | 0.89 |

Macro F1: ours 0.767
