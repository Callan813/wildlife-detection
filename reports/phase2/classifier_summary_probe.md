# Phase 2: frozen SpeciesNet features + logistic regression

Head trained on 9,715 train-camera crops, C=0.01 picked on val (val macro F1 0.925).
Test: 2,338 crops from 106 cameras never seen in training.

**Test macro F1 0.886**, accuracy 0.923.

| Class | Test crops | Precision | Recall | F1 |
|---|---|---|---|---|
| squirrel | 23 | 0.67 | 0.61 | 0.64 |
| small_cat | 48 | 1.00 | 0.62 | 0.77 |
| marten | 41 | 1.00 | 0.66 | 0.79 |
| rodent | 32 | 0.81 | 0.91 | 0.85 |
| civet | 110 | 0.92 | 0.82 | 0.87 |
| background | 419 | 0.85 | 0.95 | 0.90 |
| pangolin | 20 | 0.86 | 0.95 | 0.90 |
| mongoose | 46 | 0.93 | 0.89 | 0.91 |
| macaque | 127 | 0.91 | 0.94 | 0.92 |
| mouse_deer | 148 | 0.89 | 0.96 | 0.93 |
| barking_deer | 275 | 0.96 | 0.90 | 0.93 |
| porcupine | 91 | 0.98 | 0.89 | 0.93 |
| sambar | 203 | 0.91 | 0.99 | 0.95 |
| bear | 125 | 0.91 | 0.98 | 0.95 |
| wild_boar | 533 | 0.98 | 0.95 | 0.96 |
| junglefowl | 97 | 1.00 | 0.97 | 0.98 |
