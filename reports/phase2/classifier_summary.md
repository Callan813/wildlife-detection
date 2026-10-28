# Phase 2 results: species classifier on detector crops

Model: `convnext_tiny.fb_in22k`, 12 epochs on 9,907 train-camera crops, label smoothing 0.1.
Test: 2,383 crops (detector conf >= 0.2 or background) from 106 cameras never seen in training.

**Test macro F1 0.776**, accuracy 0.833 (best val macro F1 0.809).

| Class | Test crops | Precision | Recall | F1 |
|---|---|---|---|---|
| squirrel | 23 | 0.41 | 0.52 | 0.46 |
| mongoose | 46 | 0.41 | 0.61 | 0.49 |
| rodent | 32 | 0.62 | 0.81 | 0.70 |
| small_cat | 48 | 1.00 | 0.54 | 0.70 |
| marten | 41 | 0.71 | 0.73 | 0.72 |
| sambar | 203 | 0.69 | 0.86 | 0.77 |
| civet | 110 | 0.82 | 0.74 | 0.78 |
| porcupine | 91 | 0.76 | 0.86 | 0.80 |
| barking_deer | 275 | 0.88 | 0.75 | 0.81 |
| pangolin | 20 | 0.89 | 0.80 | 0.84 |
| background | 464 | 0.85 | 0.84 | 0.84 |
| mouse_deer | 148 | 0.88 | 0.86 | 0.87 |
| macaque | 127 | 0.87 | 0.88 | 0.88 |
| bear | 125 | 0.83 | 0.98 | 0.90 |
| wild_boar | 533 | 0.95 | 0.88 | 0.91 |
| junglefowl | 97 | 0.94 | 0.93 | 0.93 |

Classes with fewer than ~50 test crops have noisy F1: one or two crops move it by several points.
