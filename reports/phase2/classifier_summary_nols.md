# Phase 2 results: species classifier on detector crops

Model: `convnext_tiny.fb_in22k`, 12 epochs on 12,131 train-camera crops, label smoothing 0.0.
Test: 2,936 crops (detector conf >= 0.2 or background) from 109 cameras never seen in training.

**Test macro F1 0.781**, accuracy 0.868 (best val macro F1 0.824).

| Class | Test crops | Precision | Recall | F1 |
|---|---|---|---|---|
| squirrel | 23 | 0.52 | 0.48 | 0.50 |
| mongoose | 46 | 0.55 | 0.61 | 0.58 |
| small_cat | 48 | 1.00 | 0.46 | 0.63 |
| marten | 41 | 0.76 | 0.63 | 0.69 |
| sambar | 203 | 0.73 | 0.78 | 0.75 |
| rodent | 32 | 0.71 | 0.84 | 0.77 |
| pangolin | 20 | 0.71 | 0.85 | 0.77 |
| civet | 110 | 0.86 | 0.72 | 0.78 |
| porcupine | 91 | 0.75 | 0.86 | 0.80 |
| barking_deer | 275 | 0.84 | 0.77 | 0.80 |
| mouse_deer | 148 | 0.83 | 0.88 | 0.85 |
| macaque | 127 | 0.88 | 0.87 | 0.88 |
| wild_boar | 533 | 0.91 | 0.89 | 0.90 |
| junglefowl | 97 | 0.91 | 0.92 | 0.91 |
| bear | 125 | 0.89 | 0.98 | 0.93 |
| background | 1017 | 0.93 | 0.95 | 0.94 |

Classes with fewer than ~50 test crops have noisy F1: one or two crops move it by several points.
