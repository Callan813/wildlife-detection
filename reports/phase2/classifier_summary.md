# Phase 2 results: species classifier on detector crops

Model: `convnext_tiny.fb_in22k`, fine-tuned 12 epochs on 12,131 crops from train cameras.
Test: 2,936 crops (detector conf >= 0.2 or background) from 109 cameras never seen in training.

**Test macro F1 0.767**, accuracy 0.859 (best val macro F1 0.819).

| Class | Test crops | Precision | Recall | F1 |
|---|---|---|---|---|
| squirrel | 23 | 0.48 | 0.52 | 0.50 |
| mongoose | 46 | 0.46 | 0.72 | 0.56 |
| rodent | 32 | 0.52 | 0.81 | 0.63 |
| pangolin | 20 | 0.59 | 0.80 | 0.68 |
| marten | 41 | 0.72 | 0.68 | 0.70 |
| small_cat | 48 | 1.00 | 0.56 | 0.72 |
| sambar | 203 | 0.77 | 0.74 | 0.75 |
| civet | 110 | 0.77 | 0.75 | 0.76 |
| barking_deer | 275 | 0.84 | 0.74 | 0.79 |
| porcupine | 91 | 0.83 | 0.80 | 0.82 |
| bear | 125 | 0.77 | 0.98 | 0.86 |
| macaque | 127 | 0.88 | 0.88 | 0.88 |
| mouse_deer | 148 | 0.86 | 0.91 | 0.89 |
| wild_boar | 533 | 0.92 | 0.86 | 0.89 |
| junglefowl | 97 | 0.90 | 0.93 | 0.91 |
| background | 1017 | 0.93 | 0.94 | 0.93 |

Classes with fewer than ~50 test crops have noisy F1: one or two crops move it by several points.
