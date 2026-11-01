# Phase 4: search evaluation

566 trigger events from the test cameras (val cameras: 549 events).
AP = average precision over all test events; P@10 = share of the top 10 that are right.

## species

| Method | test AP | test P@10 | val AP |
|---|---|---|---|
| probe | 0.872 | 0.99 | 0.854 |
| probe + 0.0 x bioclip-2 / taxon / crops | 0.872 | 0.99 | 0.854 |
| bioclip-2 / taxon / crops | 0.540 | 0.70 | 0.538 |
| bioclip-2 / taxon / both | 0.522 | 0.68 | 0.537 |
| bioclip-2 / both / crops | 0.513 | 0.69 | 0.491 |
| bioclip-2 / both / both | 0.482 | 0.65 | 0.460 |
| bioclip-2 / common / crops | 0.433 | 0.62 | 0.402 |
| bioclip-2 / taxon / frames | 0.356 | 0.57 | 0.394 |
| clip-b16 / common / crops | 0.362 | 0.53 | 0.321 |
| clip-b16 / common / both | 0.340 | 0.51 | 0.296 |
| chance (random order) | 0.049 | 0.05 | 0.051 |

## empty

| Method | test AP | test P@10 | val AP |
|---|---|---|---|
| probe (1 - P animal) | 0.701 | 0.80 | 0.658 |
| bioclip-2 / frames | 0.333 | 0.20 | 0.346 |
| bioclip-2 / both | 0.333 | 0.20 | 0.340 |
| clip-b16 / both | 0.303 | 0.30 | 0.275 |
| clip-b16 / frames | 0.301 | 0.30 | 0.274 |
| chance (random order) | 0.265 | 0.27 | 0.237 |

## night

| Method | test AP | test P@10 | val AP |
|---|---|---|---|
| clip-b16 / frames | 0.888 | 1.00 | 0.874 |
| bioclip-2 / frames | 0.738 | 0.90 | 0.809 |
| colour score (grey = night) | 0.653 | 1.00 | 0.704 |
| chance (random order) | 0.417 | 0.42 | 0.388 |

## species filter

Keep an event when the probe gives the species >= 0.8 (best mean F1 on val).

| Split | precision | recall | F1 |
|---|---|---|---|
| val | 0.965 | 0.790 | 0.865 |
| test | 0.955 | 0.793 | 0.860 |

## which crops to match

Species queries (bioclip-2, scientific-name prompts, frames + crops) when only crops
with detector confidence >= the floor take part. Best on val: 0.0.

| Floor | val AP | test AP |
|---|---|---|
| 0.0 | 0.550 | 0.519 |
| 0.05 | 0.546 | 0.521 |
| 0.1 | 0.537 | 0.522 |
| 0.2 | 0.529 | 0.514 |
| 0.3 | 0.524 | 0.519 |
| 0.5 | 0.514 | 0.500 |

## species AP by group (test)

| Group | probe | bioclip-2 / taxon / crops | probe + 0.0 x bioclip-2 / taxon / crops | clip-b16 / common / both |
|---|---|---|---|---|
| wild_boar | 0.845 | 0.729 | 0.845 | 0.472 |
| sambar | 0.993 | 0.780 | 0.993 | 0.514 |
| barking_deer | 0.940 | 0.340 | 0.940 | 0.326 |
| mouse_deer | 0.973 | 0.579 | 0.973 | 0.404 |
| civet | 0.865 | 0.491 | 0.865 | 0.204 |
| mongoose | 0.948 | 0.492 | 0.948 | 0.149 |
| small_cat | 0.775 | 0.537 | 0.775 | 0.169 |
| marten | 0.760 | 0.572 | 0.760 | 0.135 |
| bear | 0.967 | 0.857 | 0.967 | 0.566 |
| pangolin | 0.902 | 0.518 | 0.902 | 0.300 |
| porcupine | 0.919 | 0.464 | 0.919 | 0.366 |
| macaque | 0.832 | 0.642 | 0.832 | 0.579 |
| squirrel | 0.608 | 0.043 | 0.608 | 0.057 |
| junglefowl | 0.962 | 0.872 | 0.962 | 0.726 |
| rodent | 0.791 | 0.185 | 0.791 | 0.128 |
