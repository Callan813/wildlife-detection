# Phase 3: DINOv2-S fine-tuned with ArcFace + LeopardID102 leopards

Trained on 3,151 crops of 310 leopards (534 flanks), 20 epochs, input 224x448, seed 1; best epoch 11 by val top-1 per leopard. Test: 1,361 crops of 107 leopards never seen in training, scored as in reid_baseline.md (no same-encounter matches, same flank only, top-k averaged per leopard).

| Model | Top-1 | Top-5 | Top-10 | mAP | Top-1 per query |
|---|---|---|---|---|---|
| DINOv2-S, untrained (same input size) | 28.4% | 39.7% | 44.3% | 22.8% | 59.2% |
| DINOv2-S + ArcFace | 53.3% | 66.1% | 72.3% | 44.5% | 74.0% |

Per-epoch val scores are in outputs/phase3/reid_lp102_s1/history.csv.
