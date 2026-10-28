# Phase 3: DINOv2-S fine-tuned with ArcFace

Trained on 2,450 crops of 209 leopards (355 flanks), 20 epochs, input 224x448; best epoch 13 by val top-1 per leopard. Test: 1,361 crops of 107 leopards never seen in training, scored as in reid_baseline.md (no same-encounter matches, same flank only, top-k averaged per leopard).

| Model | Top-1 | Top-5 | Top-10 | mAP | Top-1 per query |
|---|---|---|---|---|---|
| DINOv2-S, untrained (same input size) | 28.4% | 39.7% | 44.3% | 22.8% | 59.2% |
| DINOv2-S + ArcFace | 40.2% | 57.1% | 64.5% | 35.4% | 65.7% |

Per-epoch val scores are in outputs/phase3/reid/history.csv.
