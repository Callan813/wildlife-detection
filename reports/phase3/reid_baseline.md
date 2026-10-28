# Phase 3: leopard re-identification with off-the-shelf features (no training)

Test: 1,361 left/right flank crops of 107 leopards never used for training, in 781 encounters. Each crop is a query against all other test crops; crops from its own encounter are removed, and only the same flank counts as a match.

Top-k is averaged per leopard (each individual's hit rate, then the mean), so the few leopards photographed 100+ times don't dominate; the per-query top-1 is shown for comparison.

| Model | Top-1 | Top-5 | Top-10 | mAP | Top-1 per query | Val top-1 | Queries | Correct top-1 within 10 image ids |
|---|---|---|---|---|---|---|---|---|
| dinov2_b | 26.7% | 37.3% | 43.4% | 22.2% | 57.0% | 30.0% | 1190 | 16% |
| dinov2_s | 29.6% | 38.4% | 43.1% | 22.7% | 59.0% | 28.9% | 1190 | 15% |
| speciesnet | 36.7% | 47.0% | 50.4% | 25.9% | 62.5% | 27.9% | 1190 | 16% |

All numbers for val and test, per query and per leopard, are in reid_baseline.csv.
