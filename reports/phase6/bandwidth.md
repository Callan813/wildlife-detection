# Phase 6: data sent by a camera, with and without on-camera filtering

Phase 2 test set: 566 trigger events (416 animal, 150 empty), 2,886 frames (mean 1.54 MB), from 114 cameras not used in training.
Thresholds keep 90% of animal events on val cameras (reports/phase2/event_eval.csv).

| Policy | Events sent | Animal events kept | Empty events sent | MB sent | Share of every frame | KB per animal event |
|---|---|---|---|---|---|---|
| every frame | 566 | 100.0% | 100.0% | 4,442.2 | 100.00% | 10,678.4 |
| frames of events the detector keeps (conf >= 0.036) | 446 | 91.1% | 44.7% | 3,838.1 | 86.40% | 10,127.0 |
| frames of events the cascade keeps (P(animal) >= 0.273) | 419 | 91.3% | 26.0% | 3,648.5 | 82.13% | 9,601.2 |
| one crop per kept event + metadata | 419 | 91.3% | 26.0% | 10.8 | 0.24% | 28.5 |
| metadata only | 419 | 91.3% | 26.0% | 0.1 | 0.00% | 0.3 |

## MB per 1,000 triggers, by share of empty triggers

Projected from the per-event rates above (26% empty is this test set).

| Policy | 26% empty | 50% empty | 80% empty | 95% empty |
|---|---|---|---|---|
| every frame | 7,864 | 7,121 | 6,193 | 5,729 |
| frames, detector filter | 6,809 | 5,455 | 3,762 | 2,916 |
| frames, cascade filter | 6,481 | 4,799 | 2,697 | 1,646 |
| crop + metadata, cascade | 19 | 14 | 8 | 6 |

Events with no box at all are never sent; that's most of the missed animal events (see Phase 2).
Metadata is counted as 300 bytes per event (an events.jsonl line from scripts/26_live.py).
