# Ghats Wildlife AI

Camera-trap intelligence for the Western Ghats: find the animals in thousands of
trigger images, identify the species, recognise individual leopards, search the
archive in plain English, and run it all on a low-power device in the forest.

Forest departments and researchers in Goa (Mhadei, Bhagwan Mahavir, Cotigao,
Netravali) collect far more camera-trap images than anyone can review by hand,
and most frames are empty, triggered by wind, sun or moving leaves. This project
builds the pipeline that does the first pass automatically and sends people only
what needs a human decision.

## Roadmap

| Phase | What | Status |
|---|---|---|
| 1 | Animal detection and empty-frame filtering (MegaDetector V6) | **Done** (results below) |
| 2 | Species classification for Ghats groups, location-split evaluation, SpeciesNet baseline | In progress |
| 3 | Leopard re-identification (metric learning + travel-feasibility prior) | Planned |
| 4 | Natural-language search over detections (BioCLIP/CLIP + vector DB) | Planned |
| 5 | Reviewer dashboard with an active-learning loop | Planned |
| 6 | Edge deployment: ONNX/INT8 on Raspberry Pi 5 or Jetson, cascade design, benchmarks | Planned |

## Phase 1: detection and empty-frame filtering

**Question:** at what confidence threshold can MegaDetector remove most empty
frames while keeping (almost) every animal, on tropical Asian forest data it was
mostly not trained on?

**Data:** a balanced sample from [SWG Camera Traps 2018–2020](https://lila.science/datasets/swg-camera-traps)
(Vietnam and Laos, via LILA BC). It is the closest public stand-in for Western
Ghats conditions: dense forest, IR night images, and many of the same genera.
Species are mapped to Ghats groups in [`configs/phase1.yaml`](configs/phase1.yaml)
(e.g. red muntjac → barking deer, chevrotain → mouse deer).

### Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

# 1. Metadata + ~1,200 sequences (~5,500 images, a few GB). Resumable.
python scripts/01_fetch_subset.py

# 2. Detector. Try --limit 50 first. CPU works; a GPU (or free Colab T4) is much faster.
python scripts/02_run_detector.py

# 3. Metrics, trade-off plot and summary
python scripts/03_evaluate.py

# 4. Picture grids of misses and false alarms (use the threshold from summary.md)
python scripts/04_gallery.py --threshold 0.06

# 5. Filter rules on top of the detector vs. the plain max score (uses saved detections)
python scripts/05_filter_experiments.py

pytest   # unit + end-to-end tests, no network or GPU needed
```

The metadata file is large, and parsing it needs a few GB of RAM. Fetching is a
one-time cost; everything after works offline.

### Results

5,726 frames in 1,200 trigger events (600 animal, 600 empty) from 507 camera
locations. Full numbers in [`reports/phase1/summary.md`](reports/phase1/summary.md).

| Metric (per trigger event) | Value |
|---|---|
| Threshold | 0.06 |
| Animal events kept | 98.8% (7 of 600 missed) |
| Empty events removed | 6.2% |
| Review workload removed | 3.7% |
| Worst species group (recall) | marten 96.9%, then pangolin, small cat, civet and squirrel at 97.6% |
| Latency per image (GPU, RTX 3050 Laptop, batches of 16) | 28 ms median, 40 ms p95, including JPEG decode |

![trade-off](reports/phase1/tradeoff.png)

**The short version: out of the box, MegaDetector doesn't filter this forest well.**
Keeping 98% of animals means keeping almost every empty event too. Loosen the target
and it gets better: at 95% recall it removes 30% of empty events, and at 90% recall
it removes 55%.

**Why.** Looking at the highest-scoring "empty" events
([`false_alarms.jpg`](reports/phase1/false_alarms.jpg)) shows the same few culprits over
and over: big leaves right in front of the lens (scoring up to 0.88), dark rocks
and shadows, sun glare, and tree trunks cut off at the frame edge. Dense tropical
undergrowth is full of animal-sized, animal-shaped clutter.

Some of the errors are the labels, not the model. A couple of "empty" events contain
a person's legs or what looks like an animal at night, and most of the missed animal
events ([`missed_animals.jpg`](reports/phase1/missed_animals.jpg)) show no animal in the
frame at all, because LILA labels the whole burst. The real misses are hard cases: a
porcupine lit up by IR glare, a small cat in dark ferns.

### What I tried on top of the detector

If leaves cause the false alarms, two cheap rules should help, without any retraining.
Both are in [`src/ghats/filters.py`](src/ghats/filters.py) and compared in
[`reports/phase1/filter_experiments.md`](reports/phase1/filter_experiments.md).

| Rule (at 98% recall target) | Animal events kept | Empty events removed |
|---|---|---|
| Max score (baseline) | 98.8% | 6.2% |
| Burst agreement: 2 frames must see an animal | 98.0% | 8.7% |
| Burst agreement: the same box in 2 frames | 85.8% (misses target) | 27.3% |
| Camera background: drop boxes where this camera fired in other events | 98.8% | 6.2% |

Neither helps much. Burst agreement removes a few more empty events (8.7% instead of
6.2%), but it is a small gain for three reasons: 26% of events are a single
frame, animals often show up in only one frame of a burst, and leaves don't move, so
they "agree" with themselves. Camera background could work on a real deployment, but
this sample deliberately spreads across 507 cameras with a median of 2 events each,
which isn't enough history to learn what a camera's background looks like.

**What this means for Phase 2.** No rule on the detector's score will separate leaves
from animals here, because the scores overlap. The fix has to look at the crop itself.
Phase 2 trains a classifier on detector crops anyway, so it gets an extra
"vegetation / no animal" class trained on these exact false alarms.

### Design decisions

- **Sampling and evaluation by trigger event, not frame.** A camera fires a burst
  of frames per trigger, and LILA often labels the whole sequence. A frame where
  the animal has already left still says "wild pig", so frame-level misses are
  partly label noise. Event-level numbers are the honest headline.
- **Per-camera caps.** No camera contributes more than a few sequences, so the
  result reflects many backgrounds rather than one busy waterhole.
- **Thresholds are swept, not guessed.** The detector stores everything above
  0.01; the operating point is the highest threshold that keeps 98% of animal
  events.
- **Per-group recall.** Small and nocturnal animals are where detectors usually
  fail, so recall is reported per species group. Here the weakest was marten,
  and no group fell below 96.9%.
- **The detector runs in eval mode, in batches.** PyTorch-Wildlife 1.3's
  `single_image_detection()` for the MIT model reloads the checkpoint on every call
  and leaves the network in training mode, so BatchNorm normalises each image with
  its own statistics instead of the learned ones. `src/ghats/detector.py` loads the
  model once, calls `eval()`, and runs batches fed by a multi-process DataLoader.
  On the Phase 1 set this raised event-level AUC from 0.870 to 0.903 and cut
  latency from 826 ms to 28 ms per image (median).
- **MIT-licensed detector by default.** `MegaDetectorV6MIT` avoids the AGPL terms
  of the Ultralytics-based weights; the config can switch models for comparison.

## Repository layout

```
configs/phase1.yaml        dataset, sampling, detector, evaluation settings
configs/phase2.yaml        bigger sample for classification, crop and split settings
src/ghats/                 coco_ct (metadata), events, metrics, filters, crops, detector wrapper
scripts/01..05             fetch -> detect -> evaluate -> gallery -> filter experiments
scripts/06                 detector boxes -> labelled crops, split by camera (Phase 2)
tests/                     unit tests + synthetic end-to-end pipeline test
reports/phase1/            generated results (committed)
```

## Data and models

- SWG Camera Traps 2018–2020, CDLA-Permissive-1.0. Citation: SWG (2021),
  *Northern and Central Annamites Camera Traps 2.0*, IUCN SSC Asian Wild Cattle
  Specialist Group's Saola Working Group.
- MegaDetector V6 via [PyTorch-Wildlife](https://github.com/microsoft/CameraTraps).
