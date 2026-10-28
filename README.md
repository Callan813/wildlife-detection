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
python scripts/04_gallery.py --threshold 0.01

# 5. Filter rules on top of the detector vs. the plain max score (uses saved detections)
python scripts/05_filter_experiments.py

pytest   # unit + end-to-end tests, no network or GPU needed
```

The metadata file is large, and parsing it needs a few GB of RAM. Fetching is a
one-time cost; everything after works offline.

### Results

5,726 frames in 1,200 trigger events (600 animal, 600 empty) from 507 camera
locations. Full numbers in [`reports/phase1/summary.md`](reports/phase1/summary.md).

| Threshold (per trigger event) | Animal events kept | Empty events removed | Review workload removed |
|---|---|---|---|
| 0.01 (the lowest stored) | 96.5% (21 of 600 missed) | 28.0% | 15.8% |
| 0.02 | 94.8% | 40.0% | 22.6% |
| 0.06 | 90.2% | 65.0% | 37.4% |

Worst species group at 0.01: pangolin 87.8%, then rodent 92.7% and macaque 92.9%.
Latency per image (GPU, RTX 3050 Laptop, batches of 16): 28 ms median, 40 ms p95,
including JPEG decode.

![trade-off](reports/phase1/tradeoff.png)

**The short version: out of the box, MegaDetector doesn't filter this forest well.**
The 98% recall target can't be reached at all. 21 animal events have no box above 0.01,
and keeping 95% of animals still leaves 72% of empty events for a person to review.
Things get better if you accept 90% recall, where it removes 65% of empty events.

**Why.** Looking at the highest-scoring "empty" events
([`false_alarms.jpg`](reports/phase1/false_alarms.jpg)) shows the same few culprits over
and over: big leaves right in front of the lens (scoring up to 0.88), dark rocks
and shadows, sun glare, and tree trunks cut off at the frame edge. Dense tropical
undergrowth is full of animal-sized, animal-shaped clutter.

Some of the errors are the labels, not the model. A couple of "empty" events contain
a person's legs or what looks like an animal at night. The missed animal events
([`missed_animals.jpg`](reports/phase1/missed_animals.jpg)) are a mix. At
thumbnail size most look empty, but the pangolin bursts I checked at full size do
have the animal in them: a dim shape the same colour as the leaf litter under IR.
Pangolin is the worst group (5 of the 21 misses).

### What I tried on top of the detector

If leaves cause the false alarms, two cheap rules should help, without any retraining.
Both are in [`src/ghats/filters.py`](src/ghats/filters.py) and compared in
[`reports/phase1/filter_experiments.md`](reports/phase1/filter_experiments.md).

| Rule (at the lowest threshold, 0.01) | Animal events kept | Empty events removed |
|---|---|---|
| Max score (baseline) | 96.5% | 28.0% |
| Burst agreement: 2 frames must see an animal | 94.0% | 35.2% |
| Burst agreement: the same box in 2 frames | 84.7% | 37.8% |
| Camera background: drop boxes where this camera fired in other events (IoU 0.5) | 95.0% | 29.3% |

Neither is better than just raising the threshold: at 94% recall the plain max score
already removes about 40% of empty events, more than burst agreement's 35%. Burst agreement
fails for three reasons: 26% of events are a single frame, animals often show up in
only one frame of a burst, and leaves don't move, so they "agree" with themselves.
Camera background could work on a real deployment, but this sample deliberately
spreads across 507 cameras with a median of 2 events each, which isn't enough
history to learn what a camera's background looks like.

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
  events, or the lowest threshold when even that falls short (as it does here).
- **Per-group recall.** Small and nocturnal animals are where detectors usually
  fail, so recall is reported per species group. Here the weakest was pangolin
  (87.8%), then rodent and macaque.
- **Boxes on the letterbox padding are dropped.** SWG frames are wider than they
  are tall, so the 640×640 input is padded above and below. The model puts low-
  confidence boxes (up to ~0.2) on that padding in almost every image; after
  mapping back they sit wholly outside the frame and clamp to zero-height lines
  on the border. I first missed this, and it quietly ruined the numbers: their
  confidence is noise (event AUC 0.53), but it gave every event, empty or not, a
  score of ~0.08. The old "98.8% recall at 0.06" came from that, not from real
  boxes. I found it while looking at Phase 2 misses (`scripts/10_miss_gallery.py`).
- **The detector runs in eval mode, in batches.** PyTorch-Wildlife 1.3's
  `single_image_detection()` for the MIT model reloads the checkpoint on every call
  and leaves the network in training mode, so BatchNorm normalises each image with
  its own statistics instead of the learned ones. `src/ghats/detector.py` loads the
  model once, calls `eval()`, and runs batches fed by a multi-process DataLoader.
  On the Phase 1 set this raised event-level AUC from 0.870 to 0.903 (both measured
  before the padding fix below; 0.913 after it) and cut
  latency from 826 ms to 28 ms per image (median).
- **MIT-licensed detector by default.** `MegaDetectorV6MIT` avoids the AGPL terms
  of the Ultralytics-based weights; the config can switch models for comparison.

## Phase 2: species classifier on detector crops

**Question:** can a classifier that looks at each detector crop name the species, and
also catch the leaves and rocks that fool the detector?

```bash
python scripts/06_make_crops.py          # detector boxes -> labelled crops, split by camera
python scripts/07_train_classifier.py    # ConvNeXt-Tiny, ~75 s/epoch on an RTX 3050 Laptop
python scripts/09_event_eval.py          # empty-event filtering: detector vs classifier
python scripts/10_miss_gallery.py --threshold 0.504   # animal events the filter throws away
```

**Crops and labels.** Every animal box at 0.02 or above becomes a crop (up to 5 per frame).
Boxes in empty events are labelled "background". Boxes in animal events take the event's
species group, but only boxes at 0.2 or above are used for training, since a weak box
next to an animal is often a leaf.

**Split by camera, not by image.** Whole camera locations go to train, val or test
(70/15/15). Frames from one camera share a background, so a random split would let the
model recognise the place instead of the animal. Test is 106 cameras the model never saw.

**Model.** ConvNeXt-Tiny (ImageNet-22k weights, via timm), 12 epochs, mixed precision,
square-root class weights for the imbalance, label smoothing 0.1, best epoch picked
on val macro F1.

### Results

**Crops** (test, 2,383 crops): macro F1 **0.776**, accuracy 0.833. Best: junglefowl 0.93,
wild boar 0.91, bear 0.90. Worst: squirrel 0.46 and mongoose 0.49, small classes with
20–50 test crops each. Background F1 is 0.84. Details in
[`classifier_summary.md`](reports/phase2/classifier_summary.md).

**Empty-event filtering** (test, 566 events from unseen cameras; each threshold is picked
on val and applied unchanged to test):

| Keep 90% of animal events | Empty events removed |
|---|---|
| Detector confidence (Phase 1 rule) | 54.7% |
| Classifier: P(not background) of the best box | **69.3%** |
| Classifier × detector | 60.0% |

The classifier removes about 15 points more empty events at the same recall. Stricter
targets are out of reach for every method: 25 of 416 animal test events (6%) have no
detector box at all, so nothing downstream can recover them. Full table, including
per-group recall, in [`event_eval.md`](reports/phase2/event_eval.md).

**Where it still fails** ([`missed_events.jpg`](reports/phase2/missed_events.jpg)).
At the 90% threshold, 83 val/test animal events are dropped. 53 of them never had a box
(a detector miss), and 30 had boxes the classifier called background. The detector is
the bigger bottleneck, and it is worst on small animals: rodents, squirrels and martens.

**Things I tried that didn't help** (on the earlier run, before the padding fix):
requiring a second convincing box per event (22% of animal events have only one),
temperature scaling (better calibrated probabilities, same ranking, so the same
filtering), and turning off label smoothing (slightly higher F1, filtering within noise:
150 empty test events give about ±6 points).

## Repository layout

```
configs/phase1.yaml        dataset, sampling, detector, evaluation settings
configs/phase2.yaml        bigger sample for classification, crop and split settings
src/ghats/                 coco_ct (metadata), events, metrics, filters, crops, detector wrapper
scripts/01..05             fetch -> detect -> evaluate -> gallery -> filter experiments
scripts/06..10             crops -> classifier -> baselines -> event eval -> miss gallery (Phase 2)
tests/                     unit tests + synthetic end-to-end pipeline test
reports/phase1, phase2/    generated results (committed)
```

## Data and models

- SWG Camera Traps 2018–2020, CDLA-Permissive-1.0. Citation: SWG (2021),
  *Northern and Central Annamites Camera Traps 2.0*, IUCN SSC Asian Wild Cattle
  Specialist Group's Saola Working Group.
- MegaDetector V6 via [PyTorch-Wildlife](https://github.com/microsoft/CameraTraps).
