# Phase 4: queries judged by eye

The labels say which species is in each event, but not what it's doing, so these were
checked by looking. Top 12 events from the val + test cameras (1,115 events), the same
search the dashboard runs (`scripts/32_search_gallery.py`, sheets in `gallery/`). Each event
was judged on the frame and box the search shows for it. One person, one pass, no second
opinion, so treat the counts as rough.

| Query | How it was read | Right in top 12 | Notes |
|---|---|---|---|
| sambar at night | filters: sambar + night | **12** | every one a sambar in an IR or flash shot |
| bird on the ground | filter: junglefowl ("bird"), then text | **12** | all junglefowl; the box is on the bird in 9 |
| two wild pigs | filters: wild boar + 2 boxes | 12 pigs, **5** with two or more in view | "2 boxes" counts detector boxes >= 0.2, which can be one pig twice |
| macaque climbing a tree | filter: macaque, then text | 12 macaques, **1** climbing | the rest are on the ground |
| wild pig digging in the ground | filter: wild boar, then text | 11 pigs, **~5** nose-down | snout in the leaves, which may be feeding, not digging |
| animal drinking water | filter: animal, then text | **0** drinking | 6 of 12 are at a stream: it found water, not drinking |
| animal close to the camera | filter: animal, then text | **3** | |
| animal running | filter: animal, then text | **~3** | motion blur is the only cue in a still |
| animal in the rain | filter: animal, then text | **0** | hazy, washed-out scenes instead |
| mother with young | filter: animal, then text | **0** | 3 of the 12 are labelled empty |
| a tiger | text only | n/a (no tigers here) | 2 small cats and 4 empty frames. Search never says "nothing like that" |

## What this shows

- Anything the filters can answer is close to exact: species (the Phase 2 model), night or
  day (the capture time), camera, date. Most real questions from a forest department are
  of this kind ("sambar at night on camera 12 in March").
- The embeddings help with the scene: water, haze, a dark frame. They don't understand
  what the animal is doing (drinking, climbing, running) or relations (mother and young).
  BioCLIP 2 was trained to name species, not to describe behaviour, and a camera-trap animal is
  often a small part of a 224-pixel image.
- Text-only search always returns something. A similarity cut-off could say "no good match",
  but it would need labelled queries to set, and there are none for behaviour yet.
- Earlier in the work the boxes shown were often on leaves next to the animal: crops down to
  detector confidence 0.02 were matched against the text. Only crops >= 0.1 are matched now
  (`configs/phase4.yaml`); the species numbers barely move either way, but the sheets look right.
