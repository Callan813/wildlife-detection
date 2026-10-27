import pandas as pd

from ghats.crops import BACKGROUND, crop_box, label_boxes, split_locations


class TestCropBox:
    def test_pads_and_stays_inside_image(self):
        assert crop_box([0.25, 0.25, 0.75, 0.75], 400, 200, pad=0.1) == (80, 40, 320, 160)
        left, top, right, bottom = crop_box([0.0, 0.0, 1.0, 1.0], 400, 200, pad=0.1)
        assert (left, top, right, bottom) == (0, 0, 400, 200)

    def test_tiny_box_grows_to_min_side(self):
        left, top, right, bottom = crop_box([0.5, 0.5, 0.505, 0.505], 1000, 1000, pad=0.1, min_side=32)
        assert right - left >= 32 and bottom - top >= 32

    def test_never_empty(self):
        left, top, right, bottom = crop_box([1.0, 1.0, 1.0, 1.0], 100, 100, pad=0.0, min_side=0)
        assert right > left and bottom > top


class TestLabelBoxes:
    def test_rules(self):
        boxes = pd.DataFrame(
            {
                "seq_gt": ["empty", "empty", "animal", "animal", "animal"],
                "seq_group": [None, None, "sambar", "sambar", "sambar"],
                "conf": [0.01, 0.05, 0.05, 0.2, 0.9],
            }
        )
        out = label_boxes(boxes, min_conf=0.02, train_min_conf=0.2)
        assert len(out) == 4  # the 0.01 box is below the floor
        assert out["label"].tolist() == [BACKGROUND, "sambar", "sambar", "sambar"]
        assert out["train_ok"].tolist() == [True, False, True, True]


class TestSplitLocations:
    def events(self):
        rows = []
        for i in range(40):
            label = "rare" if i < 4 else ("sambar" if i % 2 else BACKGROUND)
            rows += [{"location": f"L{i}", "label": label}] * (1 + i % 3)
        return pd.DataFrame(rows)

    def test_every_location_in_exactly_one_split(self):
        ev = self.events()
        s = split_locations(ev, 0.15, 0.15, seed=0, tries=20)
        assert set(s) == set(ev["location"]) and set(s.values()) == {"train", "val", "test"}

    def test_rare_label_reaches_val_and_test(self):
        ev = self.events()
        s = split_locations(ev, 0.15, 0.15, seed=0, tries=50)
        rare_splits = {s[loc] for loc in ev.loc[ev["label"] == "rare", "location"]}
        assert {"val", "test"} <= rare_splits

    def test_deterministic(self):
        ev = self.events()
        assert split_locations(ev, seed=3, tries=10) == split_locations(ev, seed=3, tries=10)
