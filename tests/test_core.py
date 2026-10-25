import math

import pandas as pd
import pytest

from ghats.coco_ct import build_image_table
from ghats.events import assign_events
from ghats.metrics import pick_operating_point, recall_by_group, threshold_sweep

GROUPS = {"sambar": "sambar", "eurasian_wild_pig": "wild_boar"}
EXCLUDE = ["blurred", "human"]


def meta(images, anns):
    cats = [
        {"id": 0, "name": "empty"},
        {"id": 1, "name": "sambar"},
        {"id": 2, "name": "eurasian_wild_pig"},
        {"id": 3, "name": "blurred"},
        {"id": 4, "name": "ferret_badger"},
    ]
    return {"images": images, "annotations": anns, "categories": cats}


def img(i, loc="L1", dt="2019-06-01 10:00:00", seq=None):
    return {"id": i, "file_name": f"{loc}/{i}.jpg", "location": loc, "datetime": dt, "seq_id": seq}


def ann(i, cat):
    return {"id": f"a{i}-{cat}", "image_id": i, "category_id": cat}


class TestImageTable:
    def test_labels_map_to_ground_truth(self):
        m = meta(
            [img("a"), img("b"), img("c"), img("d"), img("e"), img("f")],
            [ann("a", 1), ann("b", 0), ann("c", 3), ann("d", 4), ann("e", 1), ann("e", 2)],
        )
        df = build_image_table(m, EXCLUDE, species_groups=GROUPS).set_index("image_id")
        assert df.loc["a", "gt"] == "animal" and df.loc["a", "group"] == "sambar"
        assert df.loc["b", "gt"] == "empty"
        assert df.loc["c", "gt"] == "excluded"  # blurred
        assert df.loc["d", "gt"] == "excluded"  # species outside the Ghats groups
        assert df.loc["e", "gt"] == "excluded"  # two groups in one frame
        assert df.loc["f", "gt"] == "excluded"  # no annotation at all

    def test_private_and_corrupt_images_are_excluded(self):
        m = meta([img("a"), img("b"), img("c")], [ann("a", 1), ann("b", 1), ann("c", 1)])
        m["images"][0]["file_name"] = "private/lao/loc_0000/a.jpg"
        m["images"][1]["corrupt"] = True
        df = build_image_table(m, EXCLUDE, species_groups=GROUPS).set_index("image_id")
        assert df.loc["a", "gt"] == "excluded" and df.loc["b", "gt"] == "excluded"
        assert df.loc["c", "gt"] == "animal"

    def test_without_groups_any_animal_counts(self):
        df = build_image_table(meta([img("d")], [ann("d", 4)]), EXCLUDE)
        assert df.loc[0, "gt"] == "animal"

    def test_label_case_is_ignored(self):
        m = meta([img("a")], [ann("a", 1)])
        m["categories"][1]["name"] = "Sambar"
        df = build_image_table(m, EXCLUDE, species_groups=GROUPS)
        assert df.loc[0, "group"] == "sambar"


class TestEvents:
    def test_seq_id_wins(self):
        df = pd.DataFrame(
            {
                "location": ["L1", "L1"],
                "datetime": pd.to_datetime(["2019-01-01 00:00", "2019-03-01 00:00"]),
                "seq_id": ["s1", "s1"],
            }
        )
        ev = assign_events(df)
        assert ev.nunique() == 1 and ev.iloc[0] == "seq:s1"

    def test_time_gap_splits_events_per_location(self):
        df = pd.DataFrame(
            {
                "location": ["L1", "L1", "L1", "L2"],
                "datetime": pd.to_datetime(
                    ["2019-01-01 10:00:00", "2019-01-01 10:00:30", "2019-01-01 10:10:00", "2019-01-01 10:00:10"]
                ),
                "seq_id": [None, None, None, None],
            }
        )
        ev = assign_events(df, max_gap_seconds=120)
        assert ev.iloc[0] == ev.iloc[1]  # 30 s apart, same burst
        assert ev.iloc[2] != ev.iloc[1]  # 9.5 min later, new trigger
        assert ev.iloc[3] not in set(ev.iloc[:3])  # other camera

    def test_missing_timestamp_is_its_own_event(self):
        df = pd.DataFrame(
            {"location": ["L1", "L1"], "datetime": pd.to_datetime(["2019-01-01 10:00", None]), "seq_id": [None, None]}
        )
        assert assign_events(df).nunique() == 2

    def test_unsorted_input_keeps_index_alignment(self):
        df = pd.DataFrame(
            {
                "location": ["L1", "L1", "L1"],
                "datetime": pd.to_datetime(["2019-01-01 12:00", "2019-01-01 10:00", "2019-01-01 10:01"]),
                "seq_id": [None, None, None],
            },
            index=[10, 20, 30],
        )
        ev = assign_events(df)
        assert ev[20] == ev[30] and ev[10] != ev[20]


class TestMetrics:
    def test_sweep_counts(self):
        s = threshold_sweep([0.9, 0.6, 0.1, 0.05], [True, False, True, False], [0.5])
        row = s.iloc[0]
        assert (row.tp, row.fp, row.fn, row.tn) == (1, 1, 1, 1)
        assert row.recall == 0.5 and row.precision == 0.5
        assert row.empty_removed == 0.5 and row.review_reduction == 0.5

    def test_threshold_is_inclusive(self):
        assert threshold_sweep([0.2], [True], [0.2]).iloc[0].tp == 1

    def test_no_positives_gives_nan_recall(self):
        assert math.isnan(threshold_sweep([0.1], [False], [0.5]).iloc[0].recall)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            threshold_sweep([0.1, 0.2], [True], [0.5])

    def test_operating_point_is_highest_threshold_meeting_target(self):
        sweep = threshold_sweep([0.9, 0.8, 0.3, 0.1], [True, True, True, False], [0.1, 0.2, 0.5])
        assert pick_operating_point(sweep, 1.0).threshold == 0.2
        # Unreachable target falls back to the best-recall row
        assert pick_operating_point(sweep[sweep.threshold == 0.5], 1.0).threshold == 0.5

    def test_recall_by_group_is_worst_first(self):
        units = pd.DataFrame(
            {
                "gt": ["animal"] * 4 + ["empty"],
                "group": ["rodent", "rodent", "sambar", "sambar", None],
                "score": [0.1, 0.6, 0.9, 0.8, 0.95],
            }
        )
        out = recall_by_group(units, 0.5)
        assert list(out.group) == ["rodent", "sambar"]
        assert list(out.recall) == [0.5, 1.0]
