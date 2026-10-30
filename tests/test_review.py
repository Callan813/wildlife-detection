import numpy as np
import pandas as pd

from ghats.reid import l2_normalise
from ghats.review import audit_ranks, shortlists, species_uncertainty, split_catalogue


def test_audit_ranks_pick_a_random_share_in_random_order():
    ranks = audit_ranks(100, 0.2, np.random.default_rng(0))
    picked = [r for r in ranks if r is not None]
    assert sorted(picked) == list(range(1, 21))
    assert ranks != audit_ranks(100, 0.2, np.random.default_rng(1))


def test_species_uncertainty_margin():
    top, margin = species_uncertainty(np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]]))
    assert top.tolist() == [0, 0]
    assert np.allclose(margin, [0.5, 0.05])


def test_catalogue_takes_first_encounter_of_known_leopards_only():
    crops = pd.DataFrame(
        {
            "name": ["A", "A", "A", "B", "B"],
            "encounter": ["e2", "e2", "e1", "e3", "e4"],
            "image_id": [5, 6, 1, 3, 9],
        }
    )
    in_cat = split_catalogue(crops, known_frac=1.0)
    assert in_cat.tolist() == [False, False, True, True, False]  # A's first is e1 (image 1)
    assert not split_catalogue(crops, known_frac=0.0).any()


def test_shortlist_ranks_leopards_by_best_crop_and_keeps_flank():
    cat = l2_normalise(np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [1.0, 0.1]]))
    names, views, ids = ["A", "A", "B", "C"], ["left", "left", "left", "right"], ["a1", "a2", "b1", "c1"]
    q = l2_normalise(np.array([[0.5, 0.9], [1.0, 0.0]]))
    out = shortlists(q, ["left", "right"], cat, names, views, ids, k=5)
    assert [c["leopard"] for c in out[0]] == ["A", "B"]  # A once, via its best crop a2
    assert out[0][0]["example_crop_id"] == "a2"
    assert [c["leopard"] for c in out[1]] == ["C"]  # only C has a right flank
    assert shortlists(q, ["front", "left"], cat, names, views, ids)[0] == []
