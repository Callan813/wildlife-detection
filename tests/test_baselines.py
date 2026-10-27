import numpy as np
import pytest

from ghats.baselines import OTHER, group_probs, speciesnet_group

RULES = [
    {"common": "blank", "group": "background"},
    {"genus": "muntiacus", "group": "barking_deer"},
    {"family": "cervidae", "group": "deer"},
]


class TestSpeciesNetGroup:
    def test_first_matching_rule_wins(self):
        assert speciesnet_group("u;mammalia;artiodactyla;cervidae;muntiacus;vaginalis;northern red muntjac", RULES) == (
            "barking_deer"
        )
        assert speciesnet_group("u;mammalia;artiodactyla;cervidae;axis;axis;chital", RULES) == "deer"

    def test_blank_and_unmapped(self):
        assert speciesnet_group("u;;;;;;blank", RULES) == "background"
        assert speciesnet_group("u;mammalia;primates;hominidae;homo;sapiens;human", RULES) == OTHER

    def test_bad_label(self):
        with pytest.raises(ValueError):
            speciesnet_group("sambar", RULES)


def test_group_probs_sums_classes_and_keeps_other():
    probs = np.array([[0.5, 0.2, 0.3], [0.1, 0.1, 0.8]])
    out = group_probs(probs, ["a", "a", "zebra"], ["a", "b"])
    assert np.allclose(out, [[0.7, 0.0, 0.3], [0.2, 0.0, 0.8]])
    assert out.sum(1) == pytest.approx([1.0, 1.0])
