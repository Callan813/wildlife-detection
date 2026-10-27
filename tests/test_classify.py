import numpy as np
import pytest

from ghats.classify import class_weights, confusion_matrix, per_class_report


class TestClassWeights:
    def test_power_zero_is_uniform(self):
        assert np.allclose(class_weights([10, 100, 1000], power=0), 1.0)

    def test_rare_class_weighs_more_and_mean_is_one(self):
        w = class_weights([100, 400], power=0.5)
        assert w[0] == pytest.approx(2 * w[1])
        assert w.mean() == pytest.approx(1.0)

    def test_empty_class_rejected(self):
        with pytest.raises(ValueError):
            class_weights([5, 0])


class TestReport:
    def test_counts(self):
        # class 0: 2 true, both found, 1 false alarm; class 1: 2 true, 1 found; class 2 absent
        rep = per_class_report([0, 0, 1, 1], [0, 0, 0, 1], ["a", "b", "c"]).set_index("class")
        assert rep.loc["a", "precision"] == pytest.approx(2 / 3)
        assert rep.loc["a", "recall"] == 1.0
        assert rep.loc["b", "f1"] == pytest.approx(2 * 1 * 0.5 / 1.5)
        assert np.isnan(rep.loc["c", "recall"])
        # macro avg skips the class with no examples
        assert rep.loc["macro avg", "f1"] == pytest.approx((rep.loc["a", "f1"] + rep.loc["b", "f1"]) / 2)

    def test_confusion_matrix(self):
        m = confusion_matrix([0, 0, 1], [0, 1, 1], 2)
        assert m.tolist() == [[1, 1], [0, 1]]
