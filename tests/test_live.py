import numpy as np
import pytest
from PIL import Image

from ghats.live import Box, EventGrouper, FrameResult, SpeciesHead, resize_bilinear, species_input


def frame(t, *p_animal, species="sambar"):
    return FrameResult(t, [Box([0.1, 0.1, 0.3, 0.3], 0.5, species, 0.9, p) for p in p_animal], {})


def test_events_split_on_gaps_and_judge_by_best_box():
    g = EventGrouper(gap=10)
    assert g.add(frame(0, 0.1), "a") is None
    assert g.add(frame(1, 0.8, 0.2), "b") is None
    done = g.add(frame(30, 0.05), "c")  # 29 s later: a new trigger
    assert [f.t for f in done.frames] == [0, 1]
    s = done.summary(keep_threshold=0.5)
    assert s["kept"] and s["species"] == "sambar" and s["best_frame"] == "b" and s["frames"] == 2
    last = g.flush().summary(keep_threshold=0.5)
    assert not last["kept"] and last["species"] is None


def test_event_species_is_a_weighted_vote_across_frames():
    g = EventGrouper(gap=10)
    g.add(frame(0, 0.9, species="civet"))
    g.add(frame(1, 0.6, species="marten"))
    g.add(frame(2, 0.6, species="marten"))
    assert g.flush().species()[0] == "marten"  # two decent frames outvote one good one


def test_empty_event_has_no_species():
    e = EventGrouper()
    e.add(FrameResult(0, [], {}))
    s = e.flush().summary(0.27)
    assert s["p_animal"] == 0.0 and not s["kept"] and s["best_box"] is None
    assert e.flush() is None  # nothing open


def test_head_is_a_softmax_over_scaled_features(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "head.npz"
    np.savez(
        path,
        mean=np.zeros(4),
        scale=np.ones(4) * 2,
        coef=rng.normal(size=(3, 4)),
        intercept=np.zeros(3),
        classes=np.array(["background", "civet", "sambar"]),
    )
    head = SpeciesHead(path)
    p = head.probs(rng.normal(size=(5, 4)))
    assert p.shape == (5, 3) and np.allclose(p.sum(1), 1)


def test_resize_matches_torch_bilinear_without_antialias():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    x = np.random.default_rng(1).random((53, 97, 3), dtype=np.float32)
    ref = (
        F.interpolate(
            torch.from_numpy(x).permute(2, 0, 1)[None],
            size=(31, 31),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    assert np.abs(resize_bilinear(x, 31) - ref).max() < 1e-5
    up = (
        F.interpolate(
            torch.from_numpy(x).permute(2, 0, 1)[None],
            size=(120, 120),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    assert np.abs(resize_bilinear(x, 120) - up).max() < 1e-5


def test_species_input_matches_speciesnet_preprocess():
    pytest.importorskip("torchvision")
    from ghats.baselines import speciesnet_preprocess

    im = Image.fromarray(np.random.default_rng(2).integers(0, 256, (211, 157, 3), dtype=np.uint8))
    ref = speciesnet_preprocess(im).numpy()
    ours = species_input(im)
    # The uint8 round trip can flip a value by one step where torch and numpy round 1e-7 apart.
    assert np.abs(ours - ref).max() <= 1 / 255 + 1e-6
    assert (np.abs(ours - ref) > 1e-6).mean() < 0.001


def test_video_spans_keep_only_animal_frames():
    from ghats.live import AnimalSpans

    s = AnimalSpans(keep_threshold=0.5, gap=3)
    assert s.add(frame(0.0, 0.1)) is None  # empty: nothing opens
    assert s.add(frame(0.5, 0.9), "0:00.5") is None
    assert s.add(frame(1.0, 0.2)) is None  # a missed frame mid-span doesn't end it
    assert s.add(frame(2.0, 0.8), "0:02.0") is None
    done = s.add(frame(6.0, 0.1))  # 4 s with no animal
    assert done.names == ["0:00.5", "0:02.0"] and done.summary(0.5)["kept"]
    assert s.flush() is None


def test_camera_code_needs_no_pandas_torch_or_opencv():
    # A Raspberry Pi only gets numpy, Pillow, PyYAML and onnxruntime installed.
    import subprocess
    import sys

    code = (
        "import sys\n"
        "for m in ['pandas', 'torch', 'torchvision', 'sklearn', 'cv2', 'matplotlib', 'timm']: sys.modules[m] = None\n"
        "import ghats.live, ghats.config\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
