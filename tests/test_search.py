import numpy as np
import pandas as pd

from ghats.search import Index, average_precision, event_max, parse_query, precision_at

VOCAB = {
    "wild_boar": ["wild boar", "wild pig", "pig"],
    "mouse_deer": ["mouse deer"],
    "rodent": ["rat", "mouse"],
    "civet": ["civet", "civets"],
    "small_cat": ["leopard cat", "cat"],
}


def test_parse_species_and_time():
    q = parse_query("Civets at night", VOCAB)
    assert q.species == ["civet"] and q.time == "night" and q.rest == ""


def test_parse_longest_name_wins():
    assert parse_query("mouse deer", VOCAB).species == ["mouse_deer"]
    assert parse_query("a mouse", VOCAB).species == ["rodent"]
    assert parse_query("leopard cat", VOCAB).species == ["small_cat"]
    assert parse_query("mouse deer and a rat", VOCAB).species == ["mouse_deer", "rodent"]


def test_parse_plurals():
    q = parse_query("two wild pigs and some civets", VOCAB)
    assert q.species == ["civet", "wild_boar"] and q.rest == ""


def test_parse_counts_cameras_dates():
    q = parse_query("two wild pigs on camera 12 in March 2018", VOCAB)
    assert q.species == ["wild_boar"]
    assert q.min_count == 2 and q.cameras == ["loc_0012"] and q.years == [2018] and q.months == [3]


def test_parse_keeps_the_rest_for_the_embeddings():
    q = parse_query("civet drinking near water at night", VOCAB)
    assert q.species == ["civet"] and q.time == "night"
    assert q.rest == "drinking near water"


def test_parse_empty_and_day_night():
    q = parse_query("empty photos", VOCAB)
    assert q.empty and not q.animal and q.rest == ""
    assert parse_query("animals by day and night", VOCAB).time is None
    assert parse_query("an animal in daylight", VOCAB).time == "day"
    assert parse_query("deer close to the camera", VOCAB).rest == "deer close to"


def test_word_inside_another_word_is_not_a_match():
    q = parse_query("category ratio", VOCAB)
    assert q.species == [] and q.rest == "category ratio"


def test_event_max_and_ap():
    assert list(event_max(np.array([0.1, 0.5, 0.2]), np.array([0, 0, 1]), 3)) == [0.5, 0.2, -np.inf]
    assert average_precision(np.array([3.0, 2.0, 1.0]), np.array([1, 0, 1])) == (1 + 2 / 3) / 2
    assert np.isnan(average_precision(np.array([1.0]), np.array([0])))
    assert precision_at(np.array([3.0, 2.0, 1.0]), np.array([1, 0, 1]), 2) == 0.5


def make_index() -> Index:
    events = pd.DataFrame(
        {
            "seq_key": ["a", "b", "c", "d"],
            "location": ["loc_0001", "loc_0001", "loc_0002", "loc_0002"],
            "start": pd.to_datetime(["2018-03-01 02:00", "2018-03-01 12:00", "2019-05-01 22:00", "2019-05-02 09:00"]),
            "hour": [2, 12, 22, 9],
            "colour": [1.0, 30.0, 2.0, 25.0],
            "n_boxes": [1, 2, 0, 1],
            "p_animal": [0.99, 0.95, 0.05, 0.9],
            "p_civet": [0.9, 0.01, 0.01, 0.2],
            "p_wild_boar": [0.01, 0.9, 0.01, 0.6],
        }
    )
    e = np.eye(3, dtype=np.float32)
    frame_vecs = np.stack([e[0], e[1], e[2], e[1]]).astype(np.float16)
    crop_vecs = np.stack([e[0], e[1], e[1]]).astype(np.float16)
    return Index(
        events=events,
        frame_vecs=frame_vecs,
        frame_event=np.array([0, 1, 2, 3]),
        crop_vecs=crop_vecs,
        crop_event=np.array([0, 1, 3]),
        groups=["civet", "wild_boar"],
    )


def test_filters():
    idx = make_index()
    assert list(idx.mask(parse_query("civet at night", VOCAB))) == [True, False, False, False]
    assert list(idx.mask(parse_query("pigs", VOCAB))) == [False, True, False, True]
    assert list(idx.mask(parse_query("empty", VOCAB))) == [False, False, True, False]
    assert list(idx.mask(parse_query("two animals", VOCAB))) == [False, True, False, False]
    assert list(idx.mask(parse_query("camera 2 in 2019", VOCAB))) == [False, False, True, True]
    assert list(idx.mask(parse_query("anything in May", VOCAB))) == [False, False, False, True]


def test_ranking():
    idx = make_index()
    calls = []

    def encode(texts):
        calls.append(texts)
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    # Only filters: ranked by the probe, the text model isn't needed.
    assert idx.search(parse_query("pigs", VOCAB), encode) == [1, 3]
    assert calls == []
    # Words left over: ranked by similarity, filters still apply.
    assert idx.search(parse_query("pig sleeping", VOCAB), encode) == [1, 3]
    assert idx.search(parse_query("something sleeping", VOCAB), encode)[:2] == [1, 3]
    assert len(calls) == 2
