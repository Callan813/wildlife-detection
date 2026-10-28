import numpy as np
import pandas as pd

from ghats.reid import l2_normalise, retrieval_metrics, split_individuals


def test_split_keeps_individuals_whole_and_hits_shares():
    names = pd.Series([f"leo{i % 40}" for i in range(400)])
    split = split_individuals(names, val_frac=0.15, test_frac=0.3)
    assert set(split) == set(names)
    share = names.map(split).value_counts(normalize=True)
    assert 0.25 <= share["test"] <= 0.4
    assert 0.1 <= share["val"] <= 0.25


def test_same_encounter_matches_do_not_count():
    # Two crops of A from one encounter are identical; the only other A is far away.
    emb = l2_normalise(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]))
    ids = ["A", "A", "A", "B"]
    enc = ["e1", "e1", "e2", "e3"]
    m = retrieval_metrics(emb, ids, enc)
    # Query 0 can't use crop 1 (same encounter); B (crop 3) outranks the far A.
    assert m["top1"] < 1.0
    # Without the encounter rule, queries 0 and 1 would match each other.
    naive = retrieval_metrics(emb, ids, ["a", "b", "c", "d"])
    assert naive["top1"] > m["top1"]


def test_viewpoint_must_match():
    emb = l2_normalise(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    m = retrieval_metrics(emb, ["A", "A", "A"], ["e1", "e2", "e3"], views=["left", "right", "left"])
    # Only crops 0 and 2 (both left) can match each other.
    assert m["queries"] == 2


def test_perfect_embedding_scores_one():
    emb = l2_normalise(np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]]))
    m = retrieval_metrics(emb, ["A", "A", "B", "B"], ["e1", "e2", "e3", "e4"])
    assert m["top1"] == 1.0 and m["mAP"] == 1.0


def test_encounters_are_runs_of_the_same_leopard():
    from ghats.reid import assign_encounters

    # Images 1-2 show A, 3 shows B, 4 shows A again: two A encounters. Image 2 also shows C.
    img = pd.Series([1, 2, 2, 3, 4])
    name = pd.Series(["A", "A", "C", "B", "A"])
    enc = assign_encounters(img, name)
    assert enc[0] == enc[1]
    assert enc[4] != enc[0]
    assert len({enc[2], enc[3], enc[0]}) == 3


def test_per_id_average_weighs_every_leopard_equally():
    # A has 4 crops that all find each other; B's two crops are closer to A than to each other.
    emb = l2_normalise(np.array([[1, 0], [1, 0.01], [1, 0.02], [1, 0.03], [0.9, 0.3], [0.9, -0.3]], dtype=float))
    ids = ["A", "A", "A", "A", "B", "B"]
    m = retrieval_metrics(emb, ids, [f"e{i}" for i in range(6)])
    assert m["top1"] == 4 / 6
    assert m["top1_per_id"] == 0.5


def test_open_set_hidden_leopards_are_new():
    from ghats.reid import open_set_queries, open_set_rates

    # A and B each have two encounters; A's crops are near each other, B's too.
    emb = l2_normalise(np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1]], dtype=float))
    ids, enc, views = ["A", "A", "B", "B"], ["e1", "e2", "e3", "e4"], ["left"] * 4
    q = open_set_queries(emb, ids, enc, views, hidden={"B"})
    # B's queries only see A in the gallery: new. A's queries find each other.
    assert q.loc[q["id"] == "B", "is_new"].all()
    assert q.loc[q["id"] == "A", "top1_ok"].all()
    # A threshold between A-A similarity (~1) and B-A similarity (~0.1) gets everything right.
    r = open_set_rates(q, threshold=0.5)
    assert r["known_identified"] == 1.0 and r["new_flagged"] == 1.0
    # Accept everything: no new leopard is flagged.
    assert open_set_rates(q, threshold=-1.0)["new_flagged"] == 0.0
