"""Group camera-trap frames into trigger events.

A camera trap fires a burst of 3-10 frames per trigger. Classifying (and
evaluating) per event instead of per frame is faster and closer to how
ecologists count detections. It also matters for evaluation: LILA labels are
often assigned to a whole sequence, so a frame where the animal has already
left the field of view can still be labelled "wild pig". Event-level metrics
are not distorted by that label noise.
"""

from __future__ import annotations

import pandas as pd


def assign_events(df: pd.DataFrame, max_gap_seconds: float = 120) -> pd.Series:
    """Return an event id for every row of ``df``.

    Uses ``seq_id`` when present. Rows without one are grouped per location:
    a new event starts when the gap to the previous frame exceeds
    ``max_gap_seconds`` or the timestamp is missing.
    """
    if df.empty:
        return pd.Series([], dtype=object, index=df.index)

    event = pd.Series(index=df.index, dtype=object)

    has_seq = df["seq_id"].notna() if "seq_id" in df else pd.Series(False, index=df.index)
    event[has_seq] = "seq:" + df.loc[has_seq, "seq_id"].astype(str)

    rest = df.loc[~has_seq]
    if not rest.empty:
        ordered = rest.sort_values(["location", "datetime"], na_position="last")
        gap = ordered.groupby("location")["datetime"].diff().dt.total_seconds()
        new_event = gap.isna() | (gap > max_gap_seconds)
        # Separate counters per location keep ids readable: loc:<location>:<n>
        counter = new_event.astype(int).groupby(ordered["location"]).cumsum()
        ids = "loc:" + ordered["location"].astype(str) + ":" + counter.astype(str)
        event[ordered.index] = ids

    return event.astype(str)


def event_table(images: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Aggregate frame rows (with event_id, gt, group, score) to one row per event.

    An event is an animal event if any frame is; its score is the max frame score.
    """
    agg = images.groupby("event_id").agg(
        gt=("gt", lambda s: "animal" if (s == "animal").any() else "empty"),
        group=("group", lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        score=(score_col, "max"),
        n_frames=("gt", "size"),
        location=("location", "first"),
    )
    return agg.reset_index()
