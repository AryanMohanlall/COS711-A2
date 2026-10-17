"""Grouped train/validation splitting.

Rules from the spec:
- annotation sets of the same observation must never be split across train and val;
- observations close in time, or from the same instrument, are not independent.

Approach: split at the observation level (so all annotation sets of an observation move
together), group observations into time windows (chained: a new group starts after a gap
longer than `max_gap`), and stratify folds by site so every fold sees every GONG site.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def time_groups(times: pd.Series, max_gap: str = "48h") -> pd.Series:
    """Group ids for observations, chaining any observations less than `max_gap` apart."""
    order = times.sort_values()
    new_group = order.diff() > pd.Timedelta(max_gap)
    return new_group.cumsum().reindex(times.index).rename("group")


def make_folds(obs: pd.DataFrame, n_splits: int = 5, max_gap: str = "48h", seed: int = 42) -> pd.DataFrame:
    """Assign a fold to every observation.

    `obs` needs one row per observation with columns obs, time, site.
    Returns obs with added `group` and `fold` columns.
    """
    out = obs.copy()
    out["group"] = time_groups(out["time"], max_gap)
    out["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(sgkf.split(out, y=out["site"], groups=out["group"])):
        out.iloc[val_idx, out.columns.get_loc("fold")] = fold
    return out
