"""
Validation split utilities.

Splits are always at match level — never at event level — to prevent
match-context leakage.

Available strategies:
  match_kfold            — k-fold where each fold contains whole matches
  competition_holdout    — hold out one or more competitions entirely
  team_holdout           — hold out all matches involving one or more teams
  temporal_split         — chronological split by match_date
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def match_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    match_id_col: str = "match_id",
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, val_idx) pairs where whole matches are kept together.

    Parameters
    ----------
    df            : DataFrame with a match_id column
    n_splits      : number of folds
    match_id_col  : column identifying the match
    random_state  : reproducibility seed

    Yields
    ------
    (train_indices, val_indices) as numpy integer arrays into df.index
    """
    rng = np.random.default_rng(random_state)
    match_ids = df[match_id_col].unique()
    rng.shuffle(match_ids)
    folds = np.array_split(match_ids, n_splits)

    for i in range(n_splits):
        val_matches = set(folds[i])
        train_matches = set(mid for j, fold in enumerate(folds) if j != i for mid in fold)
        train_idx = df.index[df[match_id_col].isin(train_matches)].to_numpy()
        val_idx = df.index[df[match_id_col].isin(val_matches)].to_numpy()
        logger.debug(
            "Fold %d/%d — train: %d events (%d matches) | val: %d events (%d matches)",
            i + 1,
            n_splits,
            len(train_idx),
            len(train_matches),
            len(val_idx),
            len(val_matches),
        )
        yield train_idx, val_idx


def competition_holdout(
    df: pd.DataFrame,
    holdout_competition_ids: list[str],
    competition_id_col: str = "competition_id",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (train_idx, holdout_idx) by holding out entire competitions.

    Parameters
    ----------
    holdout_competition_ids : list of competition_id values to hold out

    Returns
    -------
    (train_indices, holdout_indices)
    """
    holdout_set = set(str(c) for c in holdout_competition_ids)
    in_holdout = df[competition_id_col].astype(str).isin(holdout_set)
    train_idx = df.index[~in_holdout].to_numpy()
    holdout_idx = df.index[in_holdout].to_numpy()
    logger.info(
        "Competition holdout: %d train events | %d holdout events (competitions: %s)",
        len(train_idx),
        len(holdout_idx),
        holdout_competition_ids,
    )
    return train_idx, holdout_idx


def team_holdout(
    df: pd.DataFrame,
    holdout_team_ids: list[str],
    team_id_col: str = "team_id",
    opponent_id_col: str = "opponent_id",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hold out all events involving specific teams (as attacker or opponent).
    Useful for stress-testing generalisation to unseen teams.
    """
    holdout_set = set(str(t) for t in holdout_team_ids)
    in_holdout = df[team_id_col].astype(str).isin(holdout_set) | (
        df[opponent_id_col].astype(str).isin(holdout_set)
        if opponent_id_col in df.columns
        else pd.Series(False, index=df.index)
    )
    train_idx = df.index[~in_holdout].to_numpy()
    holdout_idx = df.index[in_holdout].to_numpy()
    logger.info(
        "Team holdout: %d train events | %d holdout events (teams: %s)",
        len(train_idx),
        len(holdout_idx),
        holdout_team_ids,
    )
    return train_idx, holdout_idx


def temporal_split(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
    date_col: str = "match_date",
    match_id_col: str = "match_id",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chronological split: most-recent `test_fraction` of matches form the test set.
    Operates at match level to avoid leakage.
    """
    match_dates = df.groupby(match_id_col)[date_col].first().sort_values()
    n_test = max(1, int(len(match_dates) * test_fraction))
    test_matches = set(match_dates.index[-n_test:])
    train_matches = set(match_dates.index[:-n_test])

    train_idx = df.index[df[match_id_col].isin(train_matches)].to_numpy()
    test_idx = df.index[df[match_id_col].isin(test_matches)].to_numpy()
    logger.info(
        "Temporal split: %d train events (%d matches) | %d test events (%d matches)",
        len(train_idx),
        len(train_matches),
        len(test_idx),
        len(test_matches),
    )
    return train_idx, test_idx
