"""
Tests for src/models/sequence/sequence_classifier.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.schema import SequenceType
from src.models.sequence.sequence_classifier import (
    _CLASS_TO_IDX,
    _LABEL_CLASSES,
    SequenceClassifier,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_possessions_df(n: int = 40) -> pd.DataFrame:
    """Synthetic possessions DataFrame with all required columns.

    Ensures every label class appears at least once so LightGBM sees all
    classes during training and predict_proba returns the full class matrix.
    """
    rng = np.random.default_rng(42)
    # Tile all classes so each appears at least once, then pad with random choices
    import math

    repeats = math.ceil(n / len(_LABEL_CLASSES))
    base = (_LABEL_CLASSES * repeats)[:n]
    labels = np.array(base)
    rng.shuffle(labels)
    confidences = rng.uniform(0.7, 1.0, size=n)

    return pd.DataFrame(
        {
            "n_events": rng.integers(2, 15, size=n),
            "n_passes": rng.integers(0, 8, size=n),
            "n_carries": rng.integers(0, 6, size=n),
            "vertical_progression": rng.uniform(-5.0, 60.0, size=n),
            "passes_before_action": rng.integers(0, 8, size=n),
            "carries_before_action": rng.integers(0, 5, size=n),
            "time_from_possession_start": rng.uniform(1.0, 40.0, size=n),
            "vertical_progression_speed": rng.uniform(0.0, 10.0, size=n),
            "directness": rng.uniform(0.0, 1.0, size=n),
            "possession_speed": rng.uniform(0.1, 3.0, size=n),
            "number_of_switches": rng.integers(0, 5, size=n),
            "set_piece_flag": rng.choice([True, False], size=n),
            "counterpress_regain_flag": rng.choice([True, False], size=n),
            "start_x": rng.uniform(0.0, 105.0, size=n),
            "start_y": rng.uniform(0.0, 68.0, size=n),
            "possession_start_zone": rng.choice(
                ["defensive_third", "mid_third", "attacking_third"], size=n
            ),
            "regain_zone": rng.choice(["defensive_third", "mid_third", "attacking_third"], size=n),
            "phase_of_play": rng.choice(
                ["buildup", "transition", "final_third", "progression"], size=n
            ),
            "transition_or_settled": rng.choice(["transition", "settled"], size=n),
            "sequence_type_rule": labels,
            "sequence_type_confidence": confidences,
        }
    )


# ── Label classes ─────────────────────────────────────────────────────────────


class TestLabelClasses:
    def test_unknown_not_in_classes(self):
        assert SequenceType.UNKNOWN.value not in _LABEL_CLASSES

    def test_class_to_idx_bijective(self):
        assert len(_CLASS_TO_IDX) == len(_LABEL_CLASSES)
        assert set(_CLASS_TO_IDX.keys()) == set(_LABEL_CLASSES)

    def test_idx_contiguous(self):
        indices = sorted(_CLASS_TO_IDX.values())
        assert indices == list(range(len(_LABEL_CLASSES)))


# ── SequenceClassifier fit / predict ─────────────────────────────────────────


class TestSequenceClassifierFitPredict:
    def test_fit_runs_without_error(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10})
        df = _make_possessions_df(50)
        clf.fit(df)
        assert clf._model is not None

    def test_predict_proba_shape(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10})
        df = _make_possessions_df(50)
        clf.fit(df)
        proba = clf.predict_proba(df)
        assert proba.shape == (len(df), len(_LABEL_CLASSES))

    def test_predict_proba_sums_to_one(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10})
        df = _make_possessions_df(50)
        clf.fit(df)
        proba = clf.predict_proba(df)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_predict_returns_list_of_strings(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10})
        df = _make_possessions_df(50)
        clf.fit(df)
        preds = clf.predict(df)
        assert isinstance(preds, list)
        assert len(preds) == len(df)
        valid = set(_LABEL_CLASSES) | {SequenceType.UNKNOWN.value}
        for p in preds:
            assert p in valid

    def test_label_dataframe_adds_columns(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10})
        df = _make_possessions_df(50)
        clf.fit(df)
        result = clf.label_dataframe(df)
        for col in ["sequence_type", "sequence_type_confidence", "sequence_type_source"]:
            assert col in result.columns

    def test_label_dataframe_source_is_classifier(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10})
        df = _make_possessions_df(50)
        clf.fit(df)
        result = clf.label_dataframe(df)
        assert (result["sequence_type_source"] == "classifier").all()

    def test_fit_raises_on_no_confident_labels(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10}, min_confidence=0.99)
        df = _make_possessions_df(30)
        df["sequence_type_confidence"] = 0.50  # all below threshold
        with pytest.raises(ValueError, match="No possessions"):
            clf.fit(df)

    def test_predict_before_fit_raises(self):
        clf = SequenceClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(pd.DataFrame())

    def test_low_confidence_returns_unknown(self):
        clf = SequenceClassifier(lgb_params={"n_estimators": 10}, predict_threshold=1.1)
        df = _make_possessions_df(30)
        clf.fit(df)
        preds = clf.predict(df)
        assert all(p == SequenceType.UNKNOWN.value for p in preds)
