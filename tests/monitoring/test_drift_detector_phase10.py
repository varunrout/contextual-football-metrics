"""
Phase 10 tests — Drift Detection (DriftDetector).

Covers:
  • compute_psi             — identical, shifted, extreme, empty inputs
  • compute_psi_categorical — identical, different distributions
  • compute_kl_divergence   — identical, shifted
  • _make_numeric_bins      — edge: all identical values
  • DriftEntry              — any_alert, severity properties
  • DriftReport             — n_psi_alerts, triggered_alerts, to_dataframe, summary
  • DriftDetector.fit       — columns stored, missing columns skipped
  • DriftDetector.detect    — returns DriftReport, alerts triggered correctly,
                              columns missing from current skipped gracefully,
                              categorical features handled
  • DriftDetector.detect    — RuntimeError before fit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import (
    DriftDetector,
    DriftEntry,
    DriftReport,
    PSI_MODERATE,
    PSI_NO_DRIFT,
    _make_numeric_bins,
    compute_kl_divergence,
    compute_psi,
    compute_psi_categorical,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_df(n: int = 500, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    rng = _rng(seed)
    return pd.DataFrame({
        "distance_to_goal": rng.uniform(5, 35, n) + shift,
        "x_location": rng.uniform(20, 105, n) + shift,
        "in_box": rng.integers(0, 2, n).astype(float),
        "score_state": rng.choice(["winning", "drawing", "losing"], n),
        "action_type": rng.choice(["pass", "carry", "shot"], n),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# compute_psi — numeric
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePsi:
    def test_identical_distributions_near_zero(self):
        rng = _rng(0)
        vals = rng.normal(20, 5, 500)
        psi = compute_psi(vals, vals.copy())
        assert psi < 0.01

    def test_shifted_distribution_has_higher_psi(self):
        rng = _rng(0)
        ref = rng.normal(20, 5, 500)
        cur = rng.normal(30, 5, 500)  # large shift
        psi = compute_psi(ref, cur)
        assert psi > PSI_MODERATE

    def test_psi_is_non_negative(self):
        rng = _rng(1)
        ref = rng.normal(0, 1, 300)
        cur = rng.normal(0.5, 1, 300)
        assert compute_psi(ref, cur) >= 0

    def test_empty_reference_returns_zero(self):
        assert compute_psi(np.array([]), np.array([1.0, 2.0])) == 0.0

    def test_empty_current_returns_zero(self):
        assert compute_psi(np.array([1.0, 2.0]), np.array([])) == 0.0

    def test_pre_computed_bins_used(self):
        rng = _rng(0)
        ref = rng.uniform(0, 10, 500)
        cur = rng.uniform(0, 10, 500)
        bins = _make_numeric_bins(ref, n_bins=10)
        psi1 = compute_psi(ref, cur, bins=bins)
        psi2 = compute_psi(ref, cur)  # bins computed internally
        assert abs(psi1 - psi2) < 0.05  # may differ slightly due to bin edge rounding

    def test_current_out_of_reference_range_clamped(self):
        """Values outside reference range should be clamped, not crash."""
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        cur = np.array([0.0, 10.0, 100.0, -5.0] * 100)  # all outside ref range
        psi = compute_psi(ref, cur)
        assert psi >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# compute_psi_categorical
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePsiCategorical:
    def test_identical_distributions_near_zero(self):
        cats = ["a", "b", "c"] * 200
        psi = compute_psi_categorical(cats, cats)
        assert psi < 0.05

    def test_completely_different_distribution_high_psi(self):
        ref = ["a"] * 300 + ["b"] * 100
        cur = ["b"] * 300 + ["a"] * 100
        psi = compute_psi_categorical(ref, cur)
        assert psi > PSI_NO_DRIFT

    def test_new_category_in_current_handled(self):
        ref = ["a", "b", "c"] * 100
        cur = ["a", "b", "d"] * 100  # "d" not in ref
        psi = compute_psi_categorical(ref, cur)
        assert psi >= 0

    def test_non_negative(self):
        ref = ["x", "y", "z"] * 50
        cur = ["x", "y", "z", "w"] * 40
        assert compute_psi_categorical(ref, cur) >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# compute_kl_divergence
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeKlDivergence:
    def test_identical_distributions_near_zero(self):
        rng = _rng(0)
        vals = rng.normal(10, 2, 500)
        kl = compute_kl_divergence(vals, vals.copy())
        assert kl < 0.05

    def test_shifted_distribution_positive_kl(self):
        rng = _rng(0)
        ref = rng.normal(10, 2, 500)
        cur = rng.normal(20, 2, 500)
        kl = compute_kl_divergence(ref, cur)
        assert kl > 0.1

    def test_kl_non_negative(self):
        rng = _rng(2)
        ref = rng.uniform(0, 1, 300)
        cur = rng.uniform(0.2, 1.2, 300)
        assert compute_kl_divergence(ref, cur) >= 0

    def test_empty_arrays_return_zero(self):
        assert compute_kl_divergence(np.array([]), np.array([1.0])) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# _make_numeric_bins
# ═══════════════════════════════════════════════════════════════════════════════

class TestMakeNumericBins:
    def test_returns_at_least_two_edges(self):
        vals = np.array([5.0] * 100)  # all identical
        edges = _make_numeric_bins(vals, n_bins=10)
        assert len(edges) >= 2

    def test_normal_case_length(self):
        vals = np.random.default_rng(0).normal(10, 3, 500)
        edges = _make_numeric_bins(vals, n_bins=10)
        assert len(edges) >= 2  # unique edges may be fewer than n_bins+1

    def test_edges_monotonically_increasing(self):
        vals = np.random.default_rng(0).uniform(0, 100, 500)
        edges = _make_numeric_bins(vals, n_bins=10)
        assert np.all(np.diff(edges) > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# DriftEntry properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestDriftEntry:
    def _make(self, psi: float = 0.05, kl: float = 0.05,
              psi_alert: bool = False, kl_alert: bool = False) -> DriftEntry:
        return DriftEntry("dist", False, psi, kl, psi_alert, kl_alert)

    def test_any_alert_false_when_none(self):
        assert self._make().any_alert is False

    def test_any_alert_true_on_psi_alert(self):
        assert self._make(psi_alert=True).any_alert is True

    def test_any_alert_true_on_kl_alert(self):
        assert self._make(kl_alert=True).any_alert is True

    def test_severity_none_below_threshold(self):
        assert self._make(psi=0.05).severity == "none"

    def test_severity_moderate(self):
        assert self._make(psi=0.15).severity == "moderate"

    def test_severity_significant(self):
        assert self._make(psi=0.25).severity == "significant"


# ═══════════════════════════════════════════════════════════════════════════════
# DriftReport
# ═══════════════════════════════════════════════════════════════════════════════

def _make_report(n_psi: int = 2, n_kl: int = 1) -> DriftReport:
    entries = [
        DriftEntry("f1", False, 0.25, 0.12, True, True),   # psi+kl alert
        DriftEntry("f2", False, 0.22, 0.05, True, False),   # psi alert only
        DriftEntry("f3", False, 0.04, 0.02, False, False),  # no alert
    ]
    return DriftReport(entries=entries, reference_n=500, current_n=300,
                       psi_threshold=0.2, kl_threshold=0.1)


class TestDriftReport:
    def test_n_psi_alerts(self):
        assert _make_report().n_psi_alerts == 2

    def test_n_kl_alerts(self):
        assert _make_report().n_kl_alerts == 1

    def test_n_alerts_counts_any_alert(self):
        assert _make_report().n_alerts == 2

    def test_triggered_alerts_sorted_by_psi(self):
        alerts = _make_report().triggered_alerts
        assert alerts[0] == "f1"   # psi=0.25 > f2 psi=0.22
        assert alerts[1] == "f2"
        assert len(alerts) == 2

    def test_to_dataframe_columns(self):
        df = _make_report().to_dataframe()
        for col in ("feature", "psi", "kl_divergence", "severity", "psi_alert", "kl_alert"):
            assert col in df.columns

    def test_to_dataframe_sorted_by_psi_descending(self):
        df = _make_report().to_dataframe()
        assert df["psi"].tolist() == sorted(df["psi"].tolist(), reverse=True)

    def test_to_dataframe_row_count(self):
        assert len(_make_report().to_dataframe()) == 3

    def test_summary_dict_has_required_keys(self):
        s = _make_report().summary()
        for key in ("reference_n", "current_n", "n_features_checked",
                    "n_psi_alerts", "n_kl_alerts", "n_total_alerts", "triggered"):
            assert key in s

    def test_summary_n_features_checked(self):
        assert _make_report().summary()["n_features_checked"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# DriftDetector.fit
# ═══════════════════════════════════════════════════════════════════════════════

class TestDriftDetectorFit:
    def test_fit_returns_self(self):
        df = _make_df()
        det = DriftDetector(numeric_cols=["distance_to_goal"])
        assert det.fit(df) is det

    def test_fitted_flag_set(self):
        df = _make_df()
        det = DriftDetector(numeric_cols=["distance_to_goal"])
        det.fit(df)
        assert det._fitted is True

    def test_reference_n_stored(self):
        df = _make_df(n=300)
        det = DriftDetector(numeric_cols=["distance_to_goal"])
        det.fit(df)
        assert det._reference_n == 300

    def test_missing_column_silently_skipped(self):
        df = _make_df()
        det = DriftDetector(numeric_cols=["nonexistent_column"])
        det.fit(df)  # should not raise
        assert "nonexistent_column" not in det._reference_values

    def test_reference_values_stored_for_present_columns(self):
        df = _make_df()
        det = DriftDetector(numeric_cols=["distance_to_goal", "x_location"])
        det.fit(df)
        assert "distance_to_goal" in det._reference_values
        assert "x_location" in det._reference_values

    def test_categorical_columns_stored(self):
        df = _make_df()
        det = DriftDetector(numeric_cols=[], categorical_cols=["action_type"])
        det.fit(df)
        assert "action_type" in det._reference_values


# ═══════════════════════════════════════════════════════════════════════════════
# DriftDetector.detect
# ═══════════════════════════════════════════════════════════════════════════════

class TestDriftDetectorDetect:
    @pytest.fixture(scope="class")
    def fitted_detector(self):
        ref = _make_df(n=500, seed=0)
        det = DriftDetector(
            numeric_cols=["distance_to_goal", "x_location", "in_box"],
            categorical_cols=["score_state", "action_type"],
            psi_threshold=0.2,
            kl_threshold=0.1,
        )
        det.fit(ref)
        return det

    def test_raises_before_fit(self):
        det = DriftDetector(numeric_cols=["x"])
        with pytest.raises(RuntimeError, match="fit"):
            det.detect(pd.DataFrame({"x": [1.0]}))

    def test_returns_drift_report(self, fitted_detector):
        cur = _make_df(n=200, seed=1)
        report = fitted_detector.detect(cur)
        assert isinstance(report, DriftReport)

    def test_report_has_entries_for_each_monitored_column(self, fitted_detector):
        cur = _make_df(n=200, seed=1)
        report = fitted_detector.detect(cur)
        names = {e.feature for e in report.entries}
        # All columns present in both ref and cur should appear
        for col in ("distance_to_goal", "x_location", "score_state"):
            assert col in names

    def test_current_n_recorded(self, fitted_detector):
        cur = _make_df(n=150, seed=2)
        report = fitted_detector.detect(cur)
        assert report.current_n == 150

    def test_no_alerts_for_same_distribution(self, fitted_detector):
        """Re-detect on same-seed data → expect few/no alerts."""
        same_dist = _make_df(n=500, seed=0)
        report = fitted_detector.detect(same_dist)
        # PSI should be very small for identically-distributed data
        for e in report.entries:
            if not e.is_categorical:
                assert e.psi < 0.15

    def test_alerts_triggered_for_large_shift(self):
        ref = _make_df(n=500, seed=0, shift=0.0)
        cur = _make_df(n=500, seed=1, shift=30.0)   # massive shift
        det = DriftDetector(
            numeric_cols=["distance_to_goal"],
            psi_threshold=0.2,
        )
        det.fit(ref)
        report = det.detect(cur)
        dist_entry = next(e for e in report.entries if e.feature == "distance_to_goal")
        assert dist_entry.psi_alert is True

    def test_column_missing_from_current_skipped(self, fitted_detector):
        """If a monitored column is absent from current_df, skip it gracefully."""
        cur = _make_df(n=200).drop(columns=["x_location"])
        report = fitted_detector.detect(cur)
        names = {e.feature for e in report.entries}
        assert "x_location" not in names

    def test_categorical_entries_flagged_as_categorical(self, fitted_detector):
        cur = _make_df(n=200, seed=1)
        report = fitted_detector.detect(cur)
        for e in report.entries:
            if e.feature in ("score_state", "action_type"):
                assert e.is_categorical is True

    def test_numeric_entries_not_flagged_as_categorical(self, fitted_detector):
        cur = _make_df(n=200, seed=1)
        report = fitted_detector.detect(cur)
        for e in report.entries:
            if e.feature in ("distance_to_goal", "x_location"):
                assert e.is_categorical is False

    def test_all_psi_values_non_negative(self, fitted_detector):
        cur = _make_df(n=200, seed=3)
        report = fitted_detector.detect(cur)
        for e in report.entries:
            assert e.psi >= 0

    def test_all_kl_values_non_negative(self, fitted_detector):
        cur = _make_df(n=200, seed=3)
        report = fitted_detector.detect(cur)
        for e in report.entries:
            assert e.kl_divergence >= 0
