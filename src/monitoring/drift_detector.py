"""
Phase 10: Statistical Drift Detection
=====================================

Detects feature and output score drift between a reference distribution
(e.g., last N training matches) and a current batch (e.g., new season data).

Two drift statistics are computed per feature:

  PSI  (Population Stability Index)
       PSI = Σ (actual% − expected%) × ln(actual% / expected%)
       < 0.1   → no meaningful drift
       0.1–0.2 → moderate drift (investigate)
       > 0.2   → significant drift (retrain)

  KL divergence  KL(current ‖ reference)
       KL = Σ P(i) × ln(P(i) / Q(i))
       Alert threshold configurable; 0.1 is a common default.

Both measures use equal-frequency bins derived from the *reference* distribution
for numeric features, and direct category counts for categorical features.

Typical usage
-------------
    detector = DriftDetector(
        numeric_cols=["distance_to_goal", "in_box"],
        psi_threshold=0.2,
    )
    detector.fit(reference_df)
    report = detector.detect(current_df)
    if report.triggered_alerts:
        print("Drift detected:", report.triggered_alerts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard industry PSI alert thresholds
PSI_NO_DRIFT = 0.1
PSI_MODERATE = 0.2      # > this → significant
_EPSILON = 1e-8         # prevent log(0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_numeric_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Build equal-frequency bin edges from ``values``.

    Duplicate edges are collapsed so that every returned bin covers a
    distinct range. Returns an array of at least 2 edges.
    """
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.nanpercentile(values, percentiles)
    edges = np.unique(edges)  # remove duplicate edges from constant regions
    if len(edges) < 2:
        # Degenerate case: all values are identical
        val = float(edges[0])
        edges = np.array([val - 0.5, val + 0.5])
    return edges


def _bin_numeric(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Assign each value to a bin defined by ``edges``.

    Values below the first edge or above the last edge are clamped into
    the outermost bins so that new-range values in ``current`` don't
    produce zero-count bins.
    """
    clipped = np.clip(values, edges[0], edges[-1])
    indices = np.searchsorted(edges[1:-1], clipped, side="right")
    return indices


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: np.ndarray | None = None,
    n_bins: int = 10,
) -> float:
    """
    Compute PSI between ``reference`` and ``current`` distributions.

    Parameters
    ----------
    reference : array-like of float
        Reference (training) distribution.
    current : array-like of float
        Current (deployment) distribution.
    bins : np.ndarray | None
        Pre-computed bin edges (from reference). If None, derived here.
    n_bins : int
        Number of bins to use when ``bins`` is None.

    Returns
    -------
    float  PSI value ≥ 0.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    if bins is None:
        bins = _make_numeric_bins(ref, n_bins)

    ref_counts = np.bincount(_bin_numeric(ref, bins), minlength=len(bins) - 1)
    cur_counts = np.bincount(_bin_numeric(cur, bins), minlength=len(bins) - 1)

    ref_pct = (ref_counts + _EPSILON) / (ref_counts.sum() + _EPSILON * len(ref_counts))
    cur_pct = (cur_counts + _EPSILON) / (cur_counts.sum() + _EPSILON * len(cur_counts))

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(psi, 0.0)


def compute_psi_categorical(
    reference: Sequence,
    current: Sequence,
) -> float:
    """
    Compute PSI for a categorical feature using observed category proportions.

    Categories present in ``current`` but not ``reference`` are included with
    an epsilon reference proportion.
    """
    ref_series = pd.Series(reference)
    cur_series = pd.Series(current)

    all_cats = sorted(set(ref_series.dropna()) | set(cur_series.dropna()))
    ref_counts = ref_series.value_counts()
    cur_counts = cur_series.value_counts()

    ref_pct = np.array(
        [(ref_counts.get(c, 0) + _EPSILON) for c in all_cats], dtype=float
    )
    cur_pct = np.array(
        [(cur_counts.get(c, 0) + _EPSILON) for c in all_cats], dtype=float
    )
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(psi, 0.0)


def compute_kl_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    bins: np.ndarray | None = None,
    n_bins: int = 10,
) -> float:
    """
    Compute KL(current ‖ reference) using the same binning as PSI.

    Returns 0.0 if either array is empty.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    if bins is None:
        bins = _make_numeric_bins(ref, n_bins)

    ref_counts = np.bincount(_bin_numeric(ref, bins), minlength=len(bins) - 1)
    cur_counts = np.bincount(_bin_numeric(cur, bins), minlength=len(bins) - 1)

    q = (ref_counts + _EPSILON) / (ref_counts.sum() + _EPSILON * len(ref_counts))
    p = (cur_counts + _EPSILON) / (cur_counts.sum() + _EPSILON * len(cur_counts))

    kl = float(np.sum(p * np.log(p / q)))
    return max(kl, 0.0)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class DriftEntry:
    """Drift statistics for a single feature."""

    feature: str
    is_categorical: bool
    psi: float
    kl_divergence: float
    psi_alert: bool
    kl_alert: bool

    @property
    def any_alert(self) -> bool:
        return self.psi_alert or self.kl_alert

    @property
    def severity(self) -> str:
        """'none' | 'moderate' | 'significant'"""
        if self.psi > PSI_MODERATE:
            return "significant"
        if self.psi > PSI_NO_DRIFT:
            return "moderate"
        return "none"


@dataclass
class DriftReport:
    """Drift report for a batch of current data vs a reference set."""

    entries: list[DriftEntry]
    reference_n: int
    current_n: int
    psi_threshold: float
    kl_threshold: float

    @property
    def n_psi_alerts(self) -> int:
        return sum(1 for e in self.entries if e.psi_alert)

    @property
    def n_kl_alerts(self) -> int:
        return sum(1 for e in self.entries if e.kl_alert)

    @property
    def n_alerts(self) -> int:
        return sum(1 for e in self.entries if e.any_alert)

    @property
    def triggered_alerts(self) -> list[str]:
        """Feature names with any triggered alert, sorted by PSI descending."""
        alerted = [e for e in self.entries if e.any_alert]
        return [e.feature for e in sorted(alerted, key=lambda e: e.psi, reverse=True)]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "feature": e.feature,
                "is_categorical": e.is_categorical,
                "psi": e.psi,
                "kl_divergence": e.kl_divergence,
                "severity": e.severity,
                "psi_alert": e.psi_alert,
                "kl_alert": e.kl_alert,
            }
            for e in self.entries
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("psi", ascending=False).reset_index(drop=True)
        return df

    def summary(self) -> dict:
        """High-level summary dict suitable for logging."""
        return {
            "reference_n": self.reference_n,
            "current_n": self.current_n,
            "n_features_checked": len(self.entries),
            "n_psi_alerts": self.n_psi_alerts,
            "n_kl_alerts": self.n_kl_alerts,
            "n_total_alerts": self.n_alerts,
            "triggered": self.triggered_alerts,
        }


# ── Detector ──────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Fits reference distributions and detects drift in subsequent batches.

    Parameters
    ----------
    numeric_cols : list[str]
        Continuous / ordinal feature columns to monitor with PSI + KL.
    categorical_cols : list[str] | None
        Categorical feature columns monitored with categorical PSI.
    psi_threshold : float
        PSI value above which an alert is triggered (default 0.2).
    kl_threshold : float
        KL divergence above which an alert is triggered (default 0.1).
    n_bins : int
        Number of equal-frequency bins for numeric features (default 10).
    """

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str] | None = None,
        psi_threshold: float = PSI_MODERATE,
        kl_threshold: float = 0.1,
        n_bins: int = 10,
    ) -> None:
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.n_bins = n_bins

        # Filled by fit()
        self._reference_values: dict[str, np.ndarray] = {}
        self._reference_bins: dict[str, np.ndarray] = {}
        self._reference_n: int = 0
        self._fitted: bool = False

    def fit(self, reference_df: pd.DataFrame) -> "DriftDetector":
        """
        Store reference distributions from ``reference_df``.

        Only columns present in the DataFrame are stored; missing columns are
        silently skipped and will be omitted from ``detect()`` reports.

        Parameters
        ----------
        reference_df : DataFrame

        Returns
        -------
        self
        """
        self._reference_values = {}
        self._reference_bins = {}
        self._reference_n = len(reference_df)

        for col in self.numeric_cols:
            if col not in reference_df.columns:
                logger.debug("DriftDetector.fit: column %r not in reference_df, skipping.", col)
                continue
            vals = reference_df[col].dropna().astype(float).values
            if len(vals) == 0:
                continue
            self._reference_values[col] = vals
            self._reference_bins[col] = _make_numeric_bins(vals, self.n_bins)

        for col in self.categorical_cols:
            if col not in reference_df.columns:
                logger.debug("DriftDetector.fit: column %r not in reference_df, skipping.", col)
                continue
            self._reference_values[col] = reference_df[col].dropna().values

        self._fitted = True
        logger.info(
            "DriftDetector fitted on %d rows; monitoring %d numeric + %d categorical features.",
            self._reference_n,
            len([c for c in self.numeric_cols if c in self._reference_values]),
            len([c for c in self.categorical_cols if c in self._reference_values]),
        )
        return self

    def detect(self, current_df: pd.DataFrame) -> DriftReport:
        """
        Compute drift between the fitted reference and ``current_df``.

        Parameters
        ----------
        current_df : DataFrame

        Returns
        -------
        DriftReport

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError(
                "DriftDetector.detect() called before fit(). "
                "Call fit(reference_df) first."
            )

        entries: list[DriftEntry] = []

        # Numeric features
        for col in self.numeric_cols:
            if col not in self._reference_values:
                continue
            if col not in current_df.columns:
                logger.debug("detect: column %r missing from current_df, skipping.", col)
                continue

            cur_vals = current_df[col].dropna().astype(float).values
            ref_vals = self._reference_values[col]
            bins = self._reference_bins[col]

            psi = compute_psi(ref_vals, cur_vals, bins=bins)
            kl = compute_kl_divergence(ref_vals, cur_vals, bins=bins)

            entries.append(
                DriftEntry(
                    feature=col,
                    is_categorical=False,
                    psi=psi,
                    kl_divergence=kl,
                    psi_alert=psi > self.psi_threshold,
                    kl_alert=kl > self.kl_threshold,
                )
            )

        # Categorical features
        for col in self.categorical_cols:
            if col not in self._reference_values:
                continue
            if col not in current_df.columns:
                logger.debug("detect: column %r missing from current_df, skipping.", col)
                continue

            cur_vals = current_df[col].dropna().values
            ref_vals = self._reference_values[col]

            psi = compute_psi_categorical(ref_vals, cur_vals)
            # KL for categorical: same as PSI but with KL formula
            kl = compute_kl_divergence(
                np.arange(len(ref_vals)).astype(float),
                np.arange(len(cur_vals)).astype(float),
            )  # Simplified: just use PSI*0.5 for categorical KL
            # Actually compute proper KL for categorical
            all_cats = sorted(set(ref_vals) | set(cur_vals))
            ref_s = pd.Series(ref_vals).value_counts()
            cur_s = pd.Series(cur_vals).value_counts()
            q = np.array([(ref_s.get(c, 0) + _EPSILON) for c in all_cats], dtype=float)
            p = np.array([(cur_s.get(c, 0) + _EPSILON) for c in all_cats], dtype=float)
            q /= q.sum()
            p /= p.sum()
            kl = float(max(np.sum(p * np.log(p / q)), 0.0))

            entries.append(
                DriftEntry(
                    feature=col,
                    is_categorical=True,
                    psi=psi,
                    kl_divergence=kl,
                    psi_alert=psi > self.psi_threshold,
                    kl_alert=kl > self.kl_threshold,
                )
            )

        return DriftReport(
            entries=entries,
            reference_n=self._reference_n,
            current_n=len(current_df),
            psi_threshold=self.psi_threshold,
            kl_threshold=self.kl_threshold,
        )
