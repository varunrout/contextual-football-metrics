"""
Phase 9: Interpretability and Reporting Layer
=============================================

Provides:
  • compute_shap_values          — SHAP explanations for any sklearn-compatible model
                                    (requires optional ``shap`` package)
  • check_coefficient_directions — plausibility of linear-model coefficient signs
                                    against domain-expected directions
  • run_ablation_study           — feature-group importance via leave-one-group-out
  • build_match_report           — per-match action-level narrative
  • build_player_report          — per-player metric summary with optional league context
  • build_interpretability_html  — combined HTML report

All functions are framework-agnostic; models only need to expose
``predict_proba(df)`` (classifiers) or ``predict(df)`` (regressors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as _sk_log_loss, mean_absolute_error as _sk_mae

logger = logging.getLogger(__name__)


# ── Domain-expected coefficient signs ────────────────────────────────────────
# +1 → feature should have a *positive* effect on the metric
# -1 → feature should have a *negative* effect on the metric

EXPECTED_SIGNS_CXG: dict[str, int] = {
    "distance_to_goal": -1,       # farther from goal → lower xG
    "in_box": +1,                  # inside box → higher xG
    "under_pressure": -1,          # under pressure → worse chance quality
    "is_central": +1,              # central position → higher xG
    "progressive_distance": +1,    # further upfield → higher threat
}

EXPECTED_SIGNS_CXT: dict[str, int] = {
    "distance_to_goal": -1,
    "in_box": +1,
    "under_pressure": -1,
    "progressive_distance": +1,
    "nearest_defender_distance": +1,   # more space → higher state value
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SHAPResult:
    """SHAP explanation for a batch of predictions."""

    shap_values: np.ndarray      # shape (n_samples, n_features)
    base_value: float            # global expected value (explainer baseline)
    feature_names: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Return SHAP values as a DataFrame with feature columns."""
        return pd.DataFrame(self.shap_values, columns=self.feature_names)

    def top_features(self, n: int = 10) -> pd.DataFrame:
        """Return features ranked by mean |SHAP| value, descending."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
        return (
            df.sort_values("mean_abs_shap", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def waterfall_data(self, row_idx: int = 0) -> list[dict]:
        """
        Return a list of dicts describing each feature's SHAP contribution
        for a single row, including a running cumulative total.
        """
        row = self.shap_values[row_idx]
        result = []
        running = self.base_value
        for feat, val in zip(self.feature_names, row):
            running += float(val)
            result.append({
                "feature": feat,
                "shap_value": float(val),
                "running_total": running,
            })
        return result


@dataclass
class DirectionViolation:
    """A single feature whose coefficient sign disagrees with domain expectations."""

    feature: str
    expected_sign: int    # +1 or -1
    actual_sign: int
    coefficient: float


@dataclass
class DirectionCheckResult:
    """Result of checking coefficient direction plausibility."""

    violations: list[DirectionViolation]
    n_checked: int
    n_violations: int
    pass_rate: float    # fraction of checked features that passed

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0


@dataclass
class AblationEntry:
    """Performance change from removing one feature group."""

    group_name: str
    features_removed: list[str]
    baseline_metric: float
    ablated_metric: float
    degradation: float            # ablated − baseline (positive = worse for MAE/log_loss)
    relative_degradation: float   # degradation / |baseline|


@dataclass
class AblationResult:
    """Full leave-one-group-out ablation result."""

    baseline_metric: float
    entries: list[AblationEntry]
    metric_name: str

    def to_dataframe(self) -> pd.DataFrame:
        """Return entries as a DataFrame sorted by degradation descending."""
        rows = [
            {
                "group": e.group_name,
                "features_removed": len(e.features_removed),
                "baseline_metric": e.baseline_metric,
                "ablated_metric": e.ablated_metric,
                "degradation": e.degradation,
                "relative_degradation": e.relative_degradation,
            }
            for e in self.entries
        ]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("degradation", ascending=False).reset_index(drop=True)

    def most_important_group(self) -> str | None:
        """Return the group whose removal causes the largest degradation."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.degradation).group_name


@dataclass
class MatchReport:
    """Per-match summary of contextual metric contributions."""

    match_id: str
    n_actions: int
    top_cxg_actions: pd.DataFrame
    top_cxt_actions: pd.DataFrame
    team_summary: pd.DataFrame
    shap_available: bool


@dataclass
class PlayerReport:
    """Per-player interpretability summary."""

    player_id: str
    n_actions: int
    total_cxg: float
    total_cxa: float
    total_cxt: float
    per_90: dict[str, float]
    vs_average: dict[str, float]          # z-scores relative to league_df
    top_features: pd.DataFrame | None      # mean |SHAP| if available


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_final_estimator(model):
    """Walk a Pipeline (possibly nested) to return the terminal estimator."""
    from sklearn.pipeline import Pipeline
    while isinstance(model, Pipeline):
        model = model.steps[-1][1]
    return model


def _get_named_coefficients(model, feature_names: list[str]) -> dict[str, float]:
    """
    Return a ``{feature_name: coefficient_or_importance}`` dict from a fitted model
    or sklearn Pipeline.

    For linear models (``coef_`` attribute), signs are meaningful.
    For tree models (``feature_importances_``), values are unsigned importances.

    Parameters
    ----------
    model : fitted estimator or Pipeline
    feature_names : list[str]
        Original feature column names (used as fallback if name extraction fails).
    """
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        preprocessor = model.steps[0][1]
        final_est = model.steps[-1][1]

        # Try to recover transformed feature names from the preprocessor
        try:
            transformed_names: list[str] | None = list(
                preprocessor.get_feature_names_out()
            )
        except Exception:
            transformed_names = None

        coefs: np.ndarray | None = None
        if hasattr(final_est, "coef_"):
            coefs = np.asarray(final_est.coef_).ravel()
        elif hasattr(final_est, "feature_importances_"):
            coefs = np.asarray(final_est.feature_importances_).ravel()

        if coefs is None:
            raise ValueError(
                f"Cannot extract coefficients from {type(final_est).__name__}. "
                "Final estimator must have coef_ or feature_importances_."
            )

        if transformed_names and len(transformed_names) == len(coefs):
            result: dict[str, float] = {}
            for tname, coef in zip(transformed_names, coefs):
                # Strip ColumnTransformer prefixes (e.g. "num__", "cat__")
                bare = tname
                for prefix in ("num__", "cat__", "remainder__"):
                    if bare.startswith(prefix):
                        bare = bare[len(prefix):]
                        break
                result[bare] = float(coef)
            return result

        # Positional fallback
        return {
            fn: float(c) for fn, c in zip(feature_names[: len(coefs)], coefs)
        }

    # Non-pipeline estimator
    if hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_).ravel()
        return {fn: float(c) for fn, c in zip(feature_names[: len(coefs)], coefs)}

    if hasattr(model, "feature_importances_"):
        imps = np.asarray(model.feature_importances_).ravel()
        return {fn: float(i) for fn, i in zip(feature_names[: len(imps)], imps)}

    raise ValueError(
        f"Cannot extract coefficients from {type(model).__name__}. "
        "Model must have coef_ or feature_importances_."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def compute_shap_values(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    explainer_type: str = "auto",
    n_background: int = 50,
    random_state: int = 42,
) -> SHAPResult:
    """
    Compute SHAP values for a batch of predictions.

    Parameters
    ----------
    model : fitted estimator or Pipeline
    df : DataFrame
    feature_cols : list[str]
    explainer_type : {"auto", "tree", "linear", "kernel"}
        "auto" inspects the final estimator and picks the cheapest explainer.
    n_background : int
        Number of background samples for LinearExplainer / KernelExplainer.
    random_state : int

    Returns
    -------
    SHAPResult

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap  # noqa: F401  (triggers ImportError early)
    except ImportError as exc:
        raise ImportError(
            "shap is required for compute_shap_values. "
            "Install it with: pip install shap"
        ) from exc

    import shap  # re-import after guard

    X = df[feature_cols].copy()

    if explainer_type == "auto":
        final = _get_final_estimator(model)
        if hasattr(final, "get_booster") or hasattr(final, "booster_"):
            explainer_type = "tree"
        elif hasattr(final, "coef_"):
            explainer_type = "linear"
        else:
            explainer_type = "kernel"

    rng = np.random.default_rng(random_state)

    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # binary: use class-1 SHAP
        expected = explainer.expected_value
        if hasattr(expected, "__len__"):
            expected = float(np.mean(expected))
        base = float(expected)

    elif explainer_type == "linear":
        idx = rng.choice(len(X), size=min(n_background, len(X)), replace=False)
        background = X.iloc[idx]
        explainer = shap.LinearExplainer(model, background)
        shap_vals = explainer.shap_values(X)
        base = float(explainer.expected_value)

    else:  # kernel
        idx = rng.choice(len(X), size=min(n_background, len(X)), replace=False)
        background = X.iloc[idx]
        if hasattr(model, "predict_proba"):
            def _pred(x):
                return model.predict_proba(
                    pd.DataFrame(x, columns=feature_cols)
                )[:, 1]
        else:
            def _pred(x):
                return model.predict(pd.DataFrame(x, columns=feature_cols))
        explainer = shap.KernelExplainer(_pred, background)
        shap_vals = explainer.shap_values(X)
        base = float(explainer.expected_value)

    return SHAPResult(
        shap_values=np.asarray(shap_vals),
        base_value=base,
        feature_names=list(feature_cols),
    )


def check_coefficient_directions(
    model,
    feature_names: list[str],
    expected_signs: dict[str, int] | None = None,
) -> DirectionCheckResult:
    """
    Verify that model coefficients have the expected sign for each named feature.

    Parameters
    ----------
    model : fitted estimator or Pipeline
    feature_names : list[str]
        Original feature names, used as a lookup fallback.
    expected_signs : dict[str, int] | None
        Mapping ``{feature: +1 | -1}``. Defaults to ``EXPECTED_SIGNS_CXG``.

    Returns
    -------
    DirectionCheckResult

    Raises
    ------
    ValueError
        If coefficients cannot be extracted from the model.
    """
    if expected_signs is None:
        expected_signs = EXPECTED_SIGNS_CXG

    coef_map = _get_named_coefficients(model, feature_names)

    violations: list[DirectionViolation] = []
    n_checked = 0

    for feat, expected in expected_signs.items():
        # Resolve the coefficient key (may have ColumnTransformer prefix stripped)
        matched_key: str | None = None
        if feat in coef_map:
            matched_key = feat
        else:
            for key in coef_map:
                if key == feat or key.endswith(f"__{feat}") or key.endswith(feat):
                    matched_key = key
                    break

        if matched_key is None:
            # Feature not found in model — skip (may not be in this feature set)
            continue

        n_checked += 1
        coef = coef_map[matched_key]
        actual_sign = 1 if coef >= 0 else -1
        if actual_sign != expected:
            violations.append(
                DirectionViolation(
                    feature=feat,
                    expected_sign=expected,
                    actual_sign=actual_sign,
                    coefficient=float(coef),
                )
            )

    n_violations = len(violations)
    pass_rate = (n_checked - n_violations) / n_checked if n_checked > 0 else 1.0

    return DirectionCheckResult(
        violations=violations,
        n_checked=n_checked,
        n_violations=n_violations,
        pass_rate=pass_rate,
    )


def run_ablation_study(
    model,
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_groups: dict[str, list[str]],
    metric_fn: Callable | None = None,
    feature_cols: list[str] | None = None,
) -> AblationResult:
    """
    Leave-one-group-out ablation study.

    For each feature group, replace that group's columns with their
    median (numeric) or mode (categorical), then re-predict and measure
    the change in a scalar metric.

    Parameters
    ----------
    model : fitted estimator
    df : DataFrame
        Must contain all feature and target columns.
    target_col : str
    task_type : {"classification", "regression"}
    feature_groups : dict[str, list[str]]
        Mapping of group name → column names to ablate.
    metric_fn : callable(y_true, y_pred) → float | None
        Defaults to ``log_loss`` for classification, ``mae`` for regression.
    feature_cols : list[str] | None
        If provided, only these columns are passed to ``model.predict_*``.
        Required when using a bare estimator (not a Pipeline) that was
        trained on a feature-only matrix.

    Returns
    -------
    AblationResult
    """
    if task_type not in ("classification", "regression"):
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )

    if metric_fn is None:
        if task_type == "classification":
            metric_fn = lambda yt, yp: float(_sk_log_loss(yt, yp))
            metric_name = "log_loss"
        else:
            metric_fn = lambda yt, yp: float(_sk_mae(yt, yp))
            metric_name = "mae"
    else:
        metric_name = getattr(metric_fn, "__name__", "custom_metric")

    y_true = df[target_col].values

    def _predict(frame: pd.DataFrame):
        X = frame[feature_cols] if feature_cols is not None else frame
        if task_type == "classification":
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    y_baseline = _predict(df)
    baseline_metric = metric_fn(y_true, y_baseline)

    entries: list[AblationEntry] = []
    for group_name, features in feature_groups.items():
        existing = [f for f in features if f in df.columns]
        if not existing:
            logger.debug("Ablation group %r: no columns found in df, skipping.", group_name)
            continue

        df_ablated = df.copy()
        for col in existing:
            if df_ablated[col].dtype.kind in ("f", "i", "u"):
                df_ablated[col] = float(df_ablated[col].median())
            else:
                modes = df_ablated[col].mode()
                fill = modes.iloc[0] if not modes.empty else df_ablated[col].iloc[0]
                df_ablated[col] = fill

        y_ablated = _predict(df_ablated)

        ablated_metric = metric_fn(y_true, y_ablated)
        degradation = ablated_metric - baseline_metric
        relative = degradation / (abs(baseline_metric) + 1e-10)

        entries.append(
            AblationEntry(
                group_name=group_name,
                features_removed=existing,
                baseline_metric=float(baseline_metric),
                ablated_metric=float(ablated_metric),
                degradation=float(degradation),
                relative_degradation=float(relative),
            )
        )

    return AblationResult(
        baseline_metric=float(baseline_metric),
        entries=entries,
        metric_name=metric_name,
    )


def build_match_report(
    scored_df: pd.DataFrame,
    match_id: str | int,
    match_id_col: str = "match_id",
    player_id_col: str = "player_id",
    team_id_col: str = "team_id",
    cxg_col: str = "cxg",
    cxa_col: str = "cxa",
    cxt_col: str = "cxt",
    top_n: int = 5,
    shap_df: pd.DataFrame | None = None,
) -> MatchReport:
    """
    Build a per-match summary report.

    Parameters
    ----------
    scored_df : DataFrame
        Event-level dataframe with metric columns already populated.
    match_id : str | int
        The match identifier to report on.
    top_n : int
        Number of top actions to include per metric.
    shap_df : DataFrame | None
        Optional SHAP values DataFrame (same row index as scored_df).

    Returns
    -------
    MatchReport

    Raises
    ------
    ValueError
        If no rows match ``match_id``.
    """
    match_df = scored_df[scored_df[match_id_col] == match_id].copy()
    if match_df.empty:
        raise ValueError(f"No actions found for match_id={match_id!r}")

    # Top CxG actions
    top_cxg = pd.DataFrame()
    if cxg_col in match_df.columns:
        cols = [c for c in [player_id_col, team_id_col, cxg_col] if c in match_df.columns]
        top_cxg = match_df.nlargest(top_n, cxg_col)[cols].reset_index(drop=True)

    # Top CxT actions (exclude NaN)
    top_cxt = pd.DataFrame()
    if cxt_col in match_df.columns:
        cxt_df = match_df.dropna(subset=[cxt_col])
        cols = [c for c in [player_id_col, team_id_col, cxt_col] if c in match_df.columns]
        top_cxt = cxt_df.nlargest(top_n, cxt_col)[cols].reset_index(drop=True)

    # Team summary
    team_summary = pd.DataFrame()
    if team_id_col in match_df.columns:
        agg_cols = {
            col: "sum"
            for col in [cxg_col, cxa_col, cxt_col]
            if col in match_df.columns
        }
        if agg_cols:
            team_summary = (
                match_df.groupby(team_id_col).agg(agg_cols).reset_index()
            )

    return MatchReport(
        match_id=str(match_id),
        n_actions=len(match_df),
        top_cxg_actions=top_cxg,
        top_cxt_actions=top_cxt,
        team_summary=team_summary,
        shap_available=shap_df is not None,
    )


def build_player_report(
    scored_df: pd.DataFrame,
    player_id: str | int,
    player_id_col: str = "player_id",
    match_id_col: str = "match_id",
    cxg_col: str = "cxg",
    cxa_col: str = "cxa",
    cxt_col: str = "cxt",
    minutes_per_match: float = 90.0,
    league_df: pd.DataFrame | None = None,
    shap_df: pd.DataFrame | None = None,
) -> PlayerReport:
    """
    Build a per-player interpretability summary.

    Parameters
    ----------
    scored_df : DataFrame
        Event-level dataframe with metric columns.
    player_id : str | int
        The player to report on.
    minutes_per_match : float
        Assumed minutes per appearance (default 90).
    league_df : DataFrame | None
        Leaderboard-style DataFrame with ``cxg_per_90`` / ``cxt_per_90`` columns
        for computing z-scores against the league.
    shap_df : DataFrame | None
        DataFrame of SHAP values with a ``player_id_col`` column for per-player
        feature attribution.

    Returns
    -------
    PlayerReport

    Raises
    ------
    ValueError
        If no rows match ``player_id``.
    """
    player_df = scored_df[scored_df[player_id_col] == player_id].copy()
    if player_df.empty:
        raise ValueError(f"No actions found for player_id={player_id!r}")

    n_matches = (
        int(player_df[match_id_col].nunique()) if match_id_col in player_df.columns else 1
    )
    total_minutes = n_matches * minutes_per_match

    total_cxg = float(player_df[cxg_col].sum()) if cxg_col in player_df.columns else 0.0
    total_cxa = float(player_df[cxa_col].sum()) if cxa_col in player_df.columns else 0.0
    total_cxt = (
        float(player_df[cxt_col].sum(skipna=True)) if cxt_col in player_df.columns else 0.0
    )

    safe_min = max(total_minutes, 1e-6)
    per_90 = {
        "cxg_per_90": total_cxg / safe_min * 90,
        "cxa_per_90": total_cxa / safe_min * 90,
        "cxt_per_90": total_cxt / safe_min * 90,
    }

    # z-scores vs league
    vs_average: dict[str, float] = {}
    if league_df is not None:
        for key in ("cxg_per_90", "cxa_per_90", "cxt_per_90"):
            if key not in league_df.columns:
                continue
            vals = league_df[key].dropna().values
            if len(vals) >= 2:
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                if sigma > 0:
                    vs_average[key] = (per_90[key] - mu) / sigma

    # SHAP top features for this player
    top_features: pd.DataFrame | None = None
    if shap_df is not None and player_id_col in shap_df.columns:
        player_shap = shap_df[shap_df[player_id_col] == player_id]
        if not player_shap.empty:
            feat_cols = [c for c in player_shap.columns if c != player_id_col]
            if feat_cols:
                mean_abs = player_shap[feat_cols].abs().mean()
                top_features = (
                    mean_abs.sort_values(ascending=False)
                    .reset_index()
                    .rename(columns={"index": "feature", 0: "mean_abs_shap"})
                    .head(10)
                )

    return PlayerReport(
        player_id=str(player_id),
        n_actions=len(player_df),
        total_cxg=total_cxg,
        total_cxa=total_cxa,
        total_cxt=total_cxt,
        per_90=per_90,
        vs_average=vs_average,
        top_features=top_features,
    )


# ── HTML report ───────────────────────────────────────────────────────────────

def build_interpretability_html(
    shap_result: SHAPResult | None = None,
    direction_check: DirectionCheckResult | None = None,
    ablation_result: AblationResult | None = None,
    output_path: str | None = None,
) -> str:
    """
    Build a combined HTML interpretability report.

    Sections included (each conditional on the argument being non-None):
      1. SHAP Feature Importance — top features by mean |SHAP|
      2. Coefficient Direction Check — list of violations (or "all clear")
      3. Ablation Sensitivity — table of degradation per feature group

    Parameters
    ----------
    shap_result : SHAPResult | None
    direction_check : DirectionCheckResult | None
    ablation_result : AblationResult | None
    output_path : str | None
        If provided, writes the HTML to disk (creating parent directories).

    Returns
    -------
    str
        Full HTML document.
    """
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #333; }
    h1 { color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
    h2 { color: #16213e; margin-top: 40px; }
    table { border-collapse: collapse; width: 100%; margin-top: 14px; }
    th { background: #0f3460; color: #fff; padding: 8px 12px; text-align: left; }
    td { padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }
    tr:nth-child(even) { background: #f8f9fa; }
    .pass { color: #28a745; font-weight: 600; }
    .fail { color: #dc3545; font-weight: 600; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
             font-size: 0.85em; font-weight: 600; }
    .badge-ok { background: #d4edda; color: #155724; }
    .badge-warn { background: #fff3cd; color: #856404; }
    .badge-fail { background: #f8d7da; color: #721c24; }
    .empty { color: #888; font-style: italic; margin-top: 10px; }
    """

    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>Interpretability Report — Contextual Football Metrics</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<h1>Interpretability Report</h1>")

    # ── 1. SHAP feature importance ────────────────────────────────────────────
    parts.append("<h2>1. SHAP Feature Importance</h2>")
    if shap_result is None:
        parts.append('<p class="empty">No SHAP result provided.</p>')
    else:
        top = shap_result.top_features(n=15)
        parts.append(f"<p>Base value: <strong>{shap_result.base_value:.4f}</strong> "
                     f"| Samples: <strong>{len(shap_result.shap_values)}</strong></p>")
        parts.append("<table>")
        parts.append("<tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th></tr>")
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            parts.append(
                f"<tr><td>{rank}</td><td>{row.feature}</td>"
                f"<td>{row.mean_abs_shap:.6f}</td></tr>"
            )
        parts.append("</table>")

    # ── 2. Coefficient direction check ───────────────────────────────────────
    parts.append("<h2>2. Coefficient Direction Plausibility</h2>")
    if direction_check is None:
        parts.append('<p class="empty">No direction check provided.</p>')
    else:
        badge_cls = "badge-ok" if direction_check.passed else "badge-fail"
        verdict = "ALL CLEAR" if direction_check.passed else f"{direction_check.n_violations} VIOLATION(S)"
        parts.append(
            f"<p>Checked <strong>{direction_check.n_checked}</strong> features — "
            f'<span class="badge {badge_cls}">{verdict}</span> '
            f"(pass rate: {direction_check.pass_rate:.0%})</p>"
        )
        if direction_check.violations:
            parts.append("<table>")
            parts.append("<tr><th>Feature</th><th>Expected Sign</th><th>Actual Sign</th><th>Coefficient</th></tr>")
            for v in direction_check.violations:
                exp_str = "+" if v.expected_sign > 0 else "−"
                act_str = "+" if v.actual_sign > 0 else "−"
                parts.append(
                    f"<tr><td>{v.feature}</td>"
                    f'<td class="pass">{exp_str}</td>'
                    f'<td class="fail">{act_str}</td>'
                    f"<td>{v.coefficient:.6f}</td></tr>"
                )
            parts.append("</table>")

    # ── 3. Ablation sensitivity ───────────────────────────────────────────────
    parts.append("<h2>3. Ablation Sensitivity</h2>")
    if ablation_result is None:
        parts.append('<p class="empty">No ablation result provided.</p>')
    else:
        df_abl = ablation_result.to_dataframe()
        parts.append(
            f"<p>Metric: <strong>{ablation_result.metric_name}</strong> "
            f"| Baseline: <strong>{ablation_result.baseline_metric:.6f}</strong></p>"
        )
        if df_abl.empty:
            parts.append('<p class="empty">No feature groups evaluated.</p>')
        else:
            parts.append("<table>")
            parts.append("<tr><th>Group</th><th>Features Removed</th>"
                         "<th>Ablated Metric</th><th>Degradation</th><th>Relative</th></tr>")
            for row in df_abl.itertuples(index=False):
                rel_pct = row.relative_degradation * 100
                badge_cls = "badge-warn" if rel_pct > 2 else "badge-ok"
                parts.append(
                    f"<tr><td>{row.group}</td><td>{row.features_removed}</td>"
                    f"<td>{row.ablated_metric:.6f}</td>"
                    f"<td>{row.degradation:+.6f}</td>"
                    f'<td><span class="badge {badge_cls}">{rel_pct:+.1f}%</span></td></tr>'
                )
            parts.append("</table>")

    parts.append("</body>")
    parts.append("</html>")

    html = "\n".join(parts)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info("Interpretability report written to %s", out)

    return html
