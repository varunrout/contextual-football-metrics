"""
analysis/19_model_comparison.py
===============================
Evaluate all CxG model variants on the same shot sample and export a comparison table + chart.

Outputs:
- reports/model_comparison_cxg.csv
- reports/model_comparison_cxg.json
- reports/figures/model_comparison/01_cxg_model_metrics.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import save_fig  # noqa: E402
from src.evaluation.model_comparison import ModelComparisonSuite, ModelEntry  # noqa: E402

BARCA_BLUE = "#004D98"
BARCA_RED = "#A50044"
NEUTRAL = "#666666"

FEATURES_PATH = _ROOT / "data" / "features" / "features.parquet"
EVENTS_PATH = _ROOT / "data" / "processed" / "events.parquet"
MODELS = [
    ("baseline_logit", "baseline", _ROOT / "models" / "cxg" / "baseline_logit.joblib"),
    ("glm_contextual", "glm", _ROOT / "models" / "cxg" / "glm_contextual.joblib"),
    ("lgbm_traditional", "tree", _ROOT / "models" / "cxg" / "lgbm_traditional.joblib"),
    ("lgbm_contextual", "tree", _ROOT / "models" / "cxg" / "lgbm_contextual.joblib"),
    ("xgb_traditional", "tree", _ROOT / "models" / "cxg" / "xgb_traditional.joblib"),
    ("xgb_contextual", "tree", _ROOT / "models" / "cxg" / "xgb_contextual.joblib"),
]


def _load_shots_with_goal() -> pd.DataFrame:
    feats = pd.read_parquet(FEATURES_PATH)
    shots = feats[feats["event_type"] == "shot"].copy()

    ev = pd.read_parquet(EVENTS_PATH)[["internal_id", "shot_outcome"]].rename(
        columns={"internal_id": "event_id"}
    )
    shots = shots.merge(ev, on="event_id", how="left")
    shots["goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    return shots


def _register_models(suite: ModelComparisonSuite) -> None:
    for name, family, path in MODELS:
        if not path.exists():
            print(f"[WARN] missing model file: {path}")
            continue
        model = joblib.load(path)
        suite.add_model(
            ModelEntry(
                name=name,
                family=family,
                metric_type="cxg",
                task_type="classification",
                feature_set="contextual" if "contextual" in name else "traditional",
                model=model,
            )
        )


def _plot_metrics(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df = plot_df.sort_values("log_loss", ascending=True)

    x = np.arange(len(plot_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 1.5 * width, plot_df["log_loss"], width, label="Log Loss", color=BARCA_BLUE)
    ax.bar(x - 0.5 * width, plot_df["brier"], width, label="Brier", color=NEUTRAL)
    ax.bar(x + 0.5 * width, plot_df["ece"], width, label="ECE", color=BARCA_RED)

    roc = plot_df["roc_auc"].fillna(0.0)
    ax.bar(x + 1.5 * width, roc, width, label="ROC AUC", color="#2A9D8F", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["name"], rotation=20, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("CxG Model Comparison on Shot Sample")
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_fig("01_cxg_model_metrics", "model_comparison")


def main() -> None:
    print("Loading shots + labels...")
    shots = _load_shots_with_goal()
    print(f"  shots: {len(shots):,}")

    suite = ModelComparisonSuite(
        n_bootstrap=20,
        player_id_col="player_id",
        match_id_col="match_id",
        random_state=42,
    )
    _register_models(suite)

    print("Running model comparison...")
    report = suite.run(
        test_df=shots,
        target_map={"cxg": "goal"},
        leaderboard_value_col="cxt",
    )

    df = report.to_dataframe().sort_values("log_loss", ascending=True)
    out_csv = _ROOT / "reports" / "model_comparison_cxg.csv"
    out_json = _ROOT / "reports" / "model_comparison_cxg.json"
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)

    print(f"  [OK] {out_csv}")
    print(f"  [OK] {out_json}")

    _plot_metrics(df)
    print("  [OK] reports/figures/model_comparison/01_cxg_model_metrics.png")

    print("\nTop models by log-loss:")
    cols = ["name", "family", "log_loss", "brier", "roc_auc", "ece"]
    print(df[cols].head(6).to_string(index=False))


if __name__ == "__main__":
    main()
