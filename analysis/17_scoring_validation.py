"""
analysis/17_scoring_validation.py
===================================
Validation charts for La Liga 2020/21 out-of-sample scores.

Reads from:
  outputs/scores/scored.parquet
  data/processed/events.parquet  (player names, actual outcomes)

Produces (reports/figures/scoring/):
  01_cxg_calibration.png     — binned pred xG vs actual goal rate + 45° diagonal
  02_cxg_player_scatter.png  — player xG vs actual goals scatter w/ Spearman ρ
  03_cxg_leaderboard.png     — top-15 player xG bar chart
  04_cxa_leaderboard.png     — top-15 player CxA bar chart
  05_cxt_leaderboard.png     — top-15 player CxT bar chart
  06_cxg_cxt_correlation.png — shot-level CxG vs CxT scatter (cross-model sanity)
  07_cxa_by_sequence.png     — mean CxA by sequence type
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from analysis._utils import save_fig  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORED_PATH  = _ROOT / "outputs" / "scores" / "scored.parquet"
EVENTS_PATH  = _ROOT / "data" / "processed" / "events.parquet"
FIGURE_DIR   = "scoring"

# ── Pitch constants ───────────────────────────────────────────────────────────
_BOX_X_MIN   = 88.5
_BOX_Y_MIN   = 13.84
_BOX_Y_MAX   = 54.16

# ── Style ─────────────────────────────────────────────────────────────────────
BARCA_BLUE   = "#004D98"
BARCA_RED    = "#A50044"
NEUTRAL_GREY = "#666666"
ACCENT       = "#F5A623"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_data():
    """Load scored output and attach player names + actual outcomes."""
    scored = pd.read_parquet(SCORED_PATH)
    events = pd.read_parquet(EVENTS_PATH)

    # Player name lookup via player_internal_id
    name_map = (
        events[["player_internal_id", "player", "team_internal_id", "team"]]
        .dropna(subset=["player"])
        .drop_duplicates("player_internal_id")
        .rename(columns={"player": "player_name", "team": "team_name"})
    )
    scored = scored.merge(
        name_map[["player_internal_id", "player_name", "team_internal_id", "team_name"]],
        on="player_internal_id",
        how="left",
        suffixes=("", "_nm"),
    )
    # Resolve team_internal_id collision
    if "team_internal_id_nm" in scored.columns:
        scored = scored.drop(columns=["team_internal_id_nm"])

    # Attach shot outcomes for calibration
    outcome_map = (
        events[["internal_id", "shot_outcome"]]
        .dropna(subset=["shot_outcome"])
        .rename(columns={"internal_id": "event_id"})
    )
    shots = scored[scored["cxg"].notna()].copy()
    shots = shots.merge(outcome_map, on="event_id", how="left")
    shots["goal"] = (shots["shot_outcome"] == "Goal").astype(int)

    return scored, shots, events


def _team_name(events: pd.DataFrame, team_internal_id: str) -> str:
    row = events[events["team_internal_id"] == team_internal_id]["team"].dropna()
    return row.iloc[0] if not row.empty else team_internal_id


# ── Chart 1 — CxG Calibration ─────────────────────────────────────────────────

def plot_cxg_calibration(shots: pd.DataFrame) -> None:
    bins = [0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 1.01]
    shots = shots.copy()
    shots["bin"] = pd.cut(shots["cxg"], bins=bins, right=True)
    cal = shots.groupby("bin", observed=True).agg(
        mean_xg=("cxg", "mean"),
        actual_rate=("goal", "mean"),
        n=("goal", "count"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Diagonal perfect calibration
    ax.plot([0, 1], [0, 1], "--", color=NEUTRAL_GREY, lw=1.2, label="Perfect calibration", zorder=1)

    # Size by sample count
    sizes = (cal["n"] / cal["n"].max() * 350 + 50).values
    sc = ax.scatter(
        cal["mean_xg"], cal["actual_rate"],
        s=sizes, color=BARCA_BLUE, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.8,
    )

    # Annotate n per bin
    for _, row in cal.iterrows():
        ax.annotate(
            f"n={int(row['n'])}",
            xy=(row["mean_xg"], row["actual_rate"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color=NEUTRAL_GREY,
        )

    # Error bars (Wilson-like: just show std across bin for context)
    ax.errorbar(
        cal["mean_xg"], cal["actual_rate"],
        yerr=np.sqrt(cal["actual_rate"] * (1 - cal["actual_rate"]) / cal["n"]),
        fmt="none", ecolor=BARCA_BLUE, elinewidth=1.2, capsize=3, alpha=0.6, zorder=2,
    )

    total_xg = shots["cxg"].sum()
    total_goals = shots["goal"].sum()
    ax.set_title(
        f"CxG Calibration — La Liga 2020/21 (Barcelona, 35 matches)\n"
        f"Total xG: {total_xg:.1f}  |  Actual goals: {total_goals}  |  "
        f"Ratio: {total_goals / total_xg:.2f}",
        pad=10,
    )
    ax.set_xlabel("Mean predicted CxG")
    ax.set_ylabel("Actual goal rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)

    # Bubble size legend
    for label, n_ref in [("n=50", 50), ("n=200", 200)]:
        ax.scatter([], [], s=n_ref / cal["n"].max() * 350 + 50,
                   color=BARCA_BLUE, alpha=0.7, label=label)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    save_fig("01_cxg_calibration", FIGURE_DIR)
    print("  ✓ 01_cxg_calibration.png")


# ── Chart 2 — CxG Player Scatter ──────────────────────────────────────────────

def plot_cxg_player_scatter(shots: pd.DataFrame, events: pd.DataFrame) -> None:
    from scipy.stats import spearmanr

    player_xg = shots.groupby("player_name")["cxg"].sum().rename("xG")

    # Actual goals from events
    ev_shots = events[events["type"] == "Shot"].copy()
    ev_shots["goal"] = (ev_shots["shot_outcome"] == "Goal").astype(int)
    actual_goals = ev_shots.groupby("player")["goal"].sum().rename("actual_goals")

    compare = pd.concat([player_xg, actual_goals], axis=1).dropna()
    compare = compare[compare["xG"] >= 0.5]  # at least 0.5 xG to appear

    rho, pval = spearmanr(compare["xG"], compare["actual_goals"])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(compare["xG"], compare["actual_goals"],
               color=BARCA_BLUE, alpha=0.75, s=60, zorder=3, edgecolors="white", linewidths=0.6)

    # Diagonal
    max_val = max(compare[["xG", "actual_goals"]].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "--", color=NEUTRAL_GREY, lw=1.2, label="xG = Goals")

    # Label the top players
    top_players = compare.nlargest(8, "xG")
    for name, row in top_players.iterrows():
        short = name.split()[-1]  # surname only
        ax.annotate(short, xy=(row["xG"], row["actual_goals"]),
                    xytext=(5, 2), textcoords="offset points", fontsize=8, color=NEUTRAL_GREY)

    ax.set_title(
        f"Player xG vs Actual Goals — La Liga 2020/21 (Barcelona)\n"
        f"Spearman ρ = {rho:.3f}  (p = {pval:.3f})",
        pad=10,
    )
    ax.set_xlabel("Total CxG (predicted)")
    ax.set_ylabel("Actual goals")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig("02_cxg_player_scatter", FIGURE_DIR)
    print("  ✓ 02_cxg_player_scatter.png")


# ── Chart 3 — CxG Leaderboard ─────────────────────────────────────────────────

def plot_cxg_leaderboard(shots: pd.DataFrame, events: pd.DataFrame) -> None:
    player_cxg = shots.groupby("player_name")["cxg"].sum().sort_values(ascending=False).head(15)

    ev_shots = events[events["type"] == "Shot"].copy()
    ev_shots["goal"] = (ev_shots["shot_outcome"] == "Goal").astype(int)
    actual_goals = ev_shots.groupby("player")["goal"].sum()
    # StatsBomb xG — join via internal_id → event_id
    sb_xg = ev_shots.groupby("player")["shot_statsbomb_xg"].sum()

    df = pd.DataFrame({"CxG": player_cxg})
    df["sb_xg"] = df.index.map(sb_xg).fillna(0)
    df["actual_goals"] = df.index.map(actual_goals).fillna(0).astype(int)

    short = [n.split()[-1] if len(n.split()) > 1 else n for n in df.index]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(df))
    bar_h = 0.26

    bars_cxg = ax.barh(y + bar_h,     df["CxG"],          height=bar_h, color=BARCA_BLUE,
                       alpha=0.85, label="CxG (our model)", zorder=3)
    bars_sbxg = ax.barh(y,             df["sb_xg"],         height=bar_h, color=NEUTRAL_GREY,
                        alpha=0.75, label="StatsBomb xG (baseline)", zorder=3)
    bars_g    = ax.barh(y - bar_h,     df["actual_goals"],  height=bar_h, color=BARCA_RED,
                        alpha=0.85, label="Actual goals", zorder=3)

    for bars, vals, fmt, col in [
        (bars_cxg,  df["CxG"],         "{:.1f}", BARCA_BLUE),
        (bars_sbxg, df["sb_xg"],       "{:.1f}", NEUTRAL_GREY),
        (bars_g,    df["actual_goals"], "{:d}",  BARCA_RED),
    ]:
        for bar, v in zip(bars, vals):
            w = bar.get_width()
            ax.text(w + 0.15, bar.get_y() + bar.get_height() / 2,
                    fmt.format(int(v) if fmt == "{:d}" else v),
                    va="center", ha="left", fontsize=7.5, color=col)

    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Goals / xG")
    ax.set_title(
        "CxG Leaderboard — La Liga 2020/21 (Barcelona, top 15 by CxG)\n"
        "CxG vs StatsBomb xG vs Actual Goals",
        pad=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, df["CxG"].max() * 1.18)

    plt.tight_layout()
    save_fig("03_cxg_leaderboard", FIGURE_DIR)
    print("  ✓ 03_cxg_leaderboard.png")


# ── Chart 4 — CxA Leaderboard ─────────────────────────────────────────────────
BARCA_TEAM_ID = "2b04064b002670d8"
_CXA_MIN_ACTIONS = 50  # minimum actions to appear in leaderboard


def plot_cxa_leaderboard(scored: pd.DataFrame, events: pd.DataFrame) -> None:
    # Filter to Barcelona players and eligible CxA actions
    barca = scored[scored["team_internal_id"] == BARCA_TEAM_ID].copy()
    cxa_actions = barca[barca["cxa"] > 0]

    # xA = CxA summed on shot-building passes only.
    # StatsBomb's shot_key_pass_id points back to the last pass before a shot,
    # covering both Pass→Shot and Pass→Carry→Shot (and any other intermediate events).
    # We resolve those UUIDs to internal_ids so we can join to scored.parquet.
    shot_kp_ids = events["shot_key_pass_id"].dropna().unique()
    key_pass_internal_ids = set(
        events.loc[events["id"].isin(shot_kp_ids), "internal_id"]
    )
    xa_passes = (
        barca[barca["event_id"].isin(key_pass_internal_ids)]
        .groupby("player_name")["cxa"]
        .sum()
        .rename("xA")
    )

    agg = cxa_actions.groupby("player_name")["cxa"].agg(
        mean_cxa="mean",
        n_actions="count",
    )
    agg = agg.join(xa_passes, how="left").fillna({"xA": 0})
    agg = agg[agg["n_actions"] >= _CXA_MIN_ACTIONS]
    agg = agg.sort_values("mean_cxa", ascending=False).head(15)

    # Actual assists from raw events (StatsBomb pass_goal_assist flag)
    barca_assists = (
        events[
            (events["team_internal_id"] == BARCA_TEAM_ID)
            & (events["pass_goal_assist"] == True)
        ]
        .groupby("player")
        .size()
        .rename("actual_assists")
    )
    agg["actual_assists"] = agg.index.map(barca_assists).fillna(0).astype(int)

    short = [n.split()[-1] if len(n.split()) > 1 else n for n in agg.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel — mean CxA per action (efficiency)
    ax = axes[0]
    bars = ax.barh(np.arange(len(agg)), agg["mean_cxa"] * 100, color=BARCA_BLUE,
                   alpha=0.85, zorder=3)
    for bar, (_, row) in zip(bars, agg.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['mean_cxa'] * 100:.2f}%  (n={int(row['n_actions'])})",
                va="center", ha="left", fontsize=8, color="#444")
    ax.set_yticks(np.arange(len(agg)))
    ax.set_yticklabels(short, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CxA per action (%)")
    ax.set_title("Creativity Efficiency\n(mean CxA per action, >=50 actions)", pad=8)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f%%"))
    ax.set_xlim(0, agg["mean_cxa"].max() * 100 * 1.35)

    # Right panel — xA (shot-building passes only) vs actual assists
    ax2 = axes[1]
    y = np.arange(len(agg))
    bar_h = 0.38
    bars_xa = ax2.barh(y + bar_h / 2, agg["xA"].values, height=bar_h,
                       color=ACCENT, alpha=0.85, label="xA (CxA on shot-building passes)", zorder=3)
    bars_a  = ax2.barh(y - bar_h / 2, agg["actual_assists"].values, height=bar_h,
                       color=BARCA_RED, alpha=0.85, label="Actual assists", zorder=3)
    for bar in bars_xa:
        w = bar.get_width()
        if w > 0.02:
            ax2.text(w + 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{w:.2f}", va="center", ha="left", fontsize=8, color="#555")
    for bar in bars_a:
        w = bar.get_width()
        ax2.text(w + 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{int(w)}", va="center", ha="left", fontsize=8, color=BARCA_RED)
    ax2.set_yticks(y)
    ax2.set_yticklabels(short, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Assists")
    ax2.set_title(
        "xA vs Actual Assists\n"
        "(xA = CxA on passes preceding a shot, incl. pass\u2192carry\u2192shot)",
        pad=8,
    )
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, max(agg["xA"].max(), agg["actual_assists"].max()) * 1.25)

    fig.suptitle(
        "CxA Leaderboard — La Liga 2020/21 (Barcelona only, sorted by efficiency)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    save_fig("04_cxa_leaderboard", FIGURE_DIR)
    print("  ✓ 04_cxa_leaderboard.png")


# ── Chart 5 — CxT Leaderboard ─────────────────────────────────────────────────

def plot_cxt_leaderboard(scored: pd.DataFrame) -> None:
    cxt_df = scored[scored["cxt"].notna()].groupby("player_name")["cxt"].sum()
    top = cxt_df.sort_values(ascending=False).head(15)

    short = [n.split()[-1] if len(n.split()) > 1 else n for n in top.index]
    colors = [BARCA_BLUE if v >= 0 else BARCA_RED for v in top.values]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(np.arange(len(top)), top.values, color=colors, alpha=0.85, zorder=3)
    for bar, v in zip(bars, top.values):
        w = bar.get_width()
        ax.text(w + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", ha="left", fontsize=8, color="#555")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(short, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total CxT (Σ V(after) − V(before))")
    ax.set_title("CxT Leaderboard — La Liga 2020/21 (Barcelona, top 15 by total CxT)", pad=10)
    ax.axvline(0, color=NEUTRAL_GREY, lw=1.0, zorder=1)

    plt.tight_layout()
    save_fig("05_cxt_leaderboard", FIGURE_DIR)
    print("  ✓ 05_cxt_leaderboard.png")


# ── Chart 6 — CxG vs CxT Cross-model Sanity (possession level) ───────────────

def plot_cxg_cxt_correlation(scored: pd.DataFrame) -> None:
    """
    Possessions that ended in a shot should have higher cumulative CxT
    from the build-up actions leading to that shot.
    Aggregate per possession: max CxG (shot quality) vs sum CxT (build-up value).
    """
    from scipy.stats import spearmanr, pearsonr

    poss_col = "possession_internal_id"
    if poss_col not in scored.columns:
        print("  ⚠  possession_internal_id column missing — skipping cross-model chart.")
        return

    poss_cxg = scored[scored["cxg"].notna()].groupby(poss_col)["cxg"].sum().rename("poss_cxg")
    poss_cxt = scored[scored["cxt"].notna()].groupby(poss_col)["cxt"].sum().rename("poss_cxt")

    poss = pd.concat([poss_cxg, poss_cxt], axis=1).dropna()
    if len(poss) < 5:
        print("  ⚠  Not enough possessions with both CxG + CxT — skipping.")
        return

    rho, pval = spearmanr(poss["poss_cxg"], poss["poss_cxt"])
    r, _      = pearsonr(poss["poss_cxg"], poss["poss_cxt"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(poss["poss_cxg"], poss["poss_cxt"], gridsize=28, cmap="Blues",
              linewidths=0.3, mincnt=1)
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("Possession count", fontsize=9)
    ax.set_xlabel("Possession CxG (Σ shot xG in possession)")
    ax.set_ylabel("Possession CxT (Σ state-value Δ from build-up)")
    ax.set_title(
        f"Possession-level CxG vs CxT — La Liga 2020/21\n"
        f"Spearman ρ = {rho:.3f}  |  Pearson r = {r:.3f}  |  n = {len(poss)} possessions",
        pad=10,
    )

    plt.tight_layout()
    save_fig("06_cxg_cxt_correlation", FIGURE_DIR)
    print("  ✓ 06_cxg_cxt_correlation.png")


# ── Chart 7 — CxA by Sequence Type ───────────────────────────────────────────

def plot_cxa_by_sequence(scored: pd.DataFrame) -> None:
    df = scored[scored["cxa"] > 0].copy()
    if "sequence_type" not in df.columns or df["sequence_type"].isna().all():
        print("  ⚠  sequence_type not available — skipping chart 7.")
        return

    seq = (
        df.groupby("sequence_type", observed=True)["cxa"]
        .agg(mean_cxa="mean", total_cxa="sum", n="count")
        .sort_values("mean_cxa", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean CxA per sequence type
    ax = axes[0]
    bars = ax.barh(np.arange(len(seq)), seq["mean_cxa"], color=BARCA_BLUE, alpha=0.85, zorder=3)
    ax.set_yticks(np.arange(len(seq)))
    ax.set_yticklabels(
        [s.replace("_", " ").title() for s in seq.index],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean CxA per creative action")
    ax.set_title("Mean CxA by Sequence Type")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=8)

    # Right: total CxA volume by sequence type
    ax2 = axes[1]
    bars2 = ax2.barh(np.arange(len(seq)), seq["total_cxa"], color=ACCENT, alpha=0.85, zorder=3)
    ax2.set_yticks(np.arange(len(seq)))
    ax2.set_yticklabels(
        [s.replace("_", " ").title() for s in seq.index],
        fontsize=9,
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("Total CxA (sum across all actions)")
    ax2.set_title("Total CxA Volume by Sequence Type")
    for bar, n in zip(bars2, seq["n"]):
        w = bar.get_width()
        ax2.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"n={n}", va="center", ha="left", fontsize=8, color="#555")

    fig.suptitle("CxA by Sequence Type — La Liga 2020/21 (Barcelona)", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig("07_cxa_by_sequence", FIGURE_DIR)
    print("  ✓ 07_cxa_by_sequence.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")

    print("Loading scored.parquet and events …")
    scored, shots, events = _load_data()
    print(f"  scored: {len(scored):,} rows  |  shots: {len(shots):,}")

    print("\nGenerating charts → reports/figures/scoring/")
    plot_cxg_calibration(shots)
    plot_cxg_player_scatter(shots, events)
    plot_cxg_leaderboard(shots, events)
    plot_cxa_leaderboard(scored, events)
    plot_cxt_leaderboard(scored)
    plot_cxg_cxt_correlation(scored)
    plot_cxa_by_sequence(scored)

    print("\nDone. Saved to reports/figures/scoring/")


if __name__ == "__main__":
    main()
