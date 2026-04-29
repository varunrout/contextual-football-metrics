from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sequence_labeler import label_possessions_dataframe
from src.ingestion.possession_builder import build_possessions
from src.ingestion.provider_mapper import make_internal_id
from src.ingestion.schema import Provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("repair_possessions")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    events_path = root / "data" / "processed" / "events.parquet"
    out_path = root / "data" / "processed" / "possessions.parquet"

    events = pd.read_parquet(events_path)
    all_poss: list[pd.DataFrame] = []

    for match_id, grp in events.groupby("match_internal_id", sort=False):
        team_id_map: dict[int, str] = {}
        if "team_id" in grp.columns:
            for t in grp["team_id"].dropna().unique():
                ti = int(t)
                team_id_map[ti] = make_internal_id(Provider.STATSBOMB, "team", ti)

        possessions = build_possessions(grp, match_id, team_id_map)
        if not possessions:
            continue

        poss_df = pd.DataFrame([vars(p) for p in possessions])
        if "sequence_type" in poss_df.columns:
            poss_df["sequence_type"] = poss_df["sequence_type"].apply(
                lambda v: v.value if hasattr(v, "value") else str(v)
            )

        labeled = label_possessions_dataframe(poss_df, grp)
        all_poss.append(labeled)

    out = pd.concat(all_poss, ignore_index=True) if all_poss else pd.DataFrame()
    out.to_parquet(out_path, index=False)

    logger.info("Saved %s (%d rows, %d cols)", out_path, len(out), len(out.columns))
    if not out.empty and "sequence_type" in out.columns:
        seq = out["sequence_type"].astype(str)
        logger.info("sequence_type unknown rate: %.2f%%", 100 * seq.eq("unknown").mean())
        logger.info("Top sequence types:\n%s", seq.value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
