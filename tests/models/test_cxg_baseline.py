from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.cxg.baseline import BaselineCxGModel, filter_shot_events


def _shots_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.uniform(80, 104, size=n)
    y = rng.uniform(15, 53, size=n)
    distance = np.hypot(105 - x, 34 - y)
    # Synthetic relationship: closer shots + headers slightly better.
    p = 1.0 / (1.0 + np.exp(0.22 * (distance - 12.0)))
    header = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    p = np.clip(p + (0.04 * header), 0.01, 0.95)
    goal = rng.binomial(1, p, size=n)

    return pd.DataFrame(
        {
            "event_type": "shot",
            "x_location": x,
            "y_location": y,
            "distance_to_goal": distance,
            "shot_angle": rng.uniform(0.05, 1.2, size=n),
            "header": header.astype(bool),
            "volley": rng.choice([True, False], size=n),
            "under_pressure": rng.choice([True, False], size=n),
            "open_play": rng.choice([True, False], size=n),
            "body_part": rng.choice(["foot", "head"], size=n),
            "shot_type": rng.choice(["none", "free_kick"], size=n),
            "set_piece_type": rng.choice(["none", "free_kick", "corner"], size=n),
            "goal": goal,
        }
    )


def test_cxg_baseline_fit_predict_range() -> None:
    df = _shots_df()
    model = BaselineCxGModel()
    model.fit(df)
    p = model.predict_proba(df)
    assert p.shape == (len(df),)
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_cxg_baseline_evaluate_metrics_present() -> None:
    df = _shots_df()
    model = BaselineCxGModel().fit(df)
    m = model.evaluate(df)
    assert m.log_loss >= 0.0
    assert m.brier >= 0.0


def test_filter_shot_events_works() -> None:
    df = _shots_df(10)
    df2 = pd.concat([df, pd.DataFrame([{"event_type": "pass"}])], ignore_index=True)
    out = filter_shot_events(df2)
    assert (out["event_type"] == "shot").all()
