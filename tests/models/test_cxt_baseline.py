from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.cxt.baseline import ZoneXTBaseline, filter_xt_actions


def _xt_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(777)
    event_type = rng.choice(["pass", "carry", "shot"], size=n, p=[0.6, 0.25, 0.15])
    x = rng.uniform(5, 100, size=n)
    y = rng.uniform(3, 65, size=n)
    end_x = np.clip(x + rng.normal(6, 8, size=n), 0, 105)
    end_y = np.clip(y + rng.normal(0, 6, size=n), 0, 68)
    shot_prob_goal = 1.0 / (1.0 + np.exp(0.25 * (np.hypot(105 - x, 34 - y) - 10)))
    goal = (event_type == "shot") & (rng.random(n) < shot_prob_goal)

    return pd.DataFrame(
        {
            "event_type": event_type,
            "x_location": x,
            "y_location": y,
            "end_x": end_x,
            "end_y": end_y,
            "goal": goal,
        }
    )


def test_cxt_zone_baseline_fit_and_state_values() -> None:
    df = _xt_df()
    model = ZoneXTBaseline().fit(df)
    vals = model.predict_state_value(np.array([10.0, 95.0]), np.array([34.0, 34.0]))
    assert vals.shape == (2,)
    assert np.isfinite(vals).all()


def test_cxt_action_delta_shape() -> None:
    df = _xt_df()
    model = ZoneXTBaseline().fit(df)
    delta = model.predict_action_delta(df.head(20))
    assert delta.shape == (20,)


def test_filter_xt_actions_works() -> None:
    df = _xt_df(30)
    df = pd.concat([df, pd.DataFrame([{"event_type": "pressure"}])], ignore_index=True)
    out = filter_xt_actions(df)
    assert out["event_type"].isin(["pass", "carry", "shot"]).all()
