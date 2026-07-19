from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.cxa.baseline import BaselineCxAModel, filter_pass_events


def _passes_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    progressive = rng.uniform(0, 40, size=n)
    through_ball = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    box_entry = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    p_create = 1.0 / (
        1.0 + np.exp(-(0.08 * progressive + 0.8 * through_ball + 0.7 * box_entry - 2.0))
    )
    leads = rng.binomial(1, p_create, size=n)
    quality = np.clip(
        0.05 + 0.015 * progressive + 0.12 * through_ball + rng.normal(0, 0.05, size=n), 0, 1
    )
    resulting_xg = leads * quality

    return pd.DataFrame(
        {
            "event_type": "pass",
            "x_location": rng.uniform(20, 90, size=n),
            "y_location": rng.uniform(5, 63, size=n),
            "pass_length": rng.uniform(2, 45, size=n),
            "pass_angle": rng.uniform(-3.14, 3.14, size=n),
            "progressive_distance": progressive,
            "under_pressure": rng.choice([True, False], size=n),
            "cross": rng.choice([True, False], size=n),
            "cutback": rng.choice([True, False], size=n),
            "through_ball": through_ball.astype(bool),
            "switch": rng.choice([True, False], size=n),
            "central_progression": rng.choice([True, False], size=n),
            "box_entry": box_entry.astype(bool),
            "pass_height": rng.choice(["ground", "low", "high"], size=n),
            "pass_body_part": rng.choice(["foot", "head"], size=n),
            "set_piece_type": rng.choice(["none", "corner", "free_kick"], size=n),
            "phase_of_play": rng.choice(
                ["buildup", "progression", "final_third", "transition"], size=n
            ),
            "sequence_type": rng.choice(["settled_possession", "through_ball_sequence"], size=n),
            "leads_to_shot": leads,
            "resulting_shot_xg": resulting_xg,
            "xa_target": resulting_xg,
        }
    )


def test_cxa_baseline_fit_predict() -> None:
    df = _passes_df()
    model = BaselineCxAModel().fit(df)
    xa = model.predict_xa(df)
    assert xa.shape == (len(df),)
    assert ((xa >= 0.0) & (xa <= 1.0)).all()


def test_cxa_baseline_evaluate() -> None:
    df = _passes_df()
    model = BaselineCxAModel().fit(df)
    m = model.evaluate(df)
    assert m.creation_log_loss >= 0.0
    assert m.quality_rmse >= 0.0
    assert m.total_rmse >= 0.0


def test_filter_pass_events_works() -> None:
    df = _passes_df(8)
    df2 = pd.concat([df, pd.DataFrame([{"event_type": "shot"}])], ignore_index=True)
    out = filter_pass_events(df2)
    assert (out["event_type"] == "pass").all()
