"""
Phase 8 tests — Neural CxT state-value models that consume 360 freeze-frame data.

Covers what tests/models/test_neural_phase8.py left out:
  • GNNStateValueModel — fit, predict shape/non-negativity, save/load round-trip
  • SetTransformerStateValueModel — fit, predict shape/non-negativity, save/load round-trip
  • Device routing for both
  • StateValueLadder — include_neural + frames_path picks up both candidates

All tests are skipped when ``torch`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # whole module needs torch


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _synthetic_actions(n: int = 60, n_matches: int = 6, seed: int = 0) -> pd.DataFrame:
    """CxT-eligible actions with a positive, continuous possession_cxg target."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(20, 105, n)
    y = rng.uniform(0, 68, n)
    dist = np.hypot(105 - x, 34 - y)
    # Monotone-decreasing-in-distance, strictly positive target.
    possession_cxg = np.clip(0.35 - 0.003 * dist + rng.normal(0, 0.02, n), 0.005, None)
    return pd.DataFrame({
        "event_internal_id": [f"e{i}" for i in range(n)],
        "match_id": [f"m{i % n_matches}" for i in range(n)],
        "match_internal_id": [f"m{i % n_matches}" for i in range(n)],
        "x_location": x,
        "y_location": y,
        "distance_to_goal": dist,
        "in_box": (dist < 18).astype(bool),
        "under_pressure": rng.choice([True, False], n),
        "minute": rng.integers(0, 95, n),
        "score_differential": rng.integers(-2, 3, n),
        "events_before_action": rng.integers(0, 12, n),
        "passes_before_action": rng.integers(0, 8, n),
        "carries_before_action": rng.integers(0, 5, n),
        "time_from_possession_start": rng.uniform(0, 30, n),
        "vertical_progression_speed": rng.uniform(0, 5, n),
        "directness": rng.uniform(0, 1, n),
        "opponent_xg_conceded_rolling_5": rng.uniform(0.5, 2.5, n),
        "opponent_shots_conceded_rolling_5": rng.uniform(5, 20, n),
        "opponent_defensive_rating": rng.uniform(0.3, 0.9, n),
        "opponent_team_strength": rng.uniform(0.3, 0.9, n),
        "knockout_or_group": rng.choice([True, False], n),
        "set_piece_flag": np.zeros(n, dtype=bool),
        "counterpress_regain_flag": rng.choice([True, False], n),
        "score_state": rng.choice(["winning", "drawing", "losing"], n),
        "home_or_away": rng.choice(["home", "away"], n),
        "sequence_type": rng.choice(["open_play", "transition"], n),
        "possession_start_zone": rng.choice(["defensive", "middle", "attacking"], n),
        "transition_or_settled": rng.choice(["transition", "settled"], n),
        "possession_cxg": possession_cxg,
    })


def _synthetic_frames(actions: pd.DataFrame, players_per_event: int = 14, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for eid, ax, ay in zip(actions["event_internal_id"], actions["x_location"], actions["y_location"]):
        for k in range(players_per_event):
            teammate = bool(k % 2 == 0)
            keeper = (k == players_per_event - 1)
            if keeper:
                x = 104.0 + rng.uniform(-1, 1)
                y = 34.0 + rng.uniform(-2, 2)
            else:
                x = float(np.clip(ax + rng.normal(0, 6), 0, 105))
                y = float(np.clip(ay + rng.normal(0, 6), 0, 68))
            rows.append({
                "event_internal_id": eid,
                "x": x, "y": y,
                "teammate": teammate,
                "keeper": keeper,
            })
    return pd.DataFrame(rows)


# ── GNNStateValueModel ────────────────────────────────────────────────────────

class TestGNNStateValue:
    @pytest.fixture
    def trained_model(self, tmp_path):
        from src.models.cxt.state_value_gnn import GNNStateValueModel

        actions = _synthetic_actions(n=80, n_matches=8)
        frames = _synthetic_frames(actions, players_per_event=14)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)

        model = GNNStateValueModel(
            feature_set="contextual",
            frames_path=frames_path,
            d_model=16, n_heads=2, n_layers=1, mlp_hidden=16,
            max_epochs=2, batch_size=16, k_neighbors=3,
            device="cpu", random_state=0,
        )
        model.fit(actions, target_col="possession_cxg")
        return model, actions

    def test_predict_shape_and_nonnegative(self, trained_model):
        model, actions = trained_model
        p = model.predict(actions.head(20))
        assert p.shape == (20,)
        assert (p >= 0).all()

    def test_save_load_round_trip(self, trained_model, tmp_path):
        from src.models.cxt.state_value_gnn import GNNStateValueModel

        model, actions = trained_model
        path = tmp_path / "gnn_state_value.joblib"
        model.save(path)
        reloaded = GNNStateValueModel.load(path)
        p1 = model.predict(actions.head(10))
        p2 = reloaded.predict(actions.head(10))
        np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_device_override(self, tmp_path):
        from src.models.cxt.state_value_gnn import GNNStateValueModel

        actions = _synthetic_actions(n=20)
        frames = _synthetic_frames(actions, players_per_event=8)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)
        m = GNNStateValueModel(
            frames_path=frames_path,
            device="cpu",
            d_model=8, n_heads=2, n_layers=1, mlp_hidden=8,
            max_epochs=1, batch_size=8, k_neighbors=3,
        )
        m.fit(actions, target_col="possession_cxg")
        assert m._torch_device() == "cpu"


# ── SetTransformerStateValueModel ─────────────────────────────────────────────

class TestSetTransformerStateValue:
    @pytest.fixture
    def trained_model(self, tmp_path):
        from src.models.cxt.state_value_set_transformer import SetTransformerStateValueModel

        actions = _synthetic_actions(n=80, n_matches=8)
        frames = _synthetic_frames(actions, players_per_event=14)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)

        model = SetTransformerStateValueModel(
            feature_set="contextual",
            frames_path=frames_path,
            d_model=16, n_heads=2, n_layers=1, mlp_hidden=16,
            max_epochs=2, batch_size=16,
            device="cpu", random_state=0,
        )
        model.fit(actions, target_col="possession_cxg")
        return model, actions

    def test_predict_shape_and_nonnegative(self, trained_model):
        model, actions = trained_model
        p = model.predict(actions.head(20))
        assert p.shape == (20,)
        assert (p >= 0).all()

    def test_save_load_round_trip(self, trained_model, tmp_path):
        from src.models.cxt.state_value_set_transformer import SetTransformerStateValueModel

        model, actions = trained_model
        path = tmp_path / "set_tr_state_value.joblib"
        model.save(path)
        reloaded = SetTransformerStateValueModel.load(path)
        p1 = model.predict(actions.head(10))
        p2 = reloaded.predict(actions.head(10))
        np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_device_override(self, tmp_path):
        from src.models.cxt.state_value_set_transformer import SetTransformerStateValueModel

        actions = _synthetic_actions(n=20)
        frames = _synthetic_frames(actions, players_per_event=8)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)
        m = SetTransformerStateValueModel(
            frames_path=frames_path,
            device="cpu",
            d_model=8, n_heads=2, n_layers=1, mlp_hidden=8,
            max_epochs=1, batch_size=8,
        )
        m.fit(actions, target_col="possession_cxg")
        assert m._torch_device() == "cpu"


# ── StateValueLadder integration ──────────────────────────────────────────────

class TestStateValueLadderNeuralIntegration:
    def test_ladder_includes_gnn_and_set_transformer_when_frames_given(self, tmp_path):
        from src.models.cxt.state_value_model import StateValueLadder

        actions = _synthetic_actions(n=100, n_matches=10)
        frames = _synthetic_frames(actions, players_per_event=12)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)

        ladder = StateValueLadder()
        results = ladder.run(
            actions,
            target_col="possession_cxg",
            match_id_col="match_id",
            n_folds=2,
            include_neural=True,
            frames_path=str(frames_path),
            nn_max_epochs=2,
            random_state=0,
        )
        names = [r.name for r in results]
        assert "gnn_contextual" in names
        assert "set_transformer_contextual" in names
        assert "ffnn_contextual" in names
