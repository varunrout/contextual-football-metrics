"""
Phase 8 tests — Neural CxG/CxA models that consume 360 freeze-frame data.

Covers:
  • src.models.neural primitives (encoders, heads, freeze-frame tensorisation)
  • SetTransformerCxGModel — fit, predict_proba shape/range, save/load round-trip
  • GNNPassingNetworkCxAModel — fit, predict_proba shape/range
  • Device routing — autodetect from active CPU profile + explicit override

All tests are skipped when ``torch`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # whole module needs torch


# ── Synthetic data helpers ────────────────────────────────────────────────────


def _synthetic_shots(n: int = 60, n_matches: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(70, 105, n)
    y = rng.uniform(15, 53, n)
    dist = np.hypot(105 - x, 34 - y)
    p = 1.0 / (1.0 + np.exp(0.18 * dist - 4.0))
    goal = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(
        {
            "event_internal_id": [f"e{i}" for i in range(n)],
            "match_id": [f"m{i % n_matches}" for i in range(n)],
            "match_internal_id": [f"m{i % n_matches}" for i in range(n)],
            "x_location": x,
            "y_location": y,
            "distance_to_goal": dist,
            "shot_angle": rng.uniform(0.05, 1.4, n),
            "header": rng.choice([True, False], n),
            "volley": rng.choice([True, False], n),
            "first_time_shot": rng.choice([True, False], n),
            "open_play": np.ones(n, dtype=bool),
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
            "opponent_keeper_shot_stopping_rating": rng.uniform(0.55, 0.85, n),
            "opponent_team_strength": rng.uniform(0.3, 0.9, n),
            "knockout_or_group": rng.choice([True, False], n),
            "set_piece_flag": np.zeros(n, dtype=bool),
            "counterpress_regain_flag": rng.choice([True, False], n),
            "body_part": rng.choice(["right_foot", "left_foot", "head"], n),
            "shot_type": rng.choice(["open_play", "free_kick"], n),
            "set_piece_type": rng.choice(["none", "free_kick"], n),
            "score_state": rng.choice(["winning", "drawing", "losing"], n),
            "home_or_away": rng.choice(["home", "away"], n),
            "sequence_type": rng.choice(["open_play", "transition"], n),
            "possession_start_zone": rng.choice(["defensive", "middle", "attacking"], n),
            "transition_or_settled": rng.choice(["transition", "settled"], n),
            "goal": goal,
        }
    )


def _synthetic_frames(
    shots: pd.DataFrame, players_per_event: int = 14, seed: int = 1
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for eid, sx, sy in zip(
        shots["event_internal_id"], shots["x_location"], shots["y_location"], strict=False
    ):
        for k in range(players_per_event):
            teammate = bool(k % 2 == 0)
            keeper = k == players_per_event - 1  # last player is keeper
            if keeper:
                x = 104.0 + rng.uniform(-1, 1)
                y = 34.0 + rng.uniform(-2, 2)
            else:
                x = float(np.clip(sx + rng.normal(0, 6), 0, 105))
                y = float(np.clip(sy + rng.normal(0, 6), 0, 68))
            rows.append(
                {
                    "event_internal_id": eid,
                    "x": x,
                    "y": y,
                    "teammate": teammate,
                    "keeper": keeper,
                }
            )
    return pd.DataFrame(rows)


# ── Freeze-frame tensorisation primitives ────────────────────────────────────


class TestFreezeFrameLoader:
    def test_encode_frame_tokens_shape_and_mask(self):
        from src.models.neural import TOKEN_DIM, encode_frame_tokens

        shots = _synthetic_shots(n=8)
        frames = _synthetic_frames(shots, players_per_event=10)
        tokens, mask = encode_frame_tokens(shots, frames, max_players=22)

        assert tokens.shape == (8, 22, TOKEN_DIM)
        assert mask.shape == (8, 22)
        # First 10 slots populated, remaining masked
        assert (~mask[:, :10]).all()
        assert mask[:, 10:].all()

    def test_encode_handles_missing_frames(self):
        from src.models.neural import encode_frame_tokens

        shots = _synthetic_shots(n=4)
        empty = pd.DataFrame(columns=["event_internal_id", "x", "y", "teammate", "keeper"])
        tokens, mask = encode_frame_tokens(shots, empty, max_players=22)
        assert tokens.shape == (4, 22, 9)
        assert mask.all(), "all positions should be padded when no frames available"

    def test_knn_adjacency_excludes_padding_and_opponents(self):
        from src.models.neural import build_knn_adjacency, encode_frame_tokens

        shots = _synthetic_shots(n=2)
        frames = _synthetic_frames(shots, players_per_event=8)
        tokens, mask = encode_frame_tokens(shots, frames, max_players=12)
        adj = build_knn_adjacency(tokens, mask, k=3)
        # Padded columns must always be masked-out
        assert adj[:, :, 8:].all(), "padded columns should never be edge endpoints"
        # No self-loops
        for b in range(adj.shape[0]):
            assert adj[b].diagonal().all()


# ── SetTransformerCxGModel ────────────────────────────────────────────────────


class TestSetTransformerCxG:
    @pytest.fixture
    def trained_model(self, tmp_path):
        from src.models.cxg.set_transformer_model import SetTransformerCxGModel

        shots = _synthetic_shots(n=80, n_matches=8)
        frames = _synthetic_frames(shots, players_per_event=14)
        # Force the model to use our synthetic frames instead of disk
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)

        model = SetTransformerCxGModel(
            feature_set="contextual",
            frames_path=frames_path,
            d_model=16,
            n_heads=2,
            n_layers=1,
            mlp_hidden=16,
            max_epochs=2,
            batch_size=16,
            random_state=0,
            device="cpu",
        )
        model.fit(shots, target_col="goal")
        return model, shots, frames_path

    def test_predict_proba_shape_and_range(self, trained_model):
        model, shots, _ = trained_model
        p = model.predict_proba(shots.head(20))
        assert p.shape == (20,)
        assert (p >= 0).all() and (p <= 1).all()

    def test_save_load_round_trip(self, trained_model, tmp_path):
        from src.models.cxg.set_transformer_model import SetTransformerCxGModel

        model, shots, _ = trained_model
        path = tmp_path / "set_tr.joblib"
        model.save(path)
        reloaded = SetTransformerCxGModel.load(path)
        p1 = model.predict_proba(shots.head(10))
        p2 = reloaded.predict_proba(shots.head(10))
        np.testing.assert_allclose(p1, p2, atol=1e-5)

    def test_device_override(self, tmp_path):
        from src.models.cxg.set_transformer_model import SetTransformerCxGModel

        shots = _synthetic_shots(n=20)
        frames = _synthetic_frames(shots, players_per_event=8)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)
        m = SetTransformerCxGModel(
            frames_path=frames_path,
            device="cpu",
            d_model=8,
            n_heads=2,
            n_layers=1,
            mlp_hidden=8,
            max_epochs=1,
            batch_size=8,
        )
        m.fit(shots, target_col="goal")
        assert m._torch_device() == "cpu"


# ── GNNPassingNetworkCxAModel ─────────────────────────────────────────────────


def _synthetic_actions(shots: pd.DataFrame) -> pd.DataFrame:
    """Re-purpose synthetic shot rows as 'actions' with a shot_created target."""
    df = shots.copy()
    df["shot_created"] = (df["distance_to_goal"] < 25).astype(int)
    return df


class TestGNNPassingNetworkCxA:
    def test_fit_and_predict(self, tmp_path):
        from src.models.cxa.gnn_passing_network import GNNPassingNetworkCxAModel

        shots = _synthetic_shots(n=60)
        actions = _synthetic_actions(shots)
        frames = _synthetic_frames(shots, players_per_event=12)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)

        # CxA contextual feature set is a different schema; provide minimal
        # numeric columns the spec expects to find.
        for col in ["pressure_at_action", "x_progression", "danger_zone_entry"]:
            actions[col] = 0.0

        model = GNNPassingNetworkCxAModel(
            feature_set="contextual",
            frames_path=frames_path,
            d_model=16,
            n_heads=2,
            n_layers=1,
            mlp_hidden=16,
            max_epochs=2,
            batch_size=16,
            k_neighbors=3,
            device="cpu",
            random_state=0,
        )
        model.fit(actions, target_col="shot_created")
        p = model.predict_proba(actions.head(15))
        assert p.shape == (15,)
        assert (p >= 0).all() and (p <= 1).all()


# ── Device routing on existing NN models ─────────────────────────────────────


class TestDeviceRouting:
    def test_resolve_device_explicit_wins(self):
        from src.models.neural import resolve_device

        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda:1") == "cuda:1"

    def test_ffnn_state_value_accepts_device(self):
        from src.models.cxt.state_value_model import FFNNStateValueModel

        m = FFNNStateValueModel(device="cpu", batch_size=8, max_epochs=1)
        assert m.device == "cpu"

    def test_mlp_shot_quality_accepts_device(self):
        from src.models.cxa.shot_quality_model import MLPShotQualityModel

        m = MLPShotQualityModel(device="cpu", batch_size=8, max_epochs=1)
        assert m.device == "cpu"

    def test_transformer_shot_creation_accepts_device(self):
        from src.models.cxa.shot_creation_model import TransformerShotCreationModel

        m = TransformerShotCreationModel(device="cpu", batch_size=8, max_epochs=1)
        assert m.device == "cpu"


# ── Ladder integration ───────────────────────────────────────────────────────


class TestLadderIntegration:
    def test_ladder_includes_neural_when_flag_set(self, tmp_path):
        from src.models.cxg.ladder import CxGLadder

        shots = _synthetic_shots(n=80, n_matches=10)
        frames = _synthetic_frames(shots, players_per_event=12)
        frames_path = tmp_path / "frames.parquet"
        frames.to_parquet(frames_path)

        ladder = CxGLadder()
        results = ladder.run(
            shots,
            target_col="goal",
            match_id_col="match_id",
            n_folds=2,
            include_360=False,
            include_neural=True,
            frames_path=str(frames_path),
            n_estimators=20,
            random_state=0,
        )
        names = [r.name for r in results]
        assert "set_transformer_360" in names
