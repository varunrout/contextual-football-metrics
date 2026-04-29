"""
Phase 10 tests — InferencePipeline.

Covers:
  • _score_cxg            — shot rows get scores, non-shot rows get NaN,
                            no action_type column → score everything,
                            multi-column predict_proba handled
  • _join_pipeline_output — join by event_id, positional fallback
  • InferencePipeline     — empty CxG/CxA/CxT, full combination
  • score()               — adds metric columns, empty df raises ValueError,
                            model failure is graceful (NaN column added)
  • repr                  — reflects loaded components
  • save / load           — roundtrip preserves models, FileNotFoundError on missing
  • from_config           — null pointers → models stay None (no crash)
  • InferencePipelineConfig — defaults
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.inference_pipeline import (
    COL_CXA,
    COL_CXG,
    COL_CXT,
    InferencePipeline,
    InferencePipelineConfig,
    _join_pipeline_output,
    _score_cxg,
)


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _FakeCxGModel:
    """Stub CxG classifier: predict_proba returns constant P(goal) = 0.1."""
    def predict_proba(self, df):
        n = len(df)
        prob = np.full(n, 0.1)
        return np.column_stack([1 - prob, prob])


class _SingleColCxGModel:
    """Single-column predict_proba output."""
    def predict_proba(self, df):
        return np.full((len(df), 1), 0.2)


class _FakeCxAPipeline:
    """Stub CxA pipeline: score() adds cxa column."""
    def score(self, df):
        out = df.copy()
        out["cxa"] = np.random.default_rng(0).uniform(0, 0.2, len(df))
        return out


class _FakeCxTPipeline:
    """Stub CxT pipeline: score() adds cxt column."""
    def score(self, df):
        out = df.copy()
        out["cxt"] = np.random.default_rng(1).uniform(-0.05, 0.15, len(df))
        return out


class _ErrorPipeline:
    """Stub that always raises on score()."""
    def score(self, df):
        raise RuntimeError("Intentional failure for testing.")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_events(n: int = 100, n_shots: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = _rng(seed)
    action_types = ["pass"] * (n - n_shots) + ["shot"] * n_shots
    rng.shuffle(action_types)
    return pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)],
        "match_id": [f"m{i % 5}" for i in range(n)],
        "player_id": [f"p{i % 10}" for i in range(n)],
        "action_type": action_types,
        "x_location": rng.uniform(20, 105, n),
        "y_location": rng.uniform(0, 68, n),
        "distance_to_goal": rng.uniform(5, 40, n),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# _score_cxg helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreCxg:
    def test_shot_rows_get_scores(self):
        df = _make_events()
        scores = _score_cxg(_FakeCxGModel(), df)
        shot_mask = df["action_type"] == "shot"
        assert scores[shot_mask].notna().all()

    def test_non_shot_rows_are_nan(self):
        df = _make_events()
        scores = _score_cxg(_FakeCxGModel(), df)
        non_shot = df["action_type"] != "shot"
        assert scores[non_shot].isna().all()

    def test_scores_in_unit_interval(self):
        df = _make_events()
        scores = _score_cxg(_FakeCxGModel(), df)
        valid = scores.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_no_action_type_col_scores_all(self):
        df = _make_events().drop(columns=["action_type"])
        scores = _score_cxg(_FakeCxGModel(), df)
        assert scores.notna().all()

    def test_single_column_proba_handled(self):
        df = _make_events()
        shot_df = df[df["action_type"] == "shot"]
        scores = _score_cxg(_SingleColCxGModel(), shot_df)
        # All rows scored (no action_type filter — entire df is shots)
        assert scores.notna().all()

    def test_empty_shot_rows_returns_all_nan(self):
        df = _make_events(n=50, n_shots=0)
        scores = _score_cxg(_FakeCxGModel(), df)
        assert scores.isna().all()

    def test_index_preserved(self):
        df = _make_events().set_index("event_id")
        scores = _score_cxg(_FakeCxGModel(), df)
        assert list(scores.index) == list(df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# _join_pipeline_output helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestJoinPipelineOutput:
    def test_join_by_event_id(self):
        df = _make_events(n=20)
        scored = df.copy()
        scored["cxa"] = np.arange(20, dtype=float)
        result = _join_pipeline_output(df, scored, "cxa", event_id_col="event_id")
        assert result.notna().all()
        assert len(result) == 20

    def test_missing_col_returns_all_nan(self):
        df = _make_events(n=10, n_shots=3)
        scored = df.copy()  # no "cxt" column
        result = _join_pipeline_output(df, scored, "cxt")
        assert result.isna().all()

    def test_positional_fallback_when_no_event_id(self):
        df = _make_events(n=10, n_shots=3).drop(columns=["event_id"])
        scored = df.copy()
        scored["cxa"] = np.ones(10) * 0.5
        result = _join_pipeline_output(df, scored, "cxa", event_id_col="event_id")
        # Falls back to index alignment
        assert len(result) == 10


# ═══════════════════════════════════════════════════════════════════════════════
# InferencePipeline.score
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferencePipelineScore:
    def test_empty_df_raises(self):
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        with pytest.raises(ValueError, match="empty"):
            pipe.score(pd.DataFrame())

    def test_no_models_returns_copy(self):
        df = _make_events()
        pipe = InferencePipeline()
        result = pipe.score(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert COL_CXG not in result.columns
        assert COL_CXA not in result.columns
        assert COL_CXT not in result.columns

    def test_cxg_column_added(self):
        df = _make_events(n=100, n_shots=25)
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        result = pipe.score(df)
        assert COL_CXG in result.columns

    def test_cxg_shot_rows_non_nan(self):
        df = _make_events(n=100, n_shots=25)
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        result = pipe.score(df)
        shot_mask = df["action_type"] == "shot"
        assert result.loc[shot_mask, COL_CXG].notna().all()

    def test_cxg_non_shot_rows_nan(self):
        df = _make_events(n=100, n_shots=25)
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        result = pipe.score(df)
        non_shot = df["action_type"] != "shot"
        assert result.loc[non_shot, COL_CXG].isna().all()

    def test_cxa_column_added(self):
        df = _make_events()
        pipe = InferencePipeline(cxa_pipeline=_FakeCxAPipeline())
        result = pipe.score(df)
        assert COL_CXA in result.columns

    def test_cxt_column_added(self):
        df = _make_events()
        pipe = InferencePipeline(cxt_pipeline=_FakeCxTPipeline())
        result = pipe.score(df)
        assert COL_CXT in result.columns

    def test_all_three_metrics_combined(self):
        df = _make_events(n=100, n_shots=20)
        pipe = InferencePipeline(
            cxg_model=_FakeCxGModel(),
            cxa_pipeline=_FakeCxAPipeline(),
            cxt_pipeline=_FakeCxTPipeline(),
        )
        result = pipe.score(df)
        assert COL_CXG in result.columns
        assert COL_CXA in result.columns
        assert COL_CXT in result.columns

    def test_original_df_not_mutated(self):
        df = _make_events()
        original_cols = set(df.columns)
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        pipe.score(df)
        assert set(df.columns) == original_cols

    def test_row_count_preserved(self):
        df = _make_events(n=80)
        pipe = InferencePipeline(
            cxg_model=_FakeCxGModel(),
            cxa_pipeline=_FakeCxAPipeline(),
        )
        result = pipe.score(df)
        assert len(result) == 80

    def test_cxg_failure_produces_nan_column(self):
        class _BrokenCxG:
            def predict_proba(self, df):
                raise ValueError("Broken model")

        df = _make_events(n=20)
        pipe = InferencePipeline(cxg_model=_BrokenCxG())
        result = pipe.score(df)
        assert COL_CXG in result.columns
        assert result[COL_CXG].isna().all()

    def test_cxa_failure_produces_nan_column(self):
        df = _make_events(n=20)
        pipe = InferencePipeline(cxa_pipeline=_ErrorPipeline())
        result = pipe.score(df)
        assert COL_CXA in result.columns
        assert result[COL_CXA].isna().all()

    def test_cxt_failure_produces_nan_column(self):
        df = _make_events(n=20)
        pipe = InferencePipeline(cxt_pipeline=_ErrorPipeline())
        result = pipe.score(df)
        assert COL_CXT in result.columns
        assert result[COL_CXT].isna().all()


# ═══════════════════════════════════════════════════════════════════════════════
# InferencePipeline repr
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferencePipelineRepr:
    def test_empty_pipeline_repr(self):
        pipe = InferencePipeline()
        assert "empty" in repr(pipe)

    def test_cxg_in_repr(self):
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        assert "cxg" in repr(pipe)

    def test_all_three_in_repr(self):
        pipe = InferencePipeline(
            cxg_model=_FakeCxGModel(),
            cxa_pipeline=_FakeCxAPipeline(),
            cxt_pipeline=_FakeCxTPipeline(),
        )
        r = repr(pipe)
        assert "cxg" in r
        assert "cxa" in r
        assert "cxt" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferencePipelineSaveLoad:
    def test_save_creates_file(self, tmp_path):
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        path = tmp_path / "pipeline.pkl"
        pipe.save(str(path))
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        pipe = InferencePipeline()
        path = tmp_path / "models" / "v1" / "pipeline.pkl"
        pipe.save(str(path))
        assert path.exists()

    def test_load_roundtrip_preserves_models(self, tmp_path):
        pipe = InferencePipeline(
            cxg_model=_FakeCxGModel(),
            cxt_pipeline=_FakeCxTPipeline(),
        )
        path = tmp_path / "pipe.pkl"
        pipe.save(str(path))
        loaded = InferencePipeline.load(str(path))
        assert isinstance(loaded, InferencePipeline)
        assert loaded.cxg_model is not None
        assert loaded.cxt_pipeline is not None
        assert loaded.cxa_pipeline is None

    def test_load_scores_after_roundtrip(self, tmp_path):
        pipe = InferencePipeline(cxg_model=_FakeCxGModel())
        path = tmp_path / "pipe.pkl"
        pipe.save(str(path))
        loaded = InferencePipeline.load(str(path))
        df = _make_events(n=50, n_shots=10)
        result = loaded.score(df)
        assert COL_CXG in result.columns

    def test_load_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            InferencePipeline.load(str(tmp_path / "nonexistent.pkl"))

    def test_load_raises_type_error_for_wrong_type(self, tmp_path):
        import pickle
        path = tmp_path / "wrong.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"not": "a pipeline"}, fh)
        with pytest.raises(TypeError):
            InferencePipeline.load(str(path))


# ═══════════════════════════════════════════════════════════════════════════════
# from_config — null production pointers
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferencePipelineFromConfig:
    def test_null_pointers_leave_models_none(self, tmp_path):
        """The real configs/models.yaml has null pointers — all models should be None."""
        import pathlib
        cfg_path = pathlib.Path(__file__).parent.parent.parent / "configs" / "models.yaml"
        if not cfg_path.exists():
            pytest.skip("configs/models.yaml not found")
        pipe = InferencePipeline.from_config(cfg_path)
        assert pipe.cxg_model is None
        assert pipe.cxa_pipeline is None
        assert pipe.cxt_pipeline is None

    def test_from_config_creates_inference_pipeline(self, tmp_path):
        import yaml
        cfg = {"production": {"cxg": None, "cxa": None, "cxt": None}}
        cfg_file = tmp_path / "models.yaml"
        cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
        pipe = InferencePipeline.from_config(cfg_file)
        assert isinstance(pipe, InferencePipeline)

    def test_from_config_loads_model_file_when_pointer_set(self, tmp_path):
        import pickle
        import yaml

        # Save a fake CxG model pickle
        model_file = tmp_path / "cxg_v1.pkl"
        with open(model_file, "wb") as fh:
            pickle.dump(_FakeCxGModel(), fh)

        cfg = {"production": {"cxg": "cxg_v1.pkl", "cxa": None, "cxt": None}}
        cfg_file = tmp_path / "models.yaml"
        cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")

        pipe = InferencePipeline.from_config(cfg_file, models_dir=tmp_path)
        assert pipe.cxg_model is not None

    def test_from_config_raises_file_not_found_for_missing_model(self, tmp_path):
        import yaml
        cfg = {"production": {"cxg": "does_not_exist.pkl", "cxa": None, "cxt": None}}
        cfg_file = tmp_path / "models.yaml"
        cfg_file.write_text(yaml.dump(cfg), encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            InferencePipeline.from_config(cfg_file, models_dir=tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# InferencePipelineConfig defaults
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferencePipelineConfig:
    def test_default_action_type_col(self):
        cfg = InferencePipelineConfig()
        assert cfg.action_type_col == "action_type"

    def test_default_shot_action_type(self):
        cfg = InferencePipelineConfig()
        assert cfg.shot_action_type == "shot"

    def test_default_event_id_col(self):
        cfg = InferencePipelineConfig()
        assert cfg.event_id_col == "event_id"

    def test_custom_shot_action_type_used_in_scoring(self):
        df = _make_events(n=50, n_shots=0)
        # Replace some rows with a custom action type
        df.loc[:9, "action_type"] = "free_kick_shot"
        cfg = InferencePipelineConfig(shot_action_type="free_kick_shot")
        pipe = InferencePipeline(cxg_model=_FakeCxGModel(), config=cfg)
        result = pipe.score(df)
        # Only free_kick_shot rows should have scores
        custom_mask = df["action_type"] == "free_kick_shot"
        assert result.loc[custom_mask, COL_CXG].notna().all()
        other_mask = ~custom_mask
        assert result.loc[other_mask, COL_CXG].isna().all()
