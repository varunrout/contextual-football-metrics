"""
Reconciliation test for the CxG production pointer (CONT-F03).

The production model is picked by one rule: rank 1 on 5-fold CV log-loss. Two
places must agree on which model that is, and they used to disagree:

  * configs/models.yaml -> production.cxg
  * docs/modeling/cxg/06_results_and_discussion.md -> "Production model: ..."

This test asserts they name the same model, and that the config paths are POSIX
(so scoring resolves on Linux and Windows alike, CONT-F11).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_YAML = _ROOT / "configs" / "models.yaml"
_CXG_RESULTS_DOC = _ROOT / "docs" / "modeling" / "cxg" / "06_results_and_discussion.md"


def _production() -> dict:
    cfg = yaml.safe_load(_MODELS_YAML.read_text(encoding="utf-8"))
    prod = cfg.get("production")
    assert isinstance(prod, dict), "configs/models.yaml has no production block"
    return prod


def test_production_paths_are_posix():
    """No Windows backslash separators in any production pointer."""
    for metric, path in _production().items():
        assert "\\" not in path, f"production.{metric} uses a backslash path: {path!r}"
        assert path.startswith("models/"), f"production.{metric} should be under models/: {path!r}"


def test_cxg_production_matches_results_doc():
    """The model named in the CxG results doc equals the config's production.cxg."""
    prod_cxg = _production()["cxg"]
    config_model = Path(prod_cxg).name  # e.g. baseline_logit.joblib

    doc = _CXG_RESULTS_DOC.read_text(encoding="utf-8")
    # Match a line like: **Production model: `models/cxg/baseline_logit.joblib`**
    m = re.search(r"Production model:\s*`([^`]+)`", doc)
    assert m, "Could not find a 'Production model: `...`' line in doc 06"
    doc_model = Path(m.group(1)).name

    assert doc_model == config_model, (
        f"Production model disagreement: doc 06 says {doc_model!r}, "
        f"configs/models.yaml says {config_model!r}. They must name the same model."
    )


def test_cxg_production_is_cv_selected_baseline():
    """The committed production CxG model is the CV rank-1 model (baseline_logit).

    This guards against silently reverting to a held-out-selected model, which is
    the contamination CONT-F02 exists to prevent.
    """
    assert Path(_production()["cxg"]).name == "baseline_logit.joblib"
