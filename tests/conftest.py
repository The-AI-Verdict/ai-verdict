from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core import db


@pytest.fixture
def db_path(tmp_path) -> Path:
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


@pytest.fixture
def seeded_experiment(db_path):
    """A minimal experiment with one prompt variant and one model, ready for
    runs to be planned."""
    with db.get_connection(db_path) as conn:
        exp_id = db.create_experiment(
            conn, name="Test Experiment", hypothesis="h", description="d",
            repetitions=2, randomize_order=False, temperature=0.7,
        )
        pv_id = db.create_prompt_version(conn, exp_id, "baseline", "Describe the scene.")
        db.add_experiment_model(conn, exp_id, "openai/gpt-4o-mini", provider="openai")
    return {"experiment_id": exp_id, "prompt_version_id": pv_id, "model_id": "openai/gpt-4o-mini"}
