"""Notebook wiring checks; training is replaced with an untrained checkpoint."""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/train.ipynb"


def test_notebook_is_clean_and_code_compiles():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), cell["id"], "exec")
