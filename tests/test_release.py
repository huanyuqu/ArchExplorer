import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_source", ROOT/"tools/export_source.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def test_export_is_allowlisted_and_hashes_match(tmp_path):
    destination = tmp_path/"source"
    manifest = exporter.export(destination)
    for name, digest in manifest.items():
        assert hashlib.sha256((destination/name).read_bytes()).hexdigest() == digest
        assert not exporter.FORBIDDEN.intersection(Path(name).parts)
    assert (destination/"LICENSE").is_file()
    assert not (destination/"arch_explorer/simulator/st_env.py").exists()
    assert (destination/"arch_explorer/simulator/events/engine.py").is_file()
    assert not (destination/"run_experiment.py").exists()
    assert not (destination/".git").exists()
    with pytest.raises(ValueError, match="absent or empty"):
        exporter.export(destination)


def test_export_rejects_in_tree_destinations():
    with pytest.raises(ValueError, match="outside"):
        exporter.export(ROOT/"dist/source")
