"""CLI integration tests (definition-of-done commands)."""

import json

import pytest

from polymarket.cli import main


def test_init_db(tmp_path, capsys):
    db = str(tmp_path / "empty.sqlite")
    assert main(["init-db", "--db", db]) == 0
    assert "initialized" in capsys.readouterr().out


def test_build_synthetic_audit_and_analysis(tmp_path, capsys):
    db = str(tmp_path / "synthetic.sqlite")
    assert main(["build-synthetic", "--db", db, "--overwrite"]) == 0
    assert main(["audit", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "schema version: 1" in out

    assert main(["audit", "--db", db, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1

    output = str(tmp_path / "analysis")
    assert main(["run-analysis", "--db", db, "--output", output]) == 0
    out = capsys.readouterr().out
    assert "M3" in out and "predictions" in out


def test_build_synthetic_refuses_overwrite_without_flag(tmp_path):
    db = str(tmp_path / "synthetic.sqlite")
    main(["build-synthetic", "--db", db, "--overwrite"])
    with pytest.raises(SystemExit):
        main(["build-synthetic", "--db", db])


def test_missing_db_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["audit", "--db", str(tmp_path / "missing.sqlite")])
    assert "not found" in str(excinfo.value)
