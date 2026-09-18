"""Unit tests for BACR schemas."""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCHEMAS_DIR = os.path.join(PROJECT_ROOT, "bacr", "schemas")


def test_schema_files_exist():
    expected_schemas = ["controller.json", "initial.json", "probe.json", "trajectory.json"]
    for s in expected_schemas:
        path = os.path.join(SCHEMAS_DIR, s)
        assert os.path.exists(path), f"Schema {s} not found at {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "$schema" in data, f"Schema {s} missing $schema"
            assert "title" in data, f"Schema {s} missing title"


if __name__ == "__main__":
    test_schema_files_exist()
    print("Schema tests passed!")
