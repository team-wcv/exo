"""Check that availability reconciliation preserves a resident service model."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "studio_guard", Path(__file__).with_name("studio_guard.py")
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_resident_identifier_is_not_loaded_again():
    config = {
        "lms": "lms",
        "health_url": "http://127.0.0.1:1234/v1/models",
        "identifier": "laguna-studio",
    }
    calls = []

    def command(_lms, *args):
        calls.append(args)
        return json.dumps([{"identifier": "laguna-studio"}]) if args[0] == "ps" else ""

    with (
        patch.object(guard, "command", command),
        patch.object(guard.urllib.request, "urlopen") as response,
    ):
        response.return_value.__enter__.return_value.status = 200
        assert guard.reconcile(config) == "ready"
    assert calls == [("daemon", "up"), ("ps", "--json")]


def test_bad_status_output_is_not_treated_as_permission_to_reload():
    with (
        patch.object(guard, "command", return_value='{"unexpected": []}'),
        pytest.raises(RuntimeError, match="list of loaded models"),
    ):
        guard.loaded("lms", "laguna-studio")
