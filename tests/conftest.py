from pathlib import Path

import pytest

import demux_realm.steps as realm_steps

#: The real waif runner, for tests that exercise the BioMate CLI.
REAL_RUN_WAIF = realm_steps._run_waif


class FakeWaif:
    """Stands in for the biomate CLI: records each input and returns rows."""

    def __init__(self):
        self.rows: list[dict[str, str]] = []
        self.inputs: list[dict[str, str]] = []

    def __call__(self, input_dir: Path) -> list[dict[str, str]]:
        # The temporary input is gone once the step returns, so keep a copy.
        self.inputs.append(
            {
                str(f.relative_to(input_dir)): f.read_text()
                for f in sorted(input_dir.rglob("*.csv"))
            }
        )
        return self.rows


@pytest.fixture(autouse=True)
def fake_waif(monkeypatch):
    """Keep lanes with several settings from needing a real biomate CLI."""
    fake = FakeWaif()
    monkeypatch.setattr(realm_steps, "_run_waif", fake)
    return fake
