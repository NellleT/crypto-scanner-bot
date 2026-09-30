"""Guards on the scheduled GitHub workflow.

Paper trading must stay paper: these fail if DRY_RUN stops being hard-coded,
if the run could be switched live from the Actions UI, or if a delivery secret
is ever exposed to the job.
"""

from __future__ import annotations

import re

import pytest

from scanner.config import PROJECT_ROOT

yaml = pytest.importorskip("yaml")
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "bot.yml"


def load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def triggers(doc: dict) -> dict:
    return doc.get("on") or doc.get(True)   # YAML 1.1 reads a bare `on` as True


def test_runs_once_a_day_just_after_the_binance_daily_close() -> None:
    assert triggers(load())["schedule"] == [{"cron": "5 0 * * *"}]


def test_dry_run_is_hard_coded_and_cannot_be_switched_off() -> None:
    doc = load()
    job = doc["jobs"]["paper"]
    assert job["env"]["DRY_RUN"] == "true"
    dispatch = triggers(doc).get("workflow_dispatch") or {}
    assert not dispatch.get("inputs"), "a manual input could turn dry-run off"
    for step in job["steps"]:
        assert "DRY_RUN" not in (step.get("env") or {}), "a step must not override DRY_RUN"
    (run,) = [s["run"] for s in job["steps"] if "main.py" in str(s.get("run", ""))]
    assert "--paper" in run and "--dry-run" in run


def test_no_delivery_secret_reaches_the_job() -> None:
    assert "secrets." not in WORKFLOW.read_text(encoding="utf-8")


def test_uses_binance_data_and_a_fixed_start() -> None:
    (step,) = [s for s in load()["jobs"]["paper"]["steps"] if "main.py" in str(s.get("run", ""))]
    env = step["env"]
    assert env["EXCHANGE_ID"] == "binance"
    assert env["MARKET_DATA_URL"].startswith("https://data-api.binance.vision/")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", env["PAPER_START"])
