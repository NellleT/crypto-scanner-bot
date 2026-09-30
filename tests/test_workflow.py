"""Guards on the scheduled GitHub workflow.

Paper trading must stay paper: these fail if DRY_RUN stops being hard-coded,
if the run could be switched from the Actions UI, if any secret beyond the two
Telegram ones appears, if a secret leaks into job-wide scope, or if the
workflow could run on a pull request — where a fork of this public repository
could reach the secrets.
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


def test_only_the_two_telegram_secrets_are_used_and_only_per_step() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    used = set(re.findall(r"secrets\.([A-Z_]+)", text))
    assert used == {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}
    job = load()["jobs"]["paper"]
    assert not any("secrets." in str(v) for v in job["env"].values())


def test_never_runs_on_pull_requests() -> None:
    assert set(triggers(load())) <= {"schedule", "workflow_dispatch"}


def test_paper_messages_are_routed_to_telegram() -> None:
    (step,) = [s for s in load()["jobs"]["paper"]["steps"] if "main.py" in str(s.get("run", ""))]
    assert step["env"]["PAPER_TELEGRAM"] == "true"


def test_uses_binance_data_and_a_fixed_start() -> None:
    (step,) = [s for s in load()["jobs"]["paper"]["steps"] if "main.py" in str(s.get("run", ""))]
    env = step["env"]
    assert env["EXCHANGE_ID"] == "binance"
    assert env["MARKET_DATA_URL"].startswith("https://data-api.binance.vision/")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", env["PAPER_START"])
