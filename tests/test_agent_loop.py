"""Agent-loop tests.

These drive the real loop in agent.py with a scripted fake LLM, so the
Observe → Think → Act → Update → Repeat cycle is verified end to end
with no Groq key and no network. Tool calls hit the real FastAPI app
through a TestClient.
"""

from __future__ import annotations

import csv
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake LLM — mimics the OpenAI response shape the loop reads.
# ---------------------------------------------------------------------------


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]


@dataclass
class FakeCompletionsAPI:
    """Replays a scripted list of turns and records what it was sent."""

    script: list[FakeMessage]
    seen_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    calls: int = 0

    def create(self, **kwargs: Any) -> FakeCompletion:
        # Deep-ish copy so later mutation doesn't rewrite history we assert on.
        self.seen_messages.append(list(kwargs["messages"]))
        message = self.script[self.calls]
        self.calls += 1
        return FakeCompletion(choices=[FakeChoice(message=message)])


@dataclass
class FakeClient:
    chat: Any

    @staticmethod
    def with_script(script: list[FakeMessage]) -> tuple["FakeClient", FakeCompletionsAPI]:
        completions = FakeCompletionsAPI(script=script)
        chat = type("Chat", (), {"completions": completions})()
        return FakeClient(chat=chat), completions


def tool_turn(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> FakeMessage:
    return FakeMessage(
        content=None,
        tool_calls=[
            FakeToolCall(id=call_id, function=FakeFunction(name, json.dumps(arguments)))
        ],
    )


def final_turn(text: str) -> FakeMessage:
    return FakeMessage(content=text, tool_calls=None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated agent module: temp products.csv, temp log, HTTP routed
    into a TestClient instead of a live server."""
    monkeypatch.setenv("INVENTORY_CSV", str(tmp_path / "products.csv"))

    from api import app as app_module
    from api import storage

    importlib.reload(storage)
    importlib.reload(app_module)

    client = TestClient(app_module.app)
    client.__enter__()  # run startup → seed

    import agent as agent_module

    importlib.reload(agent_module)
    agent_module.LOG_PATH = tmp_path / "conversation_log.csv"

    def fake_request(method: str, url: str, **kwargs: Any):
        kwargs.pop("timeout", None)
        path = url.replace(agent_module.API_BASE_URL, "")
        return client.request(method, path, **kwargs)

    monkeypatch.setattr(agent_module.httpx, "request", fake_request)

    yield agent_module
    client.__exit__(None, None, None)


def read_log(agent_module) -> list[dict[str, str]]:
    with agent_module.LOG_PATH.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fresh_messages(agent_module) -> list[dict[str, Any]]:
    return [{"role": "system", "content": agent_module.SYSTEM_PROMPT}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loop_terminates_when_no_tool_calls(agent_env) -> None:
    agent = agent_env
    client, _ = FakeClient.with_script([final_turn("We have plenty, no worries.")])

    result = agent.run_turn(client, fresh_messages(agent), "hello")

    assert result.reply == "We have plenty, no worries."
    assert result.steps == 1


def test_tool_call_hits_api_and_result_is_injected(agent_env) -> None:
    """The core cycle: THINK → ACT → UPDATE → THINK again."""
    agent = agent_env
    client, completions = FakeClient.with_script(
        [
            tool_turn("check_low_stock", {"threshold": 10}),
            final_turn("Oat milk, vanilla syrup and napkins are running low."),
        ]
    )

    messages = fresh_messages(agent)
    result = agent.run_turn(client, messages, "what's running low?")

    assert result.steps == 2
    assert "napkins" in result.reply.lower()

    # UPDATE: the second LLM call must have seen the tool's result.
    second_call = completions.seen_messages[1]
    tool_msgs = [m for m in second_call if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["ok"] is True
    names = {p["name"] for p in payload["data"]["products"]}
    assert names == {"Oat milk", "Vanilla syrup", "Napkins"}


def test_multi_step_add_then_check_alerts(agent_env) -> None:
    """Rubric: handles at least one multi-step interaction."""
    agent = agent_env
    client, _ = FakeClient.with_script(
        [
            tool_turn(
                "add_product",
                {"name": "Almond milk", "quantity": 3, "unit": "liters"},
                call_id="c1",
            ),
            tool_turn("check_low_stock", {"threshold": 10}, call_id="c2"),
            final_turn("Added almond milk (3 liters) — it's already below the alert line."),
        ]
    )

    result = agent.run_turn(
        client,
        fresh_messages(agent),
        "add almond milk, 3 liters, then tell me what's low",
    )

    assert result.steps == 3

    # The product really was created through the API.
    from api import storage

    assert storage.find_by_name("Almond milk") is not None

    # And it shows up in the alerts the second tool call returned.
    log = read_log(agent)
    alert_rows = [r for r in log if r["actor"] == "tool" and r["tool_call"] == "check_low_stock"]
    assert len(alert_rows) == 1
    payload = json.loads(alert_rows[0]["message"])
    assert "Almond milk" in {p["name"] for p in payload["data"]["products"]}


def test_stock_update_resolves_name_then_applies_delta(agent_env) -> None:
    """The natural-language path Carla actually uses."""
    agent = agent_env
    from api import storage

    before = storage.find_by_name("Oat milk")
    assert before is not None

    client, _ = FakeClient.with_script(
        [
            tool_turn("find_product_by_name", {"name": "oat milk"}, call_id="a"),
            tool_turn("update_stock", {"product_id": before.id, "delta": 30}, call_id="b"),
            final_turn("Logged 30 liters of oat milk in — you're at 38 liters now."),
        ]
    )

    agent.run_turn(client, fresh_messages(agent), "we just received 30 units of oat milk")

    after = storage.find_by_name("Oat milk")
    assert after is not None
    assert after.quantity == before.quantity + 30


def test_api_errors_are_returned_to_the_model_not_raised(agent_env) -> None:
    agent = agent_env
    client, completions = FakeClient.with_script(
        [
            tool_turn("update_stock", {"product_id": 9999, "delta": 5}),
            final_turn("I couldn't find that product — want me to add it?"),
        ]
    )

    result = agent.run_turn(client, fresh_messages(agent), "add 5 to product 9999")

    # Loop survived and produced a final answer.
    assert "couldn't find" in result.reply
    tool_msg = [m for m in completions.seen_messages[1] if m.get("role") == "tool"][0]
    payload = json.loads(tool_msg["content"])
    assert payload["ok"] is False
    assert payload["status"] == 404


def test_log_has_all_four_fields_on_every_event(agent_env) -> None:
    agent = agent_env
    client, _ = FakeClient.with_script(
        [
            tool_turn("list_inventory", {}),
            final_turn("Here's the full list."),
        ]
    )
    agent.run_turn(client, fresh_messages(agent), "show me everything")

    rows = read_log(agent)
    # user, agent(tool call), tool(result), agent(final)
    assert len(rows) == 4
    assert [r["actor"] for r in rows] == ["user", "agent", "tool", "agent"]

    for row in rows:
        assert set(row.keys()) == {"actor", "message", "tool_call", "timestamp"}
        assert row["actor"] and row["message"] and row["timestamp"]
        # ISO-8601 must parse
        from datetime import datetime

        datetime.fromisoformat(row["timestamp"])

    # tool_call is populated exactly on the tool-related events
    assert rows[0]["tool_call"] == ""
    assert rows[1]["tool_call"] == "list_inventory"
    assert rows[2]["tool_call"] == "list_inventory"
    assert rows[3]["tool_call"] == ""


def test_log_is_append_only_across_sessions(agent_env) -> None:
    """Rubric: the log must persist and append across multiple sessions."""
    agent = agent_env

    client1, _ = FakeClient.with_script([final_turn("first session")])
    agent.run_turn(client1, fresh_messages(agent), "session one")
    after_first = read_log(agent)

    # A brand-new session = fresh message history, same log file.
    client2, _ = FakeClient.with_script([final_turn("second session")])
    agent.run_turn(client2, fresh_messages(agent), "session two")
    after_second = read_log(agent)

    assert len(after_second) == len(after_first) + 2
    # Nothing from session one was overwritten.
    assert after_second[: len(after_first)] == after_first
    assert after_second[-2]["message"] == "session two"


def test_max_steps_guard_stops_a_runaway_model(agent_env) -> None:
    agent = agent_env
    # A model that only ever asks for more tools.
    script = [tool_turn("list_inventory", {}, call_id=f"c{i}") for i in range(20)]
    client, _ = FakeClient.with_script(script)

    result = agent.run_turn(client, fresh_messages(agent), "loop forever")

    assert result.steps == agent.MAX_STEPS_PER_TURN
    assert "stuck" in result.reply


def test_every_tool_is_wired_to_an_implementation(agent_env) -> None:
    """Guards against a tool being advertised to the LLM but not runnable."""
    agent = agent_env
    advertised = {t["function"]["name"] for t in agent.TOOLS}
    implemented = set(agent.TOOL_IMPLS)
    assert advertised == implemented
