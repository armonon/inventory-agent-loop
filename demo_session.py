"""Replay a scripted conversation through the REAL agent loop.

Why this exists: it lets you (or a reviewer) verify the whole
Observe → Think → Act → Update → Repeat cycle, the API wiring, and the
conversation_log.csv format **without a Groq API key**.

Everything here is real except the model's choices:
  - the loop is the actual `run_turn` from agent.py
  - the tool calls really hit the running FastAPI service
  - products.csv is really mutated
  - conversation_log.csv is really appended to
Only the LLM's tool-selection decisions are pre-scripted.

Usage (API must be running):
    uvicorn api.app:app --reload        # terminal 1
    python demo_session.py              # terminal 2

For the real thing, put a key in .env and run `python agent.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import agent


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _Fn
    type: str = "function"


@dataclass
class _Msg:
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Completion:
    choices: list[_Choice]


@dataclass
class _Completions:
    script: list[_Msg]
    i: int = 0

    def create(self, **_: Any) -> _Completion:
        msg = self.script[self.i]
        self.i += 1
        return _Completion(choices=[_Choice(message=msg)])


@dataclass
class ScriptedClient:
    chat: Any = field(default=None)

    @staticmethod
    def of(script: list[_Msg]) -> "ScriptedClient":
        completions = _Completions(script=script)
        return ScriptedClient(chat=type("C", (), {"completions": completions})())


def tool(name: str, args: dict[str, Any], cid: str) -> _Msg:
    return _Msg(content=None, tool_calls=[_ToolCall(cid, _Fn(name, json.dumps(args)))])


def say(text: str) -> _Msg:
    return _Msg(content=text, tool_calls=None)


def run_session(title: str, exchanges: list[tuple[str, list[_Msg]]]) -> None:
    """One session = one fresh message history, appended to the same log."""
    print(f"\n=== {title} ===")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": agent.SYSTEM_PROMPT}
    ]
    for user_text, script in exchanges:
        client = ScriptedClient.of(script)
        print(f"you   > {user_text}")
        result = agent.run_turn(client, messages, user_text)
        print(f"agent > {result.reply}   [{result.steps} loop steps]")


def main() -> None:
    # Resolve real ids from the live API so the demo isn't hardcoded wrong.
    inventory = agent.tool_list_inventory()
    if not inventory.get("ok"):
        raise SystemExit(
            f"API not reachable: {inventory.get('error')}\n"
            "Start it first:  uvicorn api.app:app --reload"
        )
    ids = {p["name"]: p["id"] for p in inventory["data"]}

    run_session(
        "Session 1 — Monday morning delivery + sales",
        [
            (
                "we just received 30 units of oat milk",
                [
                    tool("find_product_by_name", {"name": "oat milk"}, "s1a"),
                    tool("update_stock", {"product_id": ids["Oat milk"], "delta": 30}, "s1b"),
                    say("Logged the delivery — oat milk is now 38 liters."),
                ],
            ),
            (
                "we sold 12 bags of arabica today",
                [
                    tool("find_product_by_name", {"name": "arabica beans"}, "s1c"),
                    tool(
                        "update_stock",
                        {"product_id": ids["Arabica beans"], "delta": -12},
                        "s1d",
                    ),
                    say("Recorded 12 kg of arabica out — 33 kg left."),
                ],
            ),
        ],
    )

    run_session(
        "Session 2 — new product, then a stock check (multi-step)",
        [
            (
                "start stocking almond milk, we got 4 liters in, then tell me what's running low",
                [
                    tool(
                        "add_product",
                        {"name": "Almond milk", "quantity": 4, "unit": "liters"},
                        "s2a",
                    ),
                    tool("check_low_stock", {"threshold": 10}, "s2b"),
                    say(
                        "Added almond milk at 4 liters. Running low right now: "
                        "vanilla syrup (6 liters), napkins (5 packs) and the new "
                        "almond milk (4 liters)."
                    ),
                ],
            ),
        ],
    )

    print(f"\nAppended to {agent.LOG_PATH.name}. Inspect it with:  column -s, -t {agent.LOG_PATH.name}")


if __name__ == "__main__":
    main()
