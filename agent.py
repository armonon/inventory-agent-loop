"""Inventory agent — a hand-coded Observe → Think → Act → Update → Repeat loop.

No agent framework. No LangChain, no LlamaIndex, no AutoGen. The loop
below is plain Python: we call the LLM, read back any tool calls it
asked for, execute them against the FastAPI service, push the results
into the message history, and go round again until the model answers
with no pending tool calls.

Run the API first (see README), then:

    python agent.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

LOG_PATH = Path(__file__).resolve().parent / "conversation_log.csv"
LOG_FIELDS = ("actor", "message", "tool_call", "timestamp")

# Guard against a model that keeps requesting tools forever.
MAX_STEPS_PER_TURN = 8

SYSTEM_PROMPT = """\
You are the inventory assistant for a coffee-shop supply store with two
locations. You are talking to Carla, the owner. She speaks in plain
language about deliveries and sales — never in ids or JSON.

Your job is to translate what she says into the right tool calls.

Guidance:
- "we received / got / a delivery of N X"  → stock going UP   → positive delta
- "we sold / used / went through N X"      → stock going DOWN → negative delta
- To change stock you need the product id. If Carla names a product,
  call find_product_by_name first to resolve the id, then call
  update_stock. Do not guess ids.
- If find_product_by_name reports the product does not exist, ask Carla
  whether to add it rather than inventing one — except when she has
  clearly asked you to add it, in which case call add_product.
- "what's running low / what do we need to reorder" → check_low_stock.
- When a tool returns an error, explain what went wrong in plain
  language. Never pretend an action succeeded.

Answer in one or two short sentences. Include the resulting quantity and
unit when you change stock, so Carla can confirm at a glance. No
markdown, no bullet lists — this is a terminal conversation.
"""


# ---------------------------------------------------------------------------
# Tool definitions — what the LLM is allowed to do, and with what arguments.
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": (
                "List every product in the inventory with its current quantity "
                "and unit. Use when Carla asks what is in stock, or when you "
                "need to see the available product names."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_product_by_name",
            "description": (
                "Look up a single product by its name to get its id and current "
                "stock. Call this before update_stock whenever Carla refers to a "
                "product by name instead of by id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Product name, e.g. 'Oat milk'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_product",
            "description": (
                "Add a brand-new product to the catalogue. Only use for products "
                "that do not exist yet — to change the stock of an existing "
                "product use update_stock instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Product name, e.g. 'Almond milk'.",
                    },
                    "quantity": {
                        "type": "number",
                        "description": "Starting stock level. Must be zero or more.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measure: units, kg, liters, packs.",
                    },
                },
                "required": ["name", "quantity", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_stock",
            "description": (
                "Apply a change to an existing product's stock. Use a POSITIVE "
                "delta for incoming stock (a delivery) and a NEGATIVE delta for "
                "outgoing stock (a sale or usage)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The id of the product, from find_product_by_name.",
                    },
                    "delta": {
                        "type": "number",
                        "description": (
                            "Amount to add (positive) or remove (negative). "
                            "Example: 30 for a delivery of 30, -12 for a sale of 12."
                        ),
                    },
                },
                "required": ["product_id", "delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_low_stock",
            "description": (
                "List every product whose quantity is below a threshold. Use when "
                "Carla asks what is running low or what needs reordering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "description": "Alert threshold. Defaults to 10 if omitted.",
                    }
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Act — each tool maps to exactly one HTTP call against the FastAPI service.
# ---------------------------------------------------------------------------


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Call the inventory API and normalise the outcome into a dict.

    Errors are returned as data (not raised) so the LLM can read them
    and explain the problem to Carla instead of the loop crashing.
    """
    url = f"{API_BASE_URL}{path}"
    try:
        response = httpx.request(method, url, timeout=15.0, **kwargs)
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "error": (
                f"Could not reach the inventory API at {url}: {exc}. "
                "Is it running? Start it with: uvicorn api.app:app --reload"
            ),
        }

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"ok": False, "status": response.status_code, "error": detail}

    return {"ok": True, "status": response.status_code, "data": response.json()}


def tool_list_inventory() -> dict[str, Any]:
    return _request("GET", "/inventory")


def tool_find_product_by_name(name: str) -> dict[str, Any]:
    # Sent as a query param so names containing '/' (e.g. "Half/Half")
    # survive; httpx handles the encoding.
    return _request("GET", "/inventory/by-name", params={"name": name})


def tool_add_product(name: str, quantity: float, unit: str) -> dict[str, Any]:
    return _request(
        "POST",
        "/inventory",
        json={"name": name, "quantity": quantity, "unit": unit},
    )


def tool_update_stock(product_id: int, delta: float) -> dict[str, Any]:
    return _request("PATCH", f"/inventory/{product_id}", json={"delta": delta})


def tool_check_low_stock(threshold: float = 10.0) -> dict[str, Any]:
    return _request("GET", "/inventory/alerts", params={"threshold": threshold})


TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "list_inventory": tool_list_inventory,
    "find_product_by_name": tool_find_product_by_name,
    "add_product": tool_add_product,
    "update_stock": tool_update_stock,
    "check_low_stock": tool_check_low_stock,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    try:
        return impl(**arguments)
    except TypeError as exc:
        return {"ok": False, "error": f"Bad arguments for '{name}': {exc}"}


# ---------------------------------------------------------------------------
# Conversation log — append-only, four required fields.
# ---------------------------------------------------------------------------


def log_event(actor: str, message: str, tool_call: str = "") -> None:
    """Append one row to conversation_log.csv.

    Opened in append mode every call, so previous sessions are never
    overwritten and nothing is lost if the process is killed mid-turn.
    """
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "actor": actor,
                "message": message,
                "tool_call": tool_call,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    reply: str
    steps: int


def run_turn(client: Any, messages: list[dict[str, Any]], user_input: str) -> TurnResult:
    """Run one full Observe → Think → Act → Update → Repeat cycle.

    `messages` is mutated in place — it is the session memory, so the
    model can reason about earlier exchanges in the same run.
    """
    # ---- OBSERVE: take the user's message into the context ----------------
    messages.append({"role": "user", "content": user_input})
    log_event("user", user_input)

    steps = 0
    while steps < MAX_STEPS_PER_TURN:
        steps += 1

        # ---- THINK: ask the model what to do next -------------------------
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        choice = completion.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None)

        # ---- No tool calls => final answer, loop terminates cleanly -------
        if not tool_calls:
            reply = (choice.content or "").strip()
            messages.append({"role": "assistant", "content": reply})
            log_event("agent", reply)
            return TurnResult(reply=reply, steps=steps)

        # Record the assistant's tool-call turn verbatim; the API requires
        # it to precede the matching tool result messages.
        messages.append(
            {
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # ---- ACT: run each requested tool --------------------------------
        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            log_event("agent", f"calling {name} with {json.dumps(arguments)}", name)
            result = execute_tool(name, arguments)
            payload = json.dumps(result)
            log_event("tool", payload, name)

            # ---- UPDATE: inject the result back into the context ---------
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": payload,
                }
            )
        # ---- REPEAT --------------------------------------------------------

    fallback = (
        "I got stuck taking too many steps on that one — could you rephrase it?"
    )
    messages.append({"role": "assistant", "content": fallback})
    log_event("agent", fallback)
    return TurnResult(reply=fallback, steps=steps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_client() -> Any:
    """Construct the LLM client. Imported lazily so the module can be
    imported (and tested) without the openai package configured."""
    from openai import OpenAI

    if not GROQ_API_KEY:
        print(
            "error: GROQ_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key from "
            "https://console.groq.com/keys",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)


def main() -> int:
    client = build_client()

    # Session memory — resets each run; the CSV log is what persists.
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Inventory agent ready. Talk to me about stock. Ctrl+C or 'exit' to quit.")
    print(f"API: {API_BASE_URL}   model: {MODEL}\n")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("bye.")
            return 0

        try:
            result = run_turn(client, messages, user_input)
        except Exception as exc:  # noqa: BLE001 — keep the CLI alive
            print(f"agent > something went wrong: {exc}\n")
            log_event("agent", f"ERROR: {exc}")
            continue

        print(f"agent > {result.reply}\n")


if __name__ == "__main__":
    raise SystemExit(main())
