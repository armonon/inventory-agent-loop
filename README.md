# Inventory Agent Loop

An applied AI portfolio project that connects a bounded, hand-written Python
tool loop to a FastAPI inventory service.

## My contribution

I implemented the inventory API, CSV storage, model tool definitions, agent
loop, conversation logging, scripted demonstration, and tests. I also changed
name lookup to a query parameter so product names containing slashes reach the
correct route. This repository builds on a 4Geeks project brief; Carla and the
coffee-shop supply store are the exercise scenario, not a claimed client.

## Development status

This is a local learning prototype. It supports inventory reads, product
creation, stock changes, low-stock checks, and multi-step tool use. It uses CSV
files and a process-local lock, not a transactional database or coordination
across multiple server processes. Authentication, user permissions, and human
approval before stock changes are not implemented.

## Demo and screenshots

A reproducible keyless demonstration is included in
[`demo_session.py`](./demo_session.py). It runs the real loop and API calls with
scripted model decisions. A live model run requires a Groq API key. No hosted
demo or screenshots are included; use the CLI and local OpenAPI explorer below.

## How it works

An AI assistant for Carla's simulated coffee-shop supply store. Two processes that
work together:

1. **`api/app.py`** — a FastAPI service that manages inventory, persisted
   to `products.csv`.
2. **`agent.py`** — an LLM agent that uses those endpoints as **tools**
   and runs a hand-coded **Observe → Think → Act → Update → Repeat** loop.

Carla types `we just received 30 units of oat milk` and the stock updates.
She types `what's running low?` and gets a direct answer.

> **No agent framework.** No LangChain, no LlamaIndex, no AutoGen. The loop
> is plain Python in [`agent.py`](./agent.py) — see `run_turn()`.

---

## Running it — you need two terminals

Complete the first-time setup below before starting either process. The API
must be running **before** the agent starts.

```bash
# Terminal 1 — start the API
uvicorn api.app:app --reload
```

```bash
# Terminal 2 — start the agent
python agent.py
```

**Stopping:** `Ctrl+C` in each terminal — stop the agent first, then the API.
`conversation_log.csv` is appended after each event; previously written events
remain available after the session ends.

**Relaunching:** run `python agent.py` again. The API does not need
restarting. The new run starts a fresh in-memory history, but
`conversation_log.csv` keeps every previous session — rows are only ever
appended.

### First-time setup

Requires Python 3.11 or newer. Clone this repository, then run these commands
from its root. Activate the same virtual environment in both terminals.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
```

Create `.env` in the repo root (copy `.env.example`):

```
GROQ_API_KEY=your_key_here
```

Get a key at <https://console.groq.com/keys>. **`.env` is git-ignored — never commit it.**

---

## What a session looks like

```
you   > we just received 30 units of oat milk
agent > Logged the delivery — oat milk is now 38 liters.

you   > we sold 12 bags of arabica today
agent > Recorded 12 kg of arabica out — 33 kg left.

you   > start stocking almond milk, we got 4 liters in, then tell me what's running low
agent > Added almond milk at 4 liters. Running low right now: vanilla syrup
        (6 liters), napkins (5 packs) and the new almond milk (4 liters).
```

The sample replies illustrate the seeded scenario; quantities vary with the
current CSV data. The scripted demo also uses preset replies, so inspect API
results or the CSV for the current stock after repeated runs.

That last one is a **multi-step** turn: the model calls `add_product`, sees
the result, then calls `check_low_stock` before answering — two tool calls in
a single user turn.

---

## The API

| Method  | Endpoint                     | Purpose                                                       |
| ------- | ---------------------------- | ------------------------------------------------------------- |
| `GET`   | `/inventory`                 | Full product list.                                            |
| `POST`  | `/inventory`                 | Add a product (`name`, `quantity`, `unit`). → `201`           |
| `PATCH` | `/inventory/{product_id}`    | Apply a `delta` — positive = incoming, negative = outgoing.   |
| `GET`   | `/inventory/alerts`          | Products below `threshold` (default `10`).                    |
| `GET`   | `/inventory/by-name?name=…`  | Resolve a name → id, so the agent never guesses ids.          |

`by-name` takes the name as a **query** parameter rather than a path
segment, so products whose names contain `/` — "Half/Half" is a real
coffee-shop item — resolve correctly instead of hitting a routing 404.

Error responses carry a descriptive message:

| Status | When                                                             |
| ------ | ---------------------------------------------------------------- |
| `404`  | Unknown product id or name.                                      |
| `409`  | Duplicate product name, or a delta that would push stock below 0. |
| `422`  | Malformed body (e.g. negative starting quantity).                 |

Interactive docs at <http://127.0.0.1:8000/docs> while the API runs.

### Storage

`products.csv` at the repo root, columns `id,name,quantity,unit`. Every
mutation rewrites the file immediately, so data survives a server restart
(verified by `test_products_persist_to_csv_and_survive_reload`). It is
seeded with a 10-product starter catalogue on first boot and never
overwritten afterwards.

---

## The agent loop

`run_turn()` in [`agent.py`](./agent.py) is the whole cycle:

```
OBSERVE   append the user's message to the history
   │
   ├─► THINK    send history + tool definitions to the LLM
   │      │
   │      ├─ no tool calls  ──►  final answer, loop ends cleanly
   │      │
   │      └─ tool calls
   │            │
   │      ACT   run each one against the FastAPI service
   │            │
   │     UPDATE append each result to the history as a `tool` message
   │            │
   └───────────┘  REPEAT (bounded by MAX_STEPS_PER_TURN = 8)
```

The message history lives in memory for the duration of a session, so the
model can reason about earlier exchanges.

### Tools exposed to the LLM

Each has a name, a description, and typed JSON-Schema parameters:

| Tool                   | Calls                          | Typed parameters                    |
| ---------------------- | ------------------------------ | ----------------------------------- |
| `list_inventory`       | `GET /inventory`               | —                                   |
| `find_product_by_name` | `GET /inventory/by-name`       | `name: string`                      |
| `add_product`          | `POST /inventory`              | `name: string, quantity: number, unit: string` |
| `update_stock`         | `PATCH /inventory/{id}`        | `product_id: integer, delta: number`|
| `check_low_stock`      | `GET /inventory/alerts`        | `threshold: number` (optional)      |

API errors are returned to the model **as data**, not raised — so when
something fails, the agent explains it to Carla instead of crashing.

---

## Conversation log

Every event appends one row to `conversation_log.csv`:

| Field       | Description                                          |
| ----------- | ---------------------------------------------------- |
| `actor`     | `user`, `agent`, or `tool`                           |
| `message`   | The text or the tool's result payload                |
| `tool_call` | Name of the tool involved (empty when not applicable)|
| `timestamp` | ISO-8601 UTC                                         |

Opened in append mode on every write — previous sessions are never
overwritten. A two-session scripted demonstration log is committed in this repository;
it is not evidence of live model decisions.

Read it nicely:

```bash
column -s, -t conversation_log.csv | less -S
```

---

## Tests

The pytest suite runs without an API key or network:

```bash
pytest
```

- **`tests/test_api.py`** — all endpoints, status codes, duplicate and
  oversell rejection, threshold behaviour, and CSV persistence across a
  simulated restart.
- **`tests/test_agent_loop.py`** — the loop driven by a scripted fake
  LLM: tool results really get injected back into context, multi-step turns,
  clean termination, error handling, the runaway guard, all four log fields
  on every event, and append-only behaviour across sessions.

## Reproducing the demo without a Groq key

[`demo_session.py`](./demo_session.py) replays a scripted conversation
through the **real** loop:

```bash
uvicorn api.app:app --reload   # terminal 1
python demo_session.py         # terminal 2
```

Everything is real — the loop, the HTTP calls, the CSV writes. Only the
model's tool-selection decisions are pre-scripted. It exists so the whole
pipeline can be verified end to end without credentials. For the genuine
article, add your key to `.env` and run `python agent.py`.

---

## Layout

```
api/
├── app.py          FastAPI routes + schemas
└── storage.py      CSV persistence (separated so it's unit-testable)
agent.py            tool definitions + the hand-coded agent loop + CLI
demo_session.py     keyless end-to-end replay of the loop
products.csv        inventory data (seeded, then mutated at runtime)
conversation_log.csv append-only event log
tests/              API and agent-loop tests
```

