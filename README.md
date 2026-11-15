# LangGraph Agent Eval

A small multi-turn bookstore assistant built with [LangGraph](https://github.com/langchain-ai/langgraph),
plus a deterministic trajectory evaluator that scores the agent's behaviour
against a scripted test set.

> **Portfolio demo, not production.** Everything here runs offline against a
> synthetic in-memory catalog. All books, authors, and conversations are
> invented for this demonstration. There is no real user data, no paid API,
> and no production traffic.

## What it demonstrates

- **Multi-turn agent orchestration with LangGraph** — a state graph
  (`understand -> search -> respond`) with a `MemorySaver` checkpointer, so
  the agent remembers user preferences across turns in the same conversation.
- **Tool use** — the `search` node calls a catalog tool (`search_catalog`)
  and records each call on the trajectory.
- **Trajectory evaluation** — every turn is logged to JSONL (user message,
  intent, tool calls, response, remembered preferences) and scored by a
  rubric: tool-use correctness, memory use across turns, final answer
  quality.
- **Observability hooks** — LangSmith / LangFuse integration points exist
  as code, disabled by default behind an env flag (see "Remaining work").
- **Swappable LLM layer** — a deterministic stub responder runs offline by
  default; a real-LLM adapter exists as a clearly-marked scaffold.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py demo    # scripted 4-turn conversation
python main.py eval    # run the 8-conversation eval set (prints a report)
python main.py chat    # interactive REPL
pytest                 # test suite
```

No API key is needed. No network calls are made.

## How it works

### The graph (`agent.py`)

```
understand -> search -> respond
understand -> respond            (greeting / preference / chat: no tool needed)
```

- `understand` classifies the intent (preference / recommend / search /
  greeting / chat), extracts search slots (genre, keywords, price, rating),
  and updates the remembered preferences.
- `search` calls the `search_catalog` tool and records the call
  (name, arguments, result count, result titles).
- `respond` generates the reply from the intent, the tool results, and the
  remembered preferences.

State persists per `thread_id` through the checkpointer, so a preference
stated in turn 1 ("I like science fiction") shapes the recommendation in
turn 3.

### The LLM layer (`llm.py`)

- `StubResponder` — deterministic, rule-based. Intent classification and
  replies are pure functions of the input text and agent state, so eval
  scores are reproducible.
- `RealLLMAdapter` — scaffold only. It reads `LLM_API_KEY` from the
  environment, but its methods raise `NotImplementedError` until a follow-up
  wires in a real provider call. Set `USE_REAL_LLM=1` to select it
  (it will raise until implemented — that is intentional).

### Trajectory logging (`trajectory.py`)

One JSON line per turn in `trajectories/<run>.jsonl`:

```json
{"turn": 2, "thread_id": "demo", "user": "Can you recommend something?",
 "intent": "recommend",
 "tool_calls": [{"name": "search_catalog", "args": {"genre": "science fiction"},
                 "result_count": 3, "result_titles": ["Orbit of Echoes", "..."]}],
 "preferences": {"genre": "science fiction"},
 "response": "Since you like science fiction, I'd recommend ...",
 "timestamp": "2026-11-15T..."}
```

`trajectories/` and `eval_reports/` are gitignored; the logger also works
purely in-memory (no path) for tests and eval runs.

### The eval (`eval.py`, `eval_data/conversations.json`)

8 scripted multi-turn conversations with per-turn expectations. The rubric
scores three dimensions, each 0.0–1.0:

| Dimension | What it checks |
| --- | --- |
| `tool_use` | Did the agent call exactly the expected tools this turn? |
| `memory_use` | On turns that should use an earlier preference, does the reply reflect it? |
| `answer_quality` | Does the reply contain the expected titles / phrases? |

A conversation's score is the mean of the dimensions that apply to it;
the run passes if the aggregate is ≥ 0.80. Run it with `python main.py eval`.

### Tracing hooks (`tracing.py`)

`init_tracing()` returns `{"enabled": False, ...}` unless `ENABLE_TRACING=1`
is set **and** a provider key is present. `trace_turn()` is a no-op while
disabled, so nothing here can fail a demo run on tracing setup.

## Project structure

```
catalog.py                  Synthetic in-memory book catalog + search
llm.py                      Swappable LLM layer (stub default, real adapter scaffold)
agent.py                    LangGraph state graph + BookstoreAgent
trajectory.py               JSONL trajectory logging
tracing.py                  LangSmith / LangFuse hooks (disabled by default)
eval.py                     Rubric scoring + eval runner
main.py                     CLI: demo / eval / chat
eval_data/conversations.json  8 scripted multi-turn eval conversations
tests/                      pytest: catalog, graph flow, eval scoring, LLM layer
```

## Remaining work

Wiring left for a follow-up agent (deliberately not done here — this repo is
code only, no deployment, no keys, no live services):

1. **Real LLM** — implement `RealLLMAdapter.classify_intent()` and
   `.respond()` in `llm.py` against an OpenAI-compatible chat-completions
   endpoint, then set `USE_REAL_LLM=1` and `LLM_API_KEY`. Re-run the eval
   set and record the scores with the real model.
2. **Tracing** — `pip install langsmith` and/or `langfuse`, set
   `ENABLE_TRACING=1` plus the provider keys, and implement the provider
   call in `tracing.trace_turn()`. Then verify a traced run appears in the
   provider dashboard.
3. **Larger eval set** — extend `eval_data/conversations.json` beyond the
   current 8 conversations (mixed intents, ambiguous requests, preference
   changes mid-conversation) and re-baseline the threshold.
4. **CI eval gate** — run `python main.py eval` in GitHub Actions on every
   PR and fail the build below the 0.80 threshold (the sibling
   `eval-service-gcp` repo explores the hosted side of this).
5. **Deployment (optional)** — a hosted demo is out of scope for this repo;
   if wanted, wrap `BookstoreAgent` in a tiny API and deploy it separately.

## What this does not claim

- This is not production agent experience: the catalog is 12 invented books,
  the "LLM" is a rule-based stub, and there are no real users.
- It does not demonstrate production LangGraph/LangChain usage, RAG, or
  paid-model operations — those are the gaps the "Remaining work" section
  is for.
- The eval scores measure the stub's behaviour on a hand-written set, not
  general agent quality.
