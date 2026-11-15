# LangGraph Agent Eval

A small multi-turn bookstore assistant built with [LangGraph](https://github.com/langchain-ai/langgraph),
plus a deterministic trajectory evaluator that scores the agent's behaviour
against a scripted test set.

> **Portfolio demo, not production.** By default, everything runs offline against a
> synthetic in-memory catalog. All books, authors, and conversations are
> invented for this demonstration. The default has no paid API calls or
> production traffic. Real LLM requests and LangSmith exports require explicit
> opt-in and credentials; neither has been validated against a live account.

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
- **Observability** — optional LangSmith turn-summary exports, disabled by
  default and tested against the SDK with a mocked transport.
- **Swappable LLM layer** — a deterministic stub responder runs offline by
  default; an OpenAI-compatible adapter supports classification and replies.

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

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`
instead of `source .venv/bin/activate`.

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
- `RealLLMAdapter` — calls an OpenAI-compatible `/chat/completions`
  endpoint twice per turn: JSON-mode intent/slot classification, then a reply
  using current catalog results and remembered preferences. It validates
  slot types and ranges before using them in the graph. Calls have a 30-second
  HTTP timeout; request errors and invalid output raise without falling back
  to the stub. Prompts request grounded answers, but factual accuracy is not
  guaranteed by a prompt. No live model score is claimed.

To opt in (Bash syntax; set the equivalent environment variables in PowerShell):

```bash
export USE_REAL_LLM=1
export LLM_API_KEY="<your-key>"  # environment only; never commit it
export LLM_MODEL="gpt-4o-mini"  # choose a model supported by your provider
# Optional: export LLM_BASE_URL="https://your-provider.example/v1"
python main.py demo
python main.py eval eval_reports/real.json
unset USE_REAL_LLM
python main.py eval eval_reports/stub.json
```

The endpoint must support chat completions and `response_format=json_object`.
There are no automatic retries; real requests can incur provider charges.
Every eval report records the adapter class and model so comparisons identify
which responder produced the score. The small rubric uses literal titles and
phrases, so real-model paraphrases can lose points without being incorrect.

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

### LangSmith tracing (`tracing.py`)

`init_tracing()` returns `{"enabled": False, ...}` unless `ENABLE_TRACING=1`
is set **and** `LANGSMITH_API_KEY` is present. While disabled, the hook creates
no client and sends nothing. With opt-in, it sends one completed LangSmith
`chain` run per turn, containing the user text, reply, and metadata (thread,
turn, intent, tool summary and preferences). Use synthetic inputs for this demo.

```bash
pip install "langsmith>=0.14,<1"  # also provided transitively by LangGraph
export ENABLE_TRACING=1
export LANGSMITH_API_KEY="<your-key>"
export LANGSMITH_PROJECT="langgraph-agent-eval"
python main.py demo
unset ENABLE_TRACING
```

These are synchronous best-effort turn summaries, not nested graph/LLM spans.
Start/end timestamps mark export time and do not measure model latency. Setup
errors disable tracing; upload exceptions log a sanitized warning and disable
subsequent uploads without interrupting chat. Successful client setup does not
prove an upload appeared in the dashboard. Only LangSmith is implemented;
LangFuse remains future work. Native LangChain tracing environment flags are
separate; leave them unset for an offline run.

## Project structure

```
catalog.py                  Synthetic in-memory book catalog + search
llm.py                      Offline stub + opt-in OpenAI-compatible adapter
agent.py                    LangGraph state graph + BookstoreAgent
trajectory.py               JSONL trajectory logging
tracing.py                  LangSmith turn export (disabled by default)
eval.py                     Rubric scoring + eval runner
main.py                     CLI: demo / eval / chat
eval_data/conversations.json  8 scripted multi-turn eval conversations
tests/                      pytest: catalog, graph flow, eval scoring, LLM layer
```

## Remaining work

Local wiring is implemented and tested without credentials. Live checks remain:

1. **Real LLM validation** — provide an API key through the environment,
   run demo + eval with `USE_REAL_LLM=1`, and record real-model scores beside
   the stub baseline. HTTP contract tests do not validate provider quality.
2. **Tracing validation** — provide a LangSmith key, opt in with
   `ENABLE_TRACING=1`, and verify turn summaries in the provider dashboard.
   LangFuse support is not implemented.
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
  the default "LLM" is a rule-based stub, and there are no real users.
- It does not demonstrate production LangGraph/LangChain usage, RAG, or
  paid-model operations — those are the gaps the "Remaining work" section
  is for.
- The eval scores measure the stub's behaviour on a hand-written set, not
  general agent quality.
