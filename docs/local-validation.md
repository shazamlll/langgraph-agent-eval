# Local validation — 2026-11-15

Validated on Windows with Python 3.13, LangGraph 1.2.22, httpx 0.28.1 and
LangSmith 0.14.17 in a repository-local `.venv`.

## Evidence

- Baseline: `python -m pytest -q` — 31 passed.
- Adapter/tracing regressions before implementation: 26 failed, 8 passed
  in the focused suite (missing injected transports, missing key validation,
  and unimplemented tracing seam).
- Additional regressions reproduced stale catalog results on a non-search
  turn, incorrect real-mode CLI banner, and absent model metadata in reports.
- Final: `python -m pytest -q` — 60 passed.
- `python main.py demo` — four turns; remembered science-fiction preference
  shapes recommendation, followed by a separate mystery search.
- `python main.py eval` — 8/8 scripted conversations pass, aggregate 1.00,
  unchanged threshold 0.80, responder identified as `StubResponder`.
- `python -m pip check` — no broken requirements.
- `git diff --check` — no whitespace errors.

HTTP adapter tests use `httpx.MockTransport`. LangSmith tests use SDK
autospecs and the installed SDK's real serialization with its request method
intercepted. No paid provider request or cloud trace upload was made.

The 1.00 score measures a deterministic stub on eight handwritten
conversations, not general agent quality.
