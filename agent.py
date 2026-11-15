"""Multi-turn bookstore assistant built with a LangGraph state graph.

Graph (one node per step, memory via a checkpointer):

    understand  ->  search  ->  respond
    understand  ->  respond            (no tool needed)

- ``understand`` classifies the intent, extracts search slots, and updates
  the remembered user preferences (genre / topics).
- ``search`` calls the catalog tool and records the tool call on the
  trajectory.
- ``respond`` generates the reply from the intent, the tool results, and
  the remembered preferences.

State persists per ``thread_id`` through a ``MemorySaver`` checkpointer, so
preferences survive across turns in the same conversation. The LLM layer is
swappable (see ``llm.py``): the deterministic stub runs offline by default.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from catalog import search_books
from llm import get_llm
from tracing import init_tracing, trace_turn
from trajectory import TrajectoryLogger


@tool
def search_catalog_tool(
    genre: str | None = None,
    keywords: list[str] | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    limit: int = 5,
) -> list[dict]:
    """Search the bookstore catalog by genre, topic keywords, price, or rating.

    Returns matching books sorted by rating (highest first). Use this when
    the user asks to find, browse, or get recommendations for books.
    """
    return search_books(
        genre=genre,
        keywords=keywords,
        max_price=max_price,
        min_rating=min_rating,
        limit=limit,
    )


class AgentState(TypedDict):
    """Graph state. ``messages`` is append-only; the rest is overwritten."""

    messages: Annotated[list[BaseMessage], add_messages]
    preferences: dict
    tool_calls: list[dict]
    last_intent: str
    last_query: dict
    last_results: list[dict]


def _last_human_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def build_graph(llm) -> StateGraph:
    """Build the agent graph around the given LLM layer."""

    def understand(state: AgentState) -> dict:
        user_text = _last_human_text(state)
        classification = llm.classify_intent(user_text)
        intent = classification["intent"]
        prefs = dict(state.get("preferences", {}))
        if intent == "preference":
            if classification.get("genre"):
                prefs["genre"] = classification["genre"]
            if classification.get("keywords"):
                prefs["topics"] = classification["keywords"]
        if intent == "recommend":
            # Prefer the remembered genre; fall back to remembered topics.
            if prefs.get("genre"):
                query = {"genre": prefs["genre"]}
            else:
                query = {"keywords": prefs.get("topics") or []}
        elif intent == "search":
            query = {
                "genre": classification.get("genre"),
                "keywords": classification.get("keywords") or [],
                "max_price": classification.get("max_price"),
                "min_rating": classification.get("min_rating"),
            }
        else:
            query = {}
        return {"last_intent": intent, "last_query": query, "preferences": prefs}

    def route(state: AgentState) -> str:
        return "search" if state.get("last_intent") in ("search", "recommend") else "respond"

    def search(state: AgentState) -> dict:
        query = state.get("last_query", {})
        args = {k: v for k, v in query.items() if v is not None and v != []}
        results = search_catalog_tool.invoke(args)
        record = {
            "name": "search_catalog",
            "args": args,
            "result_count": len(results),
            "result_titles": [b["title"] for b in results],
        }
        return {
            "last_results": results,
            "tool_calls": [*state.get("tool_calls", []), record],
        }

    def respond(state: AgentState) -> dict:
        text = llm.respond(
            intent=state.get("last_intent", "chat"),
            results=state.get("last_results", []),
            preferences=state.get("preferences", {}),
            user_text=_last_human_text(state),
            search_genre=state.get("last_query", {}).get("genre"),
        )
        return {"messages": [AIMessage(content=text)]}

    builder = StateGraph(AgentState)
    builder.add_node("understand", understand)
    builder.add_node("search", search)
    builder.add_node("respond", respond)
    builder.add_edge(START, "understand")
    builder.add_conditional_edges("understand", route, {"search": "search", "respond": "respond"})
    builder.add_edge("search", "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=MemorySaver())


class BookstoreAgent:
    """The bookstore assistant: one graph, one thread of memory per conversation."""

    def __init__(self, llm=None, logger: TrajectoryLogger | None = None) -> None:
        self.llm = llm or get_llm()
        self.graph = build_graph(self.llm)
        self.logger = logger or TrajectoryLogger()
        self.tracing_status = init_tracing()
        self.last_turn: dict | None = None

    def chat(self, text: str, thread_id: str = "default") -> str:
        """Send one user message, return the agent's reply, log the turn."""
        config = {"configurable": {"thread_id": thread_id}}
        before = self.graph.get_state(config)
        before_values = before.values or {}
        prev_tool_count = len(before_values.get("tool_calls", []))
        prev_human_count = sum(
            1 for m in before_values.get("messages", []) if isinstance(m, HumanMessage)
        )
        result = self.graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        response = str(result["messages"][-1].content) if result.get("messages") else ""
        record = {
            "turn": prev_human_count + 1,
            "thread_id": thread_id,
            "user": text,
            "intent": result.get("last_intent", ""),
            "tool_calls": result.get("tool_calls", [])[prev_tool_count:],
            "preferences": dict(result.get("preferences", {})),
            "response": response,
        }
        self.last_turn = self.logger.log_turn(record)
        trace_turn(record, self.tracing_status)
        return response
