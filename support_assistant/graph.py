r"""Task 3/4 - LangGraph StateGraph: intent router + grounded RAG + JSON schema.

Graph:
    classify_intent --(policy_question)--> retrieve_and_answer --> END
                     \--(general_question)-> direct_answer      --> END

MOCK_LLM toggle (env var):
    unset or "1"  -> graded MOCK baseline: deterministic, no LLM call, no network
    "0"           -> optional real-LLM extension (Groq free tier by default)

Only the *generation* step inside each node branches on MOCK_LLM. Intent routing
and ChromaDB retrieval always run for real (they need no API key / no network).
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END

from prompts import (
    DIRECT_ANSWER_CANNED,
    POLICY_KEYWORDS,
    build_real_prompt,
    mock_policy_answer,
)
from vectorstore import retrieve

SNIPPET_CHARS = 200
TOP_K = 3


def is_mock() -> bool:
    """Graded default is MOCK. Only MOCK_LLM=0 turns on the real-LLM path."""
    return os.getenv("MOCK_LLM", "1") != "0"


# ---------------------------------------------------------------------------
# Task 4 - Pydantic output schema enforced on the final answer.
# ---------------------------------------------------------------------------
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved: list          # list of {id, document, similarity, ...}
    answer: str
    sources: list
    confidence: float


# ---------------------------------------------------------------------------
# Optional real-LLM call (only used when MOCK_LLM=0) with retry-on-validation.
# ---------------------------------------------------------------------------
def _call_real_llm(context: str, question: str) -> AnswerResponse:
    """Call a real LLM (Groq free tier by default) and validate its JSON output.

    Retries up to 2 additional times with a corrective instruction if the raw
    output fails schema validation, then returns a clearly-marked error response.
    """
    from groq import Groq  # lazy import; only needed on the optional path

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    prompt = build_real_prompt(context, question)
    corrective = ""

    for attempt in range(3):  # 1 initial + 2 retries
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt + corrective}],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
            return AnswerResponse(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            corrective = (
                f"\n\n[CORRECTION] Your previous reply was invalid ({exc}). "
                "Reply with ONLY a valid JSON object with keys answer (string), "
                "sources (list of strings), confidence (float 0-1)."
            )

    return AnswerResponse(
        answer="ERROR: the language model did not return valid JSON after retries.",
        sources=[],
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    """Route the query to policy_question or general_question."""
    query = state["query"]
    if is_mock():
        q = query.lower()
        intent = "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional real-LLM classification.
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content":
                "Classify this customer query as exactly 'policy_question' or "
                f"'general_question'. Reply with one word only.\nQuery: {query}"}],
        )
        out = resp.choices[0].message.content.strip().lower()
        intent = "policy_question" if "policy" in out else "general_question"
    return {"intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    """Retrieve top-3 chunks (always real) and generate the grounded answer."""
    query = state["query"]
    hits = retrieve(query, k=TOP_K)          # real cosine retrieval in both modes
    sources = [h["id"] for h in hits]

    if is_mock():
        top_snippet = hits[0]["document"][:SNIPPET_CHARS]
        answer = mock_policy_answer(top_snippet)
        confidence = 1.0
    else:
        context = "\n\n".join(f"(id: {h['id']}) {h['document']}" for h in hits)
        result = _call_real_llm(context, query)
        answer = result.answer
        # Prefer the model's cited sources if valid, else the retrieved ids.
        sources = result.sources or sources
        confidence = result.confidence

    return {"retrieved": hits, "answer": answer,
            "sources": sources, "confidence": confidence}


def direct_answer(state: GraphState) -> GraphState:
    """Handle general (non-policy) questions with no retrieval."""
    if is_mock():
        answer = DIRECT_ANSWER_CANNED
        confidence = 1.0
    else:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": state["query"]}],
        )
        answer = resp.choices[0].message.content.strip()
        confidence = 0.5
    return {"answer": answer, "sources": [], "confidence": confidence}


def _route(state: GraphState) -> str:
    return state["intent"]


# ---------------------------------------------------------------------------
# Build / compile the graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent", _route,
        {"policy_question": "retrieve_and_answer",
         "general_question": "direct_answer"},
    )
    g.add_edge("retrieve_and_answer", END)
    g.add_edge("direct_answer", END)
    return g.compile()


_GRAPH = None


def run_query(query: str) -> AnswerResponse:
    """Run the compiled graph and return the validated Pydantic response."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    final = _GRAPH.invoke({"query": query})
    # Enforce the JSON schema on the final answer.
    return AnswerResponse(
        answer=final["answer"],
        sources=final.get("sources", []),
        confidence=final.get("confidence", 0.0),
    )


if __name__ == "__main__":
    for q in ["What is the delivery fee?", "What is the capital of France?"]:
        print(f"\nQ: {q}")
        print(run_query(q).model_dump_json(indent=2))
