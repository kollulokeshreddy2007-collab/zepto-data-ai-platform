"""Task 5 - FastAPI wrapper exposing POST /ask.

Accepts a Pydantic request {"query": str} and returns the validated Pydantic
response {answer, sources, confidence}. Runs the LangGraph pipeline; with
MOCK_LLM left at its default the whole path is deterministic and offline.

Run locally:
    uvicorn app:app --reload            # from inside support_assistant/
    # or from repo root:
    uvicorn support_assistant.app:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from graph import AnswerResponse, run_query

app = FastAPI(title="Zepto Support Assistant", version="1.0")


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root() -> dict:
    return {"service": "Zepto Support Assistant",
            "mock_llm": os.getenv("MOCK_LLM", "1"),
            "endpoint": "POST /ask  {\"query\": \"...\"}"}


@app.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest) -> AnswerResponse:
    return run_query(req.query)
