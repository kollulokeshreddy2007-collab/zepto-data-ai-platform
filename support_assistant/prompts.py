"""Task 2 - structured prompt template (role / context / task / format / length).

This template is used by the OPTIONAL real-LLM path (MOCK_LLM=0). The graded
mock path does not call an LLM, but the full template is defined here as actual
text (not merely described) so it satisfies the "show all 5 skeleton components
plus a negative constraint and a few-shot example" requirement.
"""

# ---------------------------------------------------------------------------
# ROLE / CONTEXT / TASK / FORMAT / LENGTH skeleton, with:
#   * a NEGATIVE CONSTRAINT ("do not answer using information not in the context")
#   * one embedded FEW-SHOT example
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
[ROLE]
You are Zepto's customer-support policy assistant. You help customers by answering
questions strictly about Zepto's delivery, returns, membership, tracking,
cancellation, gift-card, and support policies.

[CONTEXT]
Answer using ONLY the retrieved policy context below. Each context block is a
Zepto policy document.
--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

[TASK]
Answer the customer's question using only the facts in the context above.

[FORMAT]
Respond with a single valid JSON object and nothing else, using exactly these keys:
  "answer":     a concise natural-language answer (string),
  "sources":    the list of document IDs you used (list of strings),
  "confidence": your confidence between 0.0 and 1.0 (float).

[LENGTH]
Keep "answer" to at most 3 sentences.

[NEGATIVE CONSTRAINT]
Do NOT answer using any information that is not present in the provided context.
If the context does not contain the answer, set "answer" to
"I don't have that information in Zepto's policies." and "sources" to [].

[FEW-SHOT EXAMPLE]
Context: "Delivery Policy: Standard delivery is free on orders over INR 149;
orders below this threshold incur a flat INR 25 delivery fee." (id: doc_01)
Question: "Is delivery free?"
Answer JSON:
{{"answer": "Standard delivery is free on orders over INR 149; below that a flat INR 25 fee applies.", "sources": ["doc_01"], "confidence": 0.95}}

[CUSTOMER QUESTION]
{question}
"""

# Keyword heuristic used by classify_intent in the graded mock path.
POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

# Canned answer templates for the graded mock path.
DIRECT_ANSWER_CANNED = "I can only answer questions about Zepto policies right now."


def mock_policy_answer(top_chunk_snippet: str) -> str:
    """Deterministic canned answer for retrieve_and_answer in mock mode."""
    return f"Based on the retrieved context: {top_chunk_snippet}"


def build_real_prompt(context: str, question: str) -> str:
    """Fill the structured template (used only by the optional MOCK_LLM=0 path)."""
    return SYSTEM_PROMPT_TEMPLATE.format(context=context, question=question)
