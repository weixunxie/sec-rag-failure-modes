from __future__ import annotations
SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions based only on the provided SEC filing excerpts. Always cite which company and year each piece of evidence comes from.

Always return valid JSON in exactly this format:
{
  "answer": "Your detailed answer here",
  "citations": [
    {"chunk_id": "AAPL_2023_md_a_0001", "ticker": "AAPL", "year": 2023, "section": "MD&A"}
  ],
  "confidence": "high",
  "evidence_used": "Brief description of the evidence found and how it supports the answer"
}

confidence must be one of: "high", "medium", "low"
- high: the evidence directly and fully answers the question
- medium: the evidence partially answers or requires inference
- low: evidence is indirect, incomplete, or the answer is uncertain

Return ONLY the JSON object, no other text."""


EVALUATION_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of answers to financial questions about SEC 10-K filings.

Score three dimensions on a 0-2 scale and return valid JSON:
{
  "answer_correctness": 0,
  "answer_correctness_reason": "...",
  "citation_faithfulness": 0,
  "citation_faithfulness_reason": "...",
  "evidence_coverage": 0,
  "evidence_coverage_reason": "...",
  "overall_quality": "poor|acceptable|good"
}

Scoring rubric:
answer_correctness:
  0 = Wrong or contradicts the evidence
  1 = Partially correct or missing key details
  2 = Fully correct and complete

citation_faithfulness:
  0 = Citations fabricated or do not support the answer
  1 = Citations partially support the answer
  2 = All citations directly support stated facts

evidence_coverage:
  0 = Ignores most relevant retrieved evidence
  1 = Uses some but not all relevant evidence
  2 = Fully synthesizes all relevant evidence

Return ONLY the JSON object."""


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the prompt."""
    if not chunks:
        return "No relevant filing excerpts were retrieved."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        ticker = chunk.get("ticker", "UNK")
        year = chunk.get("year", "????")
        section = chunk.get("section", "Unknown")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "").strip()

        lines.append(f"--- Excerpt {i} ---")
        lines.append(f"Company: {ticker} | Year: {year} | Section: {section} | ID: {chunk_id}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Build the user-facing prompt with the question and formatted context."""
    context = format_chunks_for_prompt(chunks)
    return f"""Question: {question}

Retrieved Filing Excerpts:
{context}

Based solely on the excerpts above, answer the question and return your response as valid JSON."""


def build_evaluation_prompt(
    question: str,
    ground_truth: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
) -> str:
    """Build the evaluation prompt for LLM-as-judge scoring."""
    context_summary = "\n".join(
        f"- [{c.get('ticker')} {c.get('year')} {c.get('section')}]: {c.get('text', '')[:200]}..."
        for c in retrieved_chunks
    )

    return f"""Evaluate this answer to a financial question about SEC 10-K filings.

QUESTION: {question}

GROUND TRUTH: {ground_truth}

GENERATED ANSWER: {generated_answer}

RETRIEVED CONTEXT (summary):
{context_summary}

Score the answer on answer_correctness, citation_faithfulness, and evidence_coverage.
Return only the JSON scoring object."""
