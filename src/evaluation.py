from __future__ import annotations
import json
import time
import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from src.config import OPENAI_API_KEY, PROCESSED_DIR
from src.prompts import build_evaluation_prompt, EVALUATION_SYSTEM_PROMPT
from src.error_taxonomy import ERROR_TYPES, get_retrieval_error_types, get_generation_error_types

_openai_client = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def evaluate_answer(
    question: str,
    reference_answer: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Use an LLM judge (gpt-4o-mini by default) to score the generated answer
    against the reference answer and the chunks that were actually retrieved.
    Returns dict with answer_correctness, citation_faithfulness, evidence_coverage (each 0-2).
    """
    client = _get_client()
    eval_prompt = build_evaluation_prompt(question, reference_answer, generated_answer, retrieved_chunks)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=512,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": eval_prompt},
            ],
        )
        raw = response.choices[0].message.content

        scores = json.loads(raw)
        return {
            "answer_correctness": int(scores.get("answer_correctness", 0)),
            "citation_faithfulness": int(scores.get("citation_faithfulness", 0)),
            "evidence_coverage": int(scores.get("evidence_coverage", 0)),
            "overall_quality": scores.get("overall_quality", "poor"),
            "answer_correctness_reason": scores.get("answer_correctness_reason", ""),
            "citation_faithfulness_reason": scores.get("citation_faithfulness_reason", ""),
            "evidence_coverage_reason": scores.get("evidence_coverage_reason", ""),
            "eval_error": None,
        }

    except Exception as e:
        return {
            "answer_correctness": 0,
            "citation_faithfulness": 0,
            "evidence_coverage": 0,
            "overall_quality": "poor",
            "answer_correctness_reason": "",
            "citation_faithfulness_reason": "",
            "evidence_coverage_reason": "",
            "eval_error": str(e),
        }


def expected_targets(q: dict) -> tuple[set[str], set[int]]:
    """Tickers and fiscal years a question needs evidence from (both sides for comparisons)."""
    tickers = {str(q.get("ticker"))} if q.get("ticker") else set()
    years = {int(q["year"])} if pd.notna(q.get("year")) else set()
    if q.get("task_type") == "cross_company" and pd.notna(q.get("comparison_ticker")):
        tickers.add(str(q["comparison_ticker"]))
    if q.get("task_type") == "cross_year" and pd.notna(q.get("comparison_year")):
        years.add(int(q["comparison_year"]))
    if q.get("task_type") == "cross_company" and pd.notna(q.get("comparison_year")):
        years.add(int(q["comparison_year"]))
    return tickers, years


def fabricated_citations(citations: list[dict], retrieved_chunks: list[dict]) -> list[str]:
    """Cited chunk IDs that were not among the retrieved chunks (deterministic check)."""
    retrieved_ids = {c.get("chunk_id") for c in retrieved_chunks}
    return [c.get("chunk_id") for c in citations if c.get("chunk_id") and c.get("chunk_id") not in retrieved_ids]


def classify_error(
    question: str,
    retrieved_chunks: list[dict],
    generated_answer: str,
    scores: dict,
    expected_tickers: set[str] | None = None,
    expected_years: set[int] | None = None,
    citations: list[dict] | None = None,
) -> str:
    """
    Classify the error type for an imperfect answer.

    Retrieval-stage checks use the chunks that were ACTUALLY RETRIEVED (not the
    chunks the model chose to cite), and for comparison questions require
    evidence for *every* entity / year being compared.  Citation fabrication
    is detected deterministically (cited ID not in the retrieved set).
    Returns an error type key from ERROR_TYPES.
    """
    answer_correctness = scores.get("answer_correctness", 0)
    citation_faithfulness = scores.get("citation_faithfulness", 0)
    evidence_coverage = scores.get("evidence_coverage", 0)
    citations = citations or []

    if answer_correctness == 2 and citation_faithfulness >= 1 and evidence_coverage >= 1 \
            and not fabricated_citations(citations, retrieved_chunks):
        return "CORRECT"

    # ---- retrieval stage ----
    if len(retrieved_chunks) == 0:
        return "INSUFFICIENT_COVERAGE"
    retrieved_tickers = {str(c.get("ticker")) for c in retrieved_chunks if c.get("ticker")}
    retrieved_years = {int(c.get("year")) for c in retrieved_chunks if c.get("year") is not None}
    if expected_tickers and not expected_tickers.issubset(retrieved_tickers):
        return "WRONG_ENTITY"
    if expected_years and not expected_years.issubset(retrieved_years):
        return "WRONG_YEAR"

    # ---- generation stage ----
    if fabricated_citations(citations, retrieved_chunks):
        return "CITATION_FABRICATION"

    if evidence_coverage == 0:
        total_content = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
        question_terms = [w for w in question.lower().split() if len(w) > 5]
        matches = sum(1 for t in question_terms if t in total_content)
        if question_terms and matches / len(question_terms) < 0.2:
            return "SEMANTIC_MISMATCH"          # retrieval: topically off despite right entity/year
        if answer_correctness < 2:
            return "INCOMPLETE_SYNTHESIS"

    if answer_correctness == 0 and citation_faithfulness == 0:
        return "HALLUCINATION"

    if answer_correctness == 0 and "high" in str(scores.get("confidence", "")).lower():
        return "OVERCONFIDENCE"

    if answer_correctness < 2:
        return "REASONING_ERROR"

    if citation_faithfulness == 0:
        return "CITATION_FABRICATION"          # IDs exist but do not support the statements

    return "INCOMPLETE_SYNTHESIS"


def run_full_evaluation(
    results_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    Evaluate all (question, strategy, answer) triples.

    ``questions_df`` must contain an ``expected_answer`` column (see
    src/reference_answers.py); evaluation refuses to run without reference
    answers rather than silently judging against the question text.
    ``retrieval_df`` supplies the chunks each strategy actually retrieved.
    Saves to data/processed/evaluation_results.csv and returns the dataframe.
    """
    if "expected_answer" not in questions_df.columns or questions_df["expected_answer"].isna().any():
        missing = questions_df.loc[questions_df.get("expected_answer", pd.Series(index=questions_df.index)).isna(), "question_id"].tolist() \
            if "expected_answer" in questions_df.columns else "all"
        raise ValueError(f"Reference answers missing for: {missing}. Run src/reference_answers.py first.")

    q_lookup = {row["question_id"]: row.to_dict() for _, row in questions_df.iterrows()}
    chunks_by_key = {
        key: grp.sort_values("rank").to_dict("records")
        for key, grp in retrieval_df.groupby(["question_id", "strategy"])
    }

    records = []
    for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Evaluating answers"):
        question_id = row["question_id"]
        strategy = row["strategy"]
        generated_answer = row.get("answer", "") if pd.notna(row.get("answer", "")) else ""
        q = q_lookup.get(question_id, {})
        retrieved = chunks_by_key.get((question_id, strategy), [])

        citations = []
        try:
            citations = json.loads(row.get("citations_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            pass

        scores = evaluate_answer(
            question=q.get("question", ""),
            reference_answer=q["expected_answer"],
            generated_answer=generated_answer,
            retrieved_chunks=retrieved,
            model=model,
        )
        scores["confidence"] = row.get("confidence", "")

        exp_t, exp_y = expected_targets(q)
        error_type = classify_error(
            question=q.get("question", ""),
            retrieved_chunks=retrieved,
            generated_answer=generated_answer,
            scores=scores,
            expected_tickers=exp_t,
            expected_years=exp_y,
            citations=citations,
        )
        error_stage = ERROR_TYPES.get(error_type, {}).get("stage", "none")

        records.append({
            "question_id": question_id,
            "task_type": q.get("task_type", ""),
            "strategy": strategy,
            "answer": generated_answer,
            "confidence": row.get("confidence", ""),
            "answer_correctness": scores["answer_correctness"],
            "citation_faithfulness": scores["citation_faithfulness"],
            "evidence_coverage": scores["evidence_coverage"],
            "overall_quality": scores["overall_quality"],
            "error_type": error_type,
            "error_stage": error_stage,
            "n_retrieved": len(retrieved),
            "n_fabricated_citations": len(fabricated_citations(citations, retrieved)),
            "answer_correctness_reason": scores["answer_correctness_reason"],
            "citation_faithfulness_reason": scores["citation_faithfulness_reason"],
            "evidence_coverage_reason": scores["evidence_coverage_reason"],
            "eval_error": scores.get("eval_error"),
        })
        time.sleep(0.05)

    eval_df = pd.DataFrame(records)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "evaluation_results.csv")
    eval_df.to_csv(output_path, index=False)
    print(f"\nEvaluation results saved to {output_path}")
    return eval_df
