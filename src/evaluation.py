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
    ground_truth: str,
    generated_answer: str,
    retrieved_chunks: list[dict],
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Use GPT-4o-mini as a judge to score the generated answer.
    Returns dict with answer_correctness, citation_faithfulness, evidence_coverage (each 0-2).
    """
    client = _get_client()
    eval_prompt = build_evaluation_prompt(question, ground_truth, generated_answer, retrieved_chunks)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=512,
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


def classify_error(
    question: str,
    retrieved_chunks: list[dict],
    generated_answer: str,
    scores: dict,
    expected_ticker: str = None,
    expected_year: int = None,
) -> str:
    """
    Classify the error type for an imperfect answer.
    Returns an error type key from ERROR_TYPES.
    """
    answer_correctness = scores.get("answer_correctness", 0)
    citation_faithfulness = scores.get("citation_faithfulness", 0)
    evidence_coverage = scores.get("evidence_coverage", 0)

    if answer_correctness == 2 and citation_faithfulness >= 1 and evidence_coverage >= 1:
        return "CORRECT"

    retrieved_tickers = {c.get("ticker") for c in retrieved_chunks if c.get("ticker")}
    retrieved_years = {c.get("year") for c in retrieved_chunks if c.get("year")}

    if expected_ticker and retrieved_tickers and expected_ticker not in retrieved_tickers:
        return "WRONG_ENTITY"

    if expected_year and retrieved_years and int(expected_year) not in {int(y) for y in retrieved_years}:
        return "WRONG_YEAR"

    if len(retrieved_chunks) == 0:
        return "INSUFFICIENT_COVERAGE"

    if evidence_coverage == 0 and answer_correctness < 2:
        if citation_faithfulness == 0:
            return "CITATION_FABRICATION"
        return "INCOMPLETE_SYNTHESIS"

    if answer_correctness == 0 and evidence_coverage >= 1:
        return "REASONING_ERROR"

    if answer_correctness == 0 and "high" in generated_answer.lower():
        return "OVERCONFIDENCE"

    if citation_faithfulness == 0:
        return "CITATION_FABRICATION"

    if evidence_coverage == 0:
        total_content = " ".join(c.get("text", "") for c in retrieved_chunks).lower()
        question_terms = [w for w in question.lower().split() if len(w) > 5]
        matches = sum(1 for t in question_terms if t in total_content)
        if question_terms and matches / len(question_terms) < 0.2:
            return "SEMANTIC_MISMATCH"
        return "INSUFFICIENT_COVERAGE"

    if answer_correctness == 1:
        return "REASONING_ERROR"

    if answer_correctness == 0:
        return "HALLUCINATION"

    return "CORRECT"


def run_full_evaluation(
    results_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    Evaluate all (question, strategy, answer) triples.
    Saves to data/processed/evaluation_results.csv and returns the dataframe.
    """
    gt_lookup = {row["question_id"]: row.to_dict() for _, row in ground_truth_df.iterrows()}

    records = []
    for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Evaluating answers"):
        question_id = row["question_id"]
        strategy = row["strategy"]
        generated_answer = row.get("answer", "")
        citations_json = row.get("citations_json", "[]")

        gt = gt_lookup.get(question_id, {})
        question = gt.get("question", "")
        ground_truth = gt.get("expected_answer", gt.get("question", ""))
        expected_ticker = gt.get("ticker")
        expected_year = gt.get("year")
        task_type = gt.get("task_type", "")

        citations = []
        try:
            citations = json.loads(citations_json) if citations_json else []
        except json.JSONDecodeError:
            pass

        retrieved_chunks_stub = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "ticker": c.get("ticker", ""),
                "year": c.get("year"),
                "section": c.get("section", ""),
                "text": "",
            }
            for c in citations
        ]

        scores = evaluate_answer(
            question=question,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            retrieved_chunks=retrieved_chunks_stub,
            model=model,
        )

        error_type = classify_error(
            question=question,
            retrieved_chunks=retrieved_chunks_stub,
            generated_answer=generated_answer,
            scores=scores,
            expected_ticker=expected_ticker,
            expected_year=int(expected_year) if pd.notna(expected_year) and expected_year else None,
        )

        error_stage = ERROR_TYPES.get(error_type, {}).get("stage", "none")

        records.append({
            "question_id": question_id,
            "task_type": task_type,
            "strategy": strategy,
            "answer": generated_answer,
            "confidence": row.get("confidence", ""),
            "answer_correctness": scores["answer_correctness"],
            "citation_faithfulness": scores["citation_faithfulness"],
            "evidence_coverage": scores["evidence_coverage"],
            "overall_quality": scores["overall_quality"],
            "error_type": error_type,
            "error_stage": error_stage,
            "answer_correctness_reason": scores["answer_correctness_reason"],
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
