from __future__ import annotations
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.prompts import SYSTEM_PROMPT, build_user_prompt, format_chunks_for_prompt

_openai_client = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    from src.prompts import format_chunks_for_prompt as _fmt
    return _fmt(chunks)


def generate_answer(
    question: str,
    chunks: list[dict],
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Call the OpenAI API to generate a structured answer from retrieved chunks.
    Returns a dict with keys: answer, citations, confidence, evidence_used, raw_response, error.
    """
    client = _get_client()
    user_prompt = build_user_prompt(question, chunks)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_response = response.choices[0].message.content

        parsed = json.loads(raw_response)
        return {
            "answer": parsed.get("answer", ""),
            "citations": parsed.get("citations", []),
            "confidence": parsed.get("confidence", "low"),
            "evidence_used": parsed.get("evidence_used", ""),
            "raw_response": raw_response,
            "error": None,
        }

    except json.JSONDecodeError as e:
        return {
            "answer": raw_response if "raw_response" in dir() else "",
            "citations": [],
            "confidence": "low",
            "evidence_used": "",
            "raw_response": raw_response if "raw_response" in dir() else "",
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "answer": "",
            "citations": [],
            "confidence": "low",
            "evidence_used": "",
            "raw_response": "",
            "error": f"API error: {e}",
        }


def run_experiment(
    questions_df: pd.DataFrame,
    retrieval_results_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    For each question × retrieval strategy combination, generate an answer.

    questions_df: columns include question_id, question
    retrieval_results_df: long-format with question_id, strategy, chunk_id, text, ...

    Returns dataframe with columns:
    question_id, strategy, answer, citations_json, confidence, evidence_used, raw_response, error
    """
    records = []
    strategies = retrieval_results_df["strategy"].unique()

    for _, q_row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Generating answers"):
        question_id = q_row["question_id"]
        question = q_row["question"]

        for strategy in strategies:
            subset = retrieval_results_df[
                (retrieval_results_df["question_id"] == question_id)
                & (retrieval_results_df["strategy"] == strategy)
            ]
            if "rank" in retrieval_results_df.columns:
                subset = subset.sort_values("rank")

            chunks = subset.to_dict("records")
            result = generate_answer(question, chunks, model=model)

            records.append({
                "question_id": question_id,
                "strategy": strategy,
                "answer": result["answer"],
                "citations_json": json.dumps(result["citations"]),
                "confidence": result["confidence"],
                "evidence_used": result["evidence_used"],
                "raw_response": result["raw_response"],
                "error": result["error"],
            })

            time.sleep(0.05)

    return pd.DataFrame(records)
