"""
Build reference ("gold") answers for the 40-question benchmark.

Why this exists
---------------
The first version of the evaluation had no reference answers: the judge was
handed the question text itself as "ground truth", so answer_correctness was
effectively a reference-free plausibility score.  This module creates an
explicit reference answer for every question, from sources that are
independent of the three retrieval strategies being compared:

1. **XBRL facts** (SEC companyfacts API) for the numeric single-fact questions
   (revenue, net income, R&D, operating income, total assets).  These values
   are exact and machine-readable.
2. **Gold-section context**: for every question, the chunks of the *correct*
   filing(s) and the *expected* 10-K section(s) (``expected_evidence_section``
   in questions.csv).  Up to ``GOLD_CHUNKS_PER_TARGET`` chunks per filing are
   taken, ranked by lexical overlap with the question (no embeddings, so the
   reference does not share the retriever's similarity function).
3. A stronger model (``REFERENCE_MODEL``, default gpt-4o) writes a concise
   reference answer using ONLY the XBRL facts and gold context.

Every row is written to data/processed/reference_answers.csv with
``needs_review = True``.  Spot-check them (especially numbers) before
trusting the evaluation; edit the CSV by hand if needed and set
``needs_review`` to False.  ``attach_reference_answers`` then merges the
``expected_answer`` column into questions.csv.

Usage (from the repo root):
    python -m src.reference_answers
"""
from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
import requests
from openai import OpenAI

from src.config import COMPANIES, OPENAI_API_KEY, PROCESSED_DIR, SEC_HEADERS

REFERENCE_MODEL = os.getenv("REFERENCE_MODEL", "gpt-4o")
GOLD_CHUNKS_PER_TARGET = 8
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Section names used by src/chunking.py for each expected_evidence_section label
SECTION_ALIASES = {
    "Financial Statements": ["Financial Statements", "MD&A"],
    "Risk Factors": ["Risk Factors"],
    "MD&A": ["MD&A", "Financial Statements"],
    "Business": ["Business", "Risk Factors"],
    "Market Risk": ["Market Risk", "MD&A"],
    "Controls": ["Controls"],
}

# XBRL concepts per numeric single-fact question: {question_id: [(label, [candidate tags])]}
XBRL_FACTS = {
    "Q001": [("Total net sales", ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"])],
    "Q002": [("Total revenue", ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"])],
    "Q003": [("Total revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])],
    "Q005": [("Net income", ["NetIncomeLoss"])],
    "Q007": [("Research and development expense", ["ResearchAndDevelopmentExpense"]),
             ("Total revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"])],
    "Q009": [("Operating income", ["OperatingIncomeLoss"]),
             ("Total revenue", ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"])],
    "Q010": [("Total assets", ["Assets"])],
}

_client = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# --------------------------------------------------------------------------- XBRL
_facts_cache: dict[str, dict] = {}


def _companyfacts(cik: str) -> dict:
    cik10 = str(cik).zfill(10)
    if cik10 not in _facts_cache:
        r = requests.get(COMPANYFACTS_URL.format(cik=cik10), headers=SEC_HEADERS, timeout=60)
        r.raise_for_status()
        _facts_cache[cik10] = r.json()
        time.sleep(0.15)
    return _facts_cache[cik10]


def xbrl_value(cik: str, tags: list[str], period_end: str, instant: bool = False) -> float | None:
    """Value of the first matching us-gaap tag reported for the period ending ``period_end``
    in an original 10-K (annual duration for flows, point in time for balances)."""
    facts = _companyfacts(cik).get("facts", {}).get("us-gaap", {})
    for tag in tags:
        units = facts.get(tag, {}).get("units", {}).get("USD", [])
        for u in units:
            if u.get("form") != "10-K" or u.get("end") != period_end:
                continue
            if instant:
                return float(u["val"])
            if u.get("start") and (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days > 300:
                return float(u["val"])
    return None


def _fmt_usd(v: float) -> str:
    return f"${v / 1e9:,.3f} billion"


def xbrl_block(qid: str, ticker: str, period_end: str, prior_end: str | None) -> str:
    if qid not in XBRL_FACTS:
        return ""
    cik = COMPANIES[ticker]["cik"]
    lines = []
    for label, tags in XBRL_FACTS[qid]:
        instant = tags == ["Assets"]
        v = xbrl_value(cik, tags, period_end, instant=instant)
        if v is not None:
            lines.append(f"- {label} (period ending {period_end}): {_fmt_usd(v)}")
        if instant and prior_end:
            pv = xbrl_value(cik, tags, prior_end, instant=True)
            if pv is not None:
                lines.append(f"- {label} (period ending {prior_end}): {_fmt_usd(pv)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- gold context
_WORD = re.compile(r"[a-z][a-z0-9&\-]{3,}")


def _overlap(question: str, text: str) -> int:
    q = set(_WORD.findall(question.lower()))
    return sum(1 for w in set(_WORD.findall(text.lower())) if w in q)


def gold_chunks(q: dict, chunks: pd.DataFrame) -> pd.DataFrame:
    targets = [(q["ticker"], int(q["year"]))]
    if q["task_type"] == "cross_year" and pd.notna(q.get("comparison_year")):
        targets.append((q["ticker"], int(q["comparison_year"])))
    if q["task_type"] == "cross_company" and pd.notna(q.get("comparison_ticker")):
        targets.append((q["comparison_ticker"], int(q.get("comparison_year") or q["year"])))
    sections = SECTION_ALIASES.get(str(q.get("expected_evidence_section")), None)

    picked = []
    for ticker, year in targets:
        sub = chunks[(chunks.ticker == ticker) & (chunks.year == year)]
        if sections:
            sec = sub[sub.section.isin(sections)]
            sub = sec if len(sec) else sub
        sub = sub.assign(_score=sub.text.map(lambda t: _overlap(q["question"], t)))
        picked.append(sub.sort_values("_score", ascending=False).head(GOLD_CHUNKS_PER_TARGET))
    return pd.concat(picked) if picked else chunks.iloc[0:0]


REFERENCE_SYSTEM = (
    "You write reference answers for an evaluation benchmark about SEC 10-K filings. "
    "Use ONLY the XBRL facts and filing excerpts provided. Prefer XBRL facts for numbers. "
    "Be concise (2-6 sentences) and specific: include the key figures, names and comparisons "
    "the question asks for. If the provided material does not contain part of the answer, "
    "say exactly which part is not disclosed in the excerpts. Return plain text only."
)


def write_reference(q: dict, gold: pd.DataFrame, xbrl: str) -> str:
    context = "\n\n".join(
        f"[{r.ticker} FY{r.year} | {r.section} | {r.chunk_id}]\n{r.text[:2500]}" for r in gold.itertuples()
    )
    prompt = (
        f"QUESTION: {q['question']}\n\n"
        f"XBRL FACTS (authoritative):\n{xbrl or '(none for this question)'}\n\n"
        f"FILING EXCERPTS:\n{context}\n\nWrite the reference answer."
    )
    resp = _openai().chat.completions.create(
        model=REFERENCE_MODEL,
        temperature=0,
        max_tokens=500,
        messages=[{"role": "system", "content": REFERENCE_SYSTEM}, {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------- driver
def build_reference_answers(overwrite: bool = False) -> pd.DataFrame:
    out_path = os.path.join(PROCESSED_DIR, "reference_answers.csv")
    if os.path.exists(out_path) and not overwrite:
        print(f"Reference answers already exist at {out_path} (pass overwrite=True to rebuild)")
        return pd.read_csv(out_path)

    questions = pd.read_csv(os.path.join(PROCESSED_DIR, "questions.csv"))
    chunks = pd.read_parquet(os.path.join(PROCESSED_DIR, "filing_chunks.parquet"))
    meta = pd.read_csv(os.path.join(PROCESSED_DIR, "filing_metadata.csv"))
    report_end = {(r.ticker, int(r.year)): r.report_date for r in meta.itertuples() if isinstance(getattr(r, "report_date", None), str)}

    rows = []
    for q in questions.to_dict("records"):
        period_end = report_end.get((q["ticker"], int(q["year"])))
        prior_end = report_end.get((q["ticker"], int(q["year"]) - 1))
        try:
            xbrl = xbrl_block(q["question_id"], q["ticker"], period_end, prior_end) if period_end else ""
        except requests.RequestException as e:
            xbrl = ""
            print(f"  XBRL lookup failed for {q['question_id']}: {e}")
        gold = gold_chunks(q, chunks)
        answer = write_reference(q, gold, xbrl)
        rows.append({
            "question_id": q["question_id"],
            "expected_answer": answer,
            "xbrl_facts": xbrl,
            "gold_chunk_ids": json.dumps(gold.chunk_id.tolist()),
            "reference_source": ("xbrl+" if xbrl else "") + f"gold_sections+{REFERENCE_MODEL}",
            "needs_review": True,
        })
        print(f"  {q['question_id']}: {answer[:90]}...")
        time.sleep(0.1)

    ref = pd.DataFrame(rows)
    ref.to_csv(out_path, index=False)
    print(f"\nSaved {len(ref)} reference answers to {out_path}. Review them before evaluating.")
    return ref


def attach_reference_answers() -> pd.DataFrame:
    """Merge expected_answer from reference_answers.csv into questions.csv."""
    q_path = os.path.join(PROCESSED_DIR, "questions.csv")
    questions = pd.read_csv(q_path).drop(columns=["expected_answer"], errors="ignore")
    ref = pd.read_csv(os.path.join(PROCESSED_DIR, "reference_answers.csv"))
    merged = questions.merge(ref[["question_id", "expected_answer"]], on="question_id", how="left")
    merged.to_csv(q_path, index=False)
    print(f"questions.csv now has expected_answer for {merged.expected_answer.notna().sum()}/{len(merged)} questions")
    return merged


if __name__ == "__main__":
    build_reference_answers()
    attach_reference_answers()
