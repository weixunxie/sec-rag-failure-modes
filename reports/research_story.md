# RAG Failure Modes in SEC 10-K Analysis: Research Story

## Executive Summary

This research evaluates three retrieval-augmented generation (RAG) strategies on a benchmark of 40 financial questions drawn from SEC 10-K filings for five major companies (AAPL, MSFT, NVDA, TSLA, JPM) across 2022–2024. The central question: **when a RAG system gets a financial question wrong, is it failing at retrieval or at reasoning?**

---

## Motivation

Financial analysts increasingly use LLM-based systems to answer questions from long corporate filings. A 10-K can exceed 100,000 words, making retrieval quality a critical bottleneck. But retrieval is not the only failure mode. Even when the right passages are found, models can hallucinate numbers, miscite sections, or fail to synthesize evidence across years or companies.

Distinguishing retrieval errors from generation errors has practical consequences:
- **Retrieval errors** call for better indexing, metadata filtering, or query rewriting
- **Generation errors** call for better prompting, fine-tuning, or output validation

---

## Experimental Design

### Dataset
- **Companies**: AAPL, MSFT, NVDA, TSLA, JPM
- **Years**: 2022, 2023, 2024
- **Filings**: 15 10-K documents from SEC EDGAR
- **Questions**: 40 questions across 4 task types

### Task Types
| Type | Count | Description |
|------|-------|-------------|
| `single_fact` | 10 | Extract a specific figure from a single company-year filing |
| `risk_factor` | 10 | Identify and summarize a specific disclosed risk |
| `cross_year` | 10 | Compare narrative or figures across two fiscal years |
| `cross_company` | 10 | Compare disclosures between two different companies |

### Retrieval Strategies
1. **Vanilla**: Pure cosine similarity over the full collection — no metadata filters
2. **Metadata-Filtered**: Pre-filter by ticker and/or year before similarity search
3. **Task-Aware**: Routes queries to strategy-appropriate retrieval (split-budget for comparisons)

### Chunking Strategy
Section-aware chunking preserves 10-K structure: Business, Risk Factors, MD&A, Market Risk, Financial Statements, Controls. Chunks are 800 tokens with 100-token overlap, bounded by paragraph breaks.

### Evaluation
LLM-as-judge (`gpt-4o-mini`, temperature 0) scores each answer against a reference answer and the chunks that were actually retrieved:
- **Answer Correctness** (0–2): Is the answer factually correct given the ground truth?
- **Citation Faithfulness** (0–2): Do citations support the stated facts?
- **Evidence Coverage** (0–2): Were all relevant retrieved passages used?

---

## Error Taxonomy

Errors are classified into two stages:

### Retrieval-Stage Errors
| Error Type | Description |
|-----------|-------------|
| `WRONG_ENTITY` | Retrieved chunks from the wrong company |
| `WRONG_YEAR` | Retrieved chunks from the wrong fiscal year |
| `WRONG_SECTION` | Retrieved chunks from the wrong 10-K section |
| `INSUFFICIENT_COVERAGE` | Retrieved chunks do not contain the needed evidence |
| `SEMANTIC_MISMATCH` | High cosine similarity but topically irrelevant |

### Generation-Stage Errors
| Error Type | Description |
|-----------|-------------|
| `HALLUCINATION` | Model states facts not in any retrieved chunk |
| `REASONING_ERROR` | Evidence present but model draws wrong conclusion |
| `INCOMPLETE_SYNTHESIS` | Model ignores some relevant retrieved chunks |
| `CITATION_FABRICATION` | Model cites chunk IDs not in the provided context |
| `OVERCONFIDENCE` | High confidence reported despite weak evidence |

---

## Hypotheses

**H1**: Vanilla retrieval will fail disproportionately on `cross_year` and `cross_company` tasks due to semantic overlap between years and companies for the same underlying concepts.

**H2**: Metadata-filtered retrieval will outperform vanilla on `single_fact` and `risk_factor` tasks by eliminating off-entity noise, but will underperform on `cross_year` tasks if year filtering is too strict.

**H3**: Task-aware retrieval will achieve the best coverage on comparison tasks by budgeting half the context window to each comparison target.

**H4**: The majority of errors on `single_fact` tasks will be retrieval errors (wrong year, wrong section), while errors on `cross_year` tasks will more often be generation errors (incomplete synthesis).

---

## Key Figures (Notebook 05)

1. **Error Stage Distribution by Strategy** — Stacked bar showing % retrieval vs. generation errors per strategy
2. **Answer Correctness Heatmap** — Correctness score by (task_type × strategy) showing where each strategy succeeds
3. **Error Type Frequency** — Horizontal bar chart counting each error type across all strategies
4. **Retrieval Precision by Task Type** — How often the right company/year appears in the top-5 retrieved chunks
5. **Score Distributions** — Box plots of correctness/faithfulness/coverage by strategy

---

## Expected Findings

Based on the design, we expect task-aware retrieval to reduce `WRONG_ENTITY` and `WRONG_YEAR` errors by ~50% compared to vanilla on comparison tasks. However, we expect `INCOMPLETE_SYNTHESIS` and `REASONING_ERROR` to remain roughly constant across strategies, confirming that generation-stage failures are strategy-independent and require prompt-level or model-level interventions.

---

## Results Against the Hypotheses (October 2026 re-run)

| Hypothesis | Verdict | Evidence |
|-----------|---------|----------|
| **H1** Vanilla fails disproportionately on comparison tasks | Partly supported | Vanilla's retrieval errors are all on comparison questions (8 of 20), none on single-entity ones |
| **H2** Metadata filtering beats Vanilla on single-entity tasks, loses on cross-year | Half supported | It ties Vanilla on single_fact and risk_factor (1.6 vs 1.6) but fails retrieval on all 10 cross_year questions (correctness 0.9 vs 1.3) |
| **H3** Task-Aware gives the best coverage on comparisons | Supported | 0 retrieval errors on 20 comparison questions; best cross_company correctness (1.4) |
| **H4** single_fact errors are mostly retrieval errors | Not supported | No strategy has a retrieval error on a single_fact question; its errors are generation-stage |

Task-Aware is best overall (45% fully correct vs 40% Vanilla and 30% Metadata Filtered). 39 of 46 generation-stage
errors are partially correct answers rather than hallucinations, and only 10 of 120 answers cite an unretrieved
chunk. Full tables are in the README.

---

## Limitations

- Reference answers are generated by gpt-4o from XBRL facts and the gold filing sections, then checked against the filings by hand (13 of 40 corrected); they are not a fully human-written answer key
- 40 questions per strategy; no significance test yet
- Only 5 companies, 3 years — results may not generalize to smaller or less-covered companies
- Chunking quality varies by filing format (EDGAR XBRL vs. legacy .htm)
- Rate limits on EDGAR API may cause gaps in the downloaded corpus

---

## Conclusion

This framework separates the two root causes of RAG failures in financial document QA and quantifies how retrieval strategy design affects each. The finding that certain error types persist regardless of retrieval strategy is actionable: it directs engineering effort toward prompt engineering and output validation rather than further retrieval refinement.
