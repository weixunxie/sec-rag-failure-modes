from __future__ import annotations
ERROR_TYPES = {
    # Retrieval-stage errors
    "WRONG_ENTITY": {
        "stage": "retrieval",
        "description": "Retrieved chunks from wrong company (ticker mismatch)",
        "example": "Question about AAPL but retrieved MSFT chunks",
    },
    "WRONG_YEAR": {
        "stage": "retrieval",
        "description": "Retrieved chunks from wrong fiscal year",
        "example": "Question about 2023 but retrieved 2022 filing",
    },
    "WRONG_SECTION": {
        "stage": "retrieval",
        "description": "Retrieved chunks from wrong 10-K section",
        "example": "Risk factor question but retrieved from Business section",
    },
    "INSUFFICIENT_COVERAGE": {
        "stage": "retrieval",
        "description": "Retrieved chunks lack the evidence needed to answer",
        "example": "Key numbers not present in any retrieved chunk",
    },
    "SEMANTIC_MISMATCH": {
        "stage": "retrieval",
        "description": "High cosine similarity but topically unrelated to question",
        "example": "Boilerplate legal text retrieved for financial question",
    },
    # Generation-stage errors
    "HALLUCINATION": {
        "stage": "generation",
        "description": "Model states facts not present in any retrieved chunk",
        "example": "Cites revenue figure that does not appear in context",
    },
    "REASONING_ERROR": {
        "stage": "generation",
        "description": "Evidence is present but model draws wrong conclusion",
        "example": "Correct numbers retrieved but arithmetic is wrong",
    },
    "INCOMPLETE_SYNTHESIS": {
        "stage": "generation",
        "description": "Model uses only a subset of relevant retrieved chunks",
        "example": "Cross-year question answered with only one year's data",
    },
    "CITATION_FABRICATION": {
        "stage": "generation",
        "description": "Model cites chunk_ids that do not exist in the retrieved set",
        "example": "Citations reference chunk IDs not present in context",
    },
    "OVERCONFIDENCE": {
        "stage": "generation",
        "description": "Model reports high confidence despite weak or absent evidence",
        "example": "confidence=high but evidence_coverage score is 0",
    },
    # No error
    "CORRECT": {
        "stage": "none",
        "description": "Answer is correct with proper evidence and citations",
        "example": "All scores are 2/2",
    },
}


def is_retrieval_error(error_type: str) -> bool:
    return ERROR_TYPES.get(error_type, {}).get("stage") == "retrieval"


def is_reasoning_error(error_type: str) -> bool:
    return ERROR_TYPES.get(error_type, {}).get("stage") == "generation"


def get_error_stage(error_type: str) -> str:
    return ERROR_TYPES.get(error_type, {}).get("stage", "none")


def get_retrieval_error_types() -> list:
    return [k for k, v in ERROR_TYPES.items() if v["stage"] == "retrieval"]


def get_generation_error_types() -> list:
    return [k for k, v in ERROR_TYPES.items() if v["stage"] == "generation"]
