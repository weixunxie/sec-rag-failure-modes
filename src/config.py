from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# Optional: a local directory for embedded Qdrant storage. When set, it is used
# instead of QDRANT_URL (no server needed; one process at a time).
QDRANT_PATH = os.getenv("QDRANT_PATH")

COMPANIES = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "MSFT": {"cik": "0000789019", "name": "Microsoft Corporation"},
    "NVDA": {"cik": "0001045810", "name": "NVIDIA Corporation"},
    "TSLA": {"cik": "0001318605", "name": "Tesla, Inc."},
    "JPM":  {"cik": "0000019617", "name": "JPMorgan Chase & Co."},
}

YEARS = [2022, 2023, 2024]
FILING_TYPE = "10-K"

QDRANT_COLLECTION = "sec_filings"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5

SEC_HEADERS = {"User-Agent": "research stephanie@example.com"}
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FILING_BASE_URL = "https://www.sec.gov/Archives/edgar/full-index"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/{quarter}/company.idx"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
