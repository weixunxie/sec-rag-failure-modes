from __future__ import annotations
import os
import re
import uuid
import pandas as pd
import tiktoken
from tqdm import tqdm

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR

SECTION_PATTERNS = [
    (re.compile(r"item\s+1a[\s\.\:\-]", re.IGNORECASE), "Risk Factors"),
    (re.compile(r"item\s+1[^a-z0-9][\s\.\:\-]", re.IGNORECASE), "Business"),
    (re.compile(r"item\s+7a[\s\.\:\-]", re.IGNORECASE), "Market Risk"),
    (re.compile(r"item\s+7[^a-z0-9][\s\.\:\-]", re.IGNORECASE), "MD&A"),
    (re.compile(r"item\s+8[\s\.\:\-]", re.IGNORECASE), "Financial Statements"),
    (re.compile(r"item\s+9a[\s\.\:\-]", re.IGNORECASE), "Controls"),
]

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def split_into_sections(text: str) -> list[dict]:
    """
    Detect 10-K section boundaries and split the filing into labelled sections.
    Returns list of {section_name: str, text: str}.
    """
    lines = text.splitlines()
    section_starts = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern, section_name in SECTION_PATTERNS:
            if pattern.match(stripped):
                section_starts.append((i, section_name))
                break

    if not section_starts:
        return [{"section_name": "Full Document", "text": text}]

    sections = []
    for idx, (line_idx, section_name) in enumerate(section_starts):
        start = line_idx
        end = section_starts[idx + 1][0] if idx + 1 < len(section_starts) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()
        if section_text:
            sections.append({"section_name": section_name, "text": section_text})

    # Prepend any text before the first detected section as a preamble
    if section_starts[0][0] > 0:
        preamble = "\n".join(lines[: section_starts[0][0]]).strip()
        if preamble:
            sections.insert(0, {"section_name": "Preamble", "text": preamble})

    return sections


def chunk_section(section_name: str, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a section's text into overlapping token-bounded chunks.
    Splits by paragraph first, then merges or splits further to hit chunk_size.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks = []
    current_tokens = []
    current_texts = []

    for para in paragraphs:
        para_tokens = TOKENIZER.encode(para)

        if len(para_tokens) > chunk_size:
            # Flush current buffer first
            if current_tokens:
                chunk_text = "\n\n".join(current_texts)
                chunks.append({
                    "section": section_name,
                    "text": chunk_text,
                    "token_count": len(current_tokens),
                })
                # Keep overlap from end
                overlap_tokens = current_tokens[-overlap:]
                current_tokens = list(overlap_tokens)
                current_texts = [TOKENIZER.decode(overlap_tokens)]

            # Split the large paragraph by sliding window
            for start in range(0, len(para_tokens), chunk_size - overlap):
                window = para_tokens[start: start + chunk_size]
                if len(window) < 10:
                    break
                chunks.append({
                    "section": section_name,
                    "text": TOKENIZER.decode(window),
                    "token_count": len(window),
                })
            continue

        if len(current_tokens) + len(para_tokens) + 2 > chunk_size:
            # Flush
            if current_tokens:
                chunk_text = "\n\n".join(current_texts)
                chunks.append({
                    "section": section_name,
                    "text": chunk_text,
                    "token_count": len(current_tokens),
                })
                # Retain overlap
                overlap_tokens = current_tokens[-overlap:]
                current_tokens = list(overlap_tokens)
                current_texts = [TOKENIZER.decode(overlap_tokens)] if overlap_tokens else []

        current_tokens.extend(para_tokens)
        current_tokens.append(198)  # newline token
        current_texts.append(para)

    if current_tokens:
        chunks.append({
            "section": section_name,
            "text": "\n\n".join(current_texts),
            "token_count": len(current_tokens),
        })

    return chunks


def chunk_filing(filing_metadata: dict, raw_text: str) -> list[dict]:
    """
    Chunk a single filing into labelled, metadata-enriched chunks.
    Returns list of chunk dicts with full provenance metadata.
    """
    ticker = filing_metadata["ticker"]
    year = filing_metadata["year"]
    company = filing_metadata.get("company", "")
    filing_type = filing_metadata.get("filing_type", "10-K")

    sections = split_into_sections(raw_text)
    all_chunks = []
    section_counters: dict[str, int] = {}

    for section_info in sections:
        section_name = section_info["section_name"]
        section_slug = _slugify(section_name)
        raw_chunks = chunk_section(section_name, section_info["text"])

        for raw_chunk in raw_chunks:
            idx = section_counters.get(section_slug, 0)
            section_counters[section_slug] = idx + 1

            chunk_id = f"{ticker}_{year}_{section_slug}_{idx:04d}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "ticker": ticker,
                "company": company,
                "year": year,
                "filing_type": filing_type,
                "section": raw_chunk["section"],
                "text": raw_chunk["text"],
                "token_count": raw_chunk["token_count"],
            })

    return all_chunks


def process_all_filings(metadata_df: pd.DataFrame, raw_dir: str = None) -> pd.DataFrame:
    """
    Process all filings in the metadata dataframe into chunks.
    Saves to data/processed/filing_chunks.parquet and returns the dataframe.
    """
    if raw_dir is None:
        from src.config import RAW_DIR
        raw_dir = RAW_DIR

    all_chunks = []
    valid_rows = metadata_df[metadata_df["local_path"].notna()].copy()

    for _, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Chunking filings"):
        local_path = row["local_path"]
        if not os.path.exists(local_path):
            print(f"  File not found: {local_path}")
            continue

        with open(local_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        filing_metadata = row.to_dict()
        chunks = chunk_filing(filing_metadata, raw_text)
        all_chunks.extend(chunks)
        print(f"  {row['ticker']} {row['year']}: {len(chunks)} chunks")

    df = pd.DataFrame(all_chunks)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "filing_chunks.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\nChunks saved to {output_path}")
    print(f"Total chunks: {len(df)}")

    return df
