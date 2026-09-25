from __future__ import annotations
import os
import re
import uuid
import pandas as pd
import tiktoken
from tqdm import tqdm

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DIR

# Order matters: more specific items (1A, 7A, 9A) are tested before their parents.
# Every other Item / Part heading is mapped to "Other" so that a section ends at the
# next heading instead of running to the end of the document (previously everything
# after Item 9A -- exhibits, signatures, JPM's full annual report -- was labelled
# "Controls").
_ITEM = r"^(?:part\s+[iv]+\s*[\.\-\u2014\u2013:]?\s*)?item\s*"
SECTION_PATTERNS = [
    (re.compile(_ITEM + r"1a\b", re.IGNORECASE), "Risk Factors"),
    (re.compile(_ITEM + r"1b\b", re.IGNORECASE), "Other"),
    (re.compile(_ITEM + r"1c\b", re.IGNORECASE), "Other"),
    (re.compile(_ITEM + r"1\b(?![0-9])", re.IGNORECASE), "Business"),
    (re.compile(_ITEM + r"7a\b", re.IGNORECASE), "Market Risk"),
    (re.compile(_ITEM + r"7\b(?![0-9])", re.IGNORECASE), "MD&A"),
    (re.compile(_ITEM + r"8\b(?![0-9])", re.IGNORECASE), "Financial Statements"),
    (re.compile(_ITEM + r"9a\b", re.IGNORECASE), "Controls"),
    (re.compile(_ITEM + r"(?:9b|9c|9|1[0-6]|[2-6])\b", re.IGNORECASE), "Other"),
    (re.compile(r"^(?:signatures|exhibit index|exhibits and financial statement schedules)\b", re.IGNORECASE), "Other"),
]

# A real section is much longer than a table-of-contents line or a cross-reference.
MIN_SECTION_CHARS = 1500

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def split_into_sections(text: str) -> list[dict]:
    """
    Detect 10-K section boundaries and split the filing into labelled sections.
    Returns list of {section_name: str, text: str}.

    Heading lines are matched at the start of a line.  Candidate sections
    shorter than MIN_SECTION_CHARS (table-of-contents entries, "see Item 7"
    cross-references) are merged into the preceding section, so the label of
    a chunk reflects the substantive section it belongs to.
    """
    lines = text.splitlines()
    starts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 200:
            continue
        for pattern, section_name in SECTION_PATTERNS:
            if pattern.match(stripped):
                starts.append((i, section_name))
                break

    if not starts:
        return [{"section_name": "Full Document", "text": text}]

    raw = []
    if starts[0][0] > 0:
        raw.append(("Preamble", "\n".join(lines[: starts[0][0]])))
    for idx, (line_idx, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        raw.append((name, "\n".join(lines[line_idx:end])))

    sections: list[dict] = []
    for name, body in raw:
        body = body.strip()
        if not body:
            continue
        if len(body) < MIN_SECTION_CHARS and sections:
            sections[-1]["text"] += "\n" + body          # TOC line / cross-reference
            continue
        if sections and sections[-1]["section_name"] == name:
            sections[-1]["text"] += "\n" + body
        else:
            sections.append({"section_name": name, "text": body})
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
