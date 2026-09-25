from __future__ import annotations
import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PayloadSchemaType,
)

from src.config import (
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    TOP_K,
)

_openai_client = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def init_qdrant_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the Qdrant collection and payload indexes if they do not already exist.

    recreate=True drops the existing collection first.  Use it whenever the
    chunking changes: chunk IDs change, and stale points from the old
    chunking would otherwise remain searchable."""
    existing = [c.name for c in client.get_collections().collections]
    if recreate and QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        existing.remove(QDRANT_COLLECTION)
        print(f"Dropped collection: {QDRANT_COLLECTION}")
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection: {QDRANT_COLLECTION}")
    else:
        print(f"Collection already exists: {QDRANT_COLLECTION}")

    # Create payload indexes required for metadata-filtered retrieval
    for field, schema_type in [
        ("ticker", PayloadSchemaType.KEYWORD),
        ("section", PayloadSchemaType.KEYWORD),
        ("year", PayloadSchemaType.INTEGER),
    ]:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema_type,
            )
        except Exception:
            pass  # index already exists


def embed_query(query_text: str) -> list[float]:
    """Embed a query string using OpenAI text-embedding-3-small."""
    client = _get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
    return response.data[0].embedding


def _embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    client = _get_openai_client()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
        time.sleep(0.05)
    return all_embeddings


def upsert_chunks(client: QdrantClient, chunks_df: pd.DataFrame, batch_size: int = 100) -> None:
    """
    Embed all chunks and upsert them into Qdrant with full metadata payloads.
    Skips chunks already present in the collection by checking existing IDs.
    """
    texts = chunks_df["text"].tolist()
    chunk_ids = chunks_df["chunk_id"].tolist()

    print(f"Embedding {len(texts)} chunks...")
    embeddings = _embed_batch(texts)

    points = []
    for i, (_, row) in enumerate(chunks_df.iterrows()):
        point_id = abs(hash(row["chunk_id"])) % (2**63)
        points.append(
            PointStruct(
                id=point_id,
                vector=embeddings[i],
                payload={
                    "chunk_id": row["chunk_id"],
                    "ticker": row["ticker"],
                    "company": row["company"],
                    "year": int(row["year"]),
                    "filing_type": row.get("filing_type", "10-K"),
                    "section": row["section"],
                    "text": row["text"],
                    "token_count": int(row["token_count"]),
                },
            )
        )

    print(f"Upserting {len(points)} points to Qdrant...")
    for i in tqdm(range(0, len(points), batch_size), desc="Upserting batches"):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

    print("Upsert complete.")


def _hits_to_results(hits) -> list[dict]:
    results = []
    for hit in hits:
        p = hit.payload
        results.append({
            "chunk_id": p.get("chunk_id"),
            "ticker": p.get("ticker"),
            "company": p.get("company"),
            "year": p.get("year"),
            "section": p.get("section"),
            "text": p.get("text"),
            "score": hit.score,
        })
    return results


def vanilla_retrieval(client: QdrantClient, query: str, top_k: int = TOP_K) -> list[dict]:
    """Pure cosine similarity retrieval with no metadata filters."""
    vector = embed_query(query)
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return _hits_to_results(result.points)


def _build_filter(ticker: str = None, year: int = None, section: str = None) -> Filter | None:
    conditions = []
    if ticker is not None:
        conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
    if year is not None:
        conditions.append(FieldCondition(key="year", match=MatchValue(value=int(year))))
    if section is not None:
        conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))
    if not conditions:
        return None
    return Filter(must=conditions)


def metadata_filtered_retrieval(
    client: QdrantClient,
    query: str,
    ticker: str = None,
    year: int = None,
    section: str = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Similarity search within Qdrant metadata filters."""
    vector = embed_query(query)
    query_filter = _build_filter(ticker=ticker, year=year, section=section)
    result = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return _hits_to_results(result.points)


def task_aware_retrieval(
    client: QdrantClient,
    query: str,
    task_type: str,
    companies: list[str] = None,
    years: list[int] = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Strategy-aware retrieval that routes to the appropriate sub-strategy.

    task_type:
      - "single_fact" / "risk_factor": filter by single company + year
      - "cross_year": retrieve top_k//2 from each of two years for the same company
      - "cross_company": retrieve top_k//2 from each of two companies for the same year
    """
    half_k = max(1, top_k // 2)

    if task_type in ("single_fact", "risk_factor"):
        ticker = companies[0] if companies else None
        year = years[0] if years else None
        return metadata_filtered_retrieval(client, query, ticker=ticker, year=year, top_k=top_k)

    elif task_type == "cross_year":
        if not companies or not years or len(years) < 2:
            return vanilla_retrieval(client, query, top_k=top_k)
        ticker = companies[0]
        results_a = metadata_filtered_retrieval(client, query, ticker=ticker, year=years[0], top_k=half_k)
        results_b = metadata_filtered_retrieval(client, query, ticker=ticker, year=years[1], top_k=half_k)
        return results_a + results_b

    elif task_type == "cross_company":
        if not companies or len(companies) < 2:
            return vanilla_retrieval(client, query, top_k=top_k)
        year = years[0] if years else None
        results_a = metadata_filtered_retrieval(client, query, ticker=companies[0], year=year, top_k=half_k)
        results_b = metadata_filtered_retrieval(client, query, ticker=companies[1], year=year, top_k=half_k)
        return results_a + results_b

    else:
        return vanilla_retrieval(client, query, top_k=top_k)


def run_retrieval_experiments(
    questions_df: pd.DataFrame,
    client: QdrantClient = None,
) -> pd.DataFrame:
    """
    Run all three retrieval strategies for every question.
    Returns a long-format dataframe with columns:
    question_id, strategy, chunk_id, ticker, year, section, text, score
    """
    if client is None:
        client = get_qdrant_client()

    strategies = ["vanilla", "metadata_filtered", "task_aware"]
    records = []

    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Retrieval experiments"):
        question_id = row["question_id"]
        question = row["question"]
        task_type = row["task_type"]
        ticker = row.get("ticker")
        year = row.get("year")
        comp_ticker = row.get("comparison_ticker")
        comp_year = row.get("comparison_year")

        companies = [t for t in [ticker, comp_ticker] if pd.notna(t) and t]
        years_list = [int(y) for y in [year, comp_year] if pd.notna(y) and y]

        for strategy in strategies:
            try:
                if strategy == "vanilla":
                    chunks = vanilla_retrieval(client, question, top_k=TOP_K)
                elif strategy == "metadata_filtered":
                    chunks = metadata_filtered_retrieval(
                        client, question,
                        ticker=ticker if pd.notna(ticker) else None,
                        year=int(year) if pd.notna(year) else None,
                        top_k=TOP_K,
                    )
                else:  # task_aware
                    chunks = task_aware_retrieval(
                        client, question,
                        task_type=task_type,
                        companies=companies,
                        years=years_list,
                        top_k=TOP_K,
                    )

                for rank, chunk in enumerate(chunks):
                    records.append({
                        "question_id": question_id,
                        "strategy": strategy,
                        "rank": rank,
                        **chunk,
                    })

            except Exception as e:
                print(f"  ERROR q={question_id} strategy={strategy}: {e}")

        time.sleep(0.05)

    return pd.DataFrame(records)
