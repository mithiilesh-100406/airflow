"""
embed.py
────────
Task 4 of the Airflow DAG.
Converts cleaned sales rows → text chunks → embeddings → ChromaDB.

Strategy
────────
Each row becomes a natural-language sentence so the LLM can reason over it:

  "Order ORD-1234 placed on 2023-03-15 (Q1 2023) by Customer Jane Doe
   (Consumer segment) from Chicago, Illinois in the Central region.
   Product: Staples (Office Supplies > Supplies). Quantity: 3,
   Sales: $245.50, Discount: 10%, Profit: $38.20 (15.56% margin)."

These chunks are embedded and stored. At query time the same embedding
model encodes the user's question and we retrieve the top-k most
similar chunks.
"""

import os
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CLEANED_PATH   = Path(os.getenv("CLEANED_DATA_PATH", "data/cleaned/sales_cleaned.csv"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "sales_knowledge"
BATCH_SIZE = 200


# ── Row → text chunk ──────────────────────────────────────────────────────────

def row_to_text(row: pd.Series) -> str:
    """Convert one cleaned sales row into a readable text chunk."""

    def g(col, default="N/A"):
        val = row.get(col, default)
        return default if pd.isna(val) else val

    date_str = str(g("order_date", "unknown date"))[:10]

    return (
        f"Order {g('order_id')} placed on {date_str} "
        f"({g('order_quarter', g('order_year', ''))}) "
        f"by customer '{g('customer_name')}' ({g('segment')} segment) "
        f"from {g('city')}, {g('state')} in the {g('region')} region. "
        f"Ship mode: {g('ship_mode')}. "
        f"Product: {g('product_name')} "
        f"(Category: {g('category')} > {g('sub_category')}). "
        f"Quantity: {g('quantity')}, "
        f"Sales: ${float(g('sales', 0)):,.2f}, "
        f"Discount: {float(g('discount', 0)) * 100:.0f}%, "
        f"Profit: ${float(g('profit', 0)):,.2f} "
        f"(margin: {g('profit_margin_pct', 'N/A')}%)."
    )


def build_chunks(df: pd.DataFrame) -> List[dict]:
    """Return list of {id, text, metadata} dicts."""
    chunks = []
    for i, row in df.iterrows():
        text = row_to_text(row)
        meta = {
            "order_id":      str(row.get("order_id", i)),
            "order_date":    str(row.get("order_date", ""))[:10],
            "region":        str(row.get("region", "")),
            "category":      str(row.get("category", "")),
            "sub_category":  str(row.get("sub_category", "")),
            "segment":       str(row.get("segment", "")),
            "sales":         float(row.get("sales", 0) or 0),
            "profit":        float(row.get("profit", 0) or 0),
        }
        chunks.append({"id": f"row_{i}", "text": text, "metadata": meta})
    logger.info(f"Built {len(chunks):,} text chunks")
    return chunks


# ── Embedding + ChromaDB upsert ───────────────────────────────────────────────

def embed_and_store(df: pd.DataFrame) -> None:
    """Embed all chunks and upsert into ChromaDB."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Delete old collection so today's data replaces yesterday's
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = build_chunks(df)

    # Upsert in batches to avoid memory spikes
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.upsert(
            ids       = [c["id"]       for c in batch],
            documents = [c["text"]     for c in batch],
            metadatas = [c["metadata"] for c in batch],
        )
        logger.info(f"  Upserted rows {start}–{start + len(batch) - 1}")

    logger.info(
        f"Vector store ready: {collection.count():,} embeddings "
        f"in '{COLLECTION_NAME}'"
    )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingest import ingest_csv
    from clean import clean
    from validate import validate

    raw     = ingest_csv()
    cleaned = clean(raw)
    validate(cleaned)
    embed_and_store(cleaned)
    print("Done. Vector store populated.")
