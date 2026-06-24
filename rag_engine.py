"""
rag_engine.py
─────────────
The retrieval-augmented generation engine.

1. User sends a natural-language question.
2. We embed the question with the same model used at ingest time.
3. We retrieve the top-k most similar sales chunks from ChromaDB.
4. We inject those chunks as context into a prompt and call the LLM.
5. The LLM returns a grounded, data-backed answer.

Supports two LLM backends (set LLM_PROVIDER in .env):
  • "ollama"  — local LLaMA 3 / Mistral (free, private, needs Ollama installed)
  • "openai"  — GPT-4o via API key (paid, hosted)
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_PATH", "data/vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "sales_knowledge"
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K           = 8   # how many chunks to retrieve per query


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a sales analytics assistant.
You answer questions about the company's sales data using ONLY the context
chunks provided below. Each chunk describes one sales order.

Rules:
- Base every answer strictly on the provided chunks.
- If the answer cannot be determined from the chunks, say so clearly.
- For aggregations (totals, averages, counts) compute from the chunks given.
- Format currency as USD with commas, percentages with 1 decimal place.
- Be concise but complete. Use bullet points for lists.
"""


# ── LLM call helpers ──────────────────────────────────────────────────────────

def _call_ollama(prompt: str, context: str) -> str:
    import ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"},
        ],
    )
    return response["message"]["content"]


def _call_openai(prompt: str, context: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ── RAG Engine ────────────────────────────────────────────────────────────────

class SalesRAGEngine:
    def __init__(self):
        logger.info("Initialising RAG engine…")
        emb_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
        )
        logger.info(
            f"Connected to '{COLLECTION_NAME}' "
            f"({self.collection.count():,} documents)"
        )

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Return the top-k most relevant chunks for a question."""
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":      doc,
                "metadata":  meta,
                "relevance": round(1 - dist, 3),   # cosine similarity
            })
        return chunks

    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Full RAG pipeline: retrieve → build context → call LLM.
        Returns {"answer": str, "sources": list[dict]}.
        """
        logger.info(f"Query: {question}")
        chunks = self.retrieve(question, top_k)

        if not chunks:
            return {
                "answer":  "No relevant sales data found for your query.",
                "sources": [],
            }

        # Build context string
        context = "\n\n".join(
            f"[{i+1}] (relevance={c['relevance']}) {c['text']}"
            for i, c in enumerate(chunks)
        )

        logger.info(f"Retrieved {len(chunks)} chunks, calling LLM ({LLM_PROVIDER})…")
        if LLM_PROVIDER == "openai":
            answer = _call_openai(question, context)
        else:
            answer = _call_ollama(question, context)

        return {
            "answer":  answer,
            "sources": chunks,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine: Optional[SalesRAGEngine] = None

def get_engine() -> SalesRAGEngine:
    global _engine
    if _engine is None:
        _engine = SalesRAGEngine()
    return _engine


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = get_engine()
    sample_questions = [
        "What were the total sales for the West region?",
        "Which product category has the highest profit margin?",
        "Show me the top 3 customers by sales amount.",
        "What is the average discount given in the Technology category?",
    ]
    for q in sample_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = engine.ask(q)
        print(f"A: {result['answer']}")
