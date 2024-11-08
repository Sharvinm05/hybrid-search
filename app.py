"""Module for querying and re-ranking user prompts using embeddings and keyword search.

This module sets up a FastAPI application that provides an endpoint for querying
a knowledge base. It integrates with ChromaDB, Redis, and a ColBERT-based
transformer model for semantic search and re-ranking results.
Additional functionality includes caching and re-ranking search results
based on embedding similarity and keyword matches.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import aiohttp
import chromadb
import redis
import torch
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException, Query
from transformers import AutoModel, AutoTokenizer
from whoosh import scoring
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in
from whoosh.query import FuzzyTerm, Or, Term

# Configure logging
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Load configuration from JSON file
with Path("config.json").open() as config_file:
    config = json.load(config_file)

# Access credentials
CHROMA_DB_USERNAME = config["CHROMA_DB_USERNAME"]
CHROMA_DB_PASSWORD = config["CHROMA_DB_PASSWORD"]
CHROMA_DB_HOST = config["CHROMA_DB_HOST"]
CHROMA_DB_PORT = config["CHROMA_DB_PORT"]
EMBEDDING_MODEL_URL = config["EMBEDDING_MODEL_URL"]
EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL_NAME"]
COLLECTION_NAME = config["COLLECTION_NAME"]

# Redis setup for caching
cache = redis.StrictRedis(host="localhost", port=6379, db=0)

# Setting up ChromaDB client
credentials = f"{CHROMA_DB_USERNAME}:{CHROMA_DB_PASSWORD}"
settings = Settings(
    chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
    chroma_client_auth_credentials=credentials,
)
chroma_client = chromadb.HttpClient(
    host=CHROMA_DB_HOST,
    port=CHROMA_DB_PORT,
    settings=settings,
)
collection_name = ""
collection = chroma_client.get_collection(collection_name)

# Load the ColBERT tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")

SIMILARITY_THRESHOLD_PRESET = 0.4


async def get_embedding(user_input: str) -> list[float]:
    """Retrieve embedding from cache or external model."""
    cached_embedding = cache.get(user_input)
    if cached_embedding:
        return ast.literal_eval(cached_embedding.decode("utf-8"))

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                EMBEDDING_MODEL_URL,
                params={"promt": user_input, "model_name": EMBEDDING_MODEL_NAME},
            ) as response:
                response.raise_for_status()
                embedding = ast.literal_eval(await response.text())
                cache.set(user_input, str(embedding), ex=3600)
                return embedding
        except (aiohttp.ClientError, ValueError):
            logging.exception("Error fetching embedding")
            return []


async def vector_search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """Perform semantic vector search on the given query embedding."""
    try:
        raw_results = collection.query(query_embeddings=query_embedding, n_results=k)
        return [
            {
                "document": meta.get("table_name", "Unknown Document"),
                "score": dist,
            }
            for meta, dist in zip(
                raw_results.get("metadatas", [[]])[0],
                raw_results.get("distances", [[]])[0],
            )
        ]
    except ValueError:
        logging.exception("Error during vector search")
        return []


async def keyword_search(user_query: str, k: int = 5) -> list[dict]:
    """Perform keyword search using Whoosh without fuzzy matching."""
    try:
        chroma_docs = collection.get()
        schema = Schema(
            document=ID(stored=True),
            table_name=TEXT(stored=True),
            table_description=TEXT(stored=True),
            kpi_name=TEXT(stored=True),
            content=TEXT(stored=True),
        )
        with TemporaryDirectory() as index_dir:
            ix = create_in(index_dir, schema)
            writer = ix.writer()
            for meta in chroma_docs.get("metadatas", []):
                if meta is not None:
                    writer.add_document(
                        document=meta.get("table_name", "Unknown Document"),
                        table_name=meta.get("table_name", ""),
                        table_description=meta.get("table_description", ""),
                        kpi_name=meta.get("kpi_name", ""),
                        content=meta.get("table_description", ""),
                    )
            writer.commit()

            with ix.searcher(weighting=scoring.BM25F()) as searcher:
                query_terms = user_query.split()
                
                # Main query using exact term matching in the "content" field
                main_query = Or([Term("content", term) for term in query_terms])
                
                # Additional queries for exact matches in other fields
                sub_queries = [
                    Term(field, term)
                    for term in query_terms
                    for field in ["table_description", "kpi_name", "table_name"]
                ]

                combined_query = Or([main_query, *sub_queries])
                results = searcher.search(combined_query, limit=k)
                return [
                    {"document": hit["document"], "score": hit.score} for hit in results
                ]
    except ValueError:
        logging.exception("Error during keyword search")
        return []

def maxsim(
    query_embedding: torch.Tensor,
    document_embedding: torch.Tensor,
) -> torch.Tensor:
    """Compute MaxSim score for ColBERT re-ranking."""
    expanded_query = query_embedding.unsqueeze(2)
    expanded_doc = document_embedding.unsqueeze(1)
    sim_matrix = torch.nn.functional.cosine_similarity(
        expanded_query,
        expanded_doc,
        dim=-1,
    )
    return torch.mean(torch.max(sim_matrix, dim=2)[0], dim=1)


async def rerank_with_colbert(
    semantic_results: list[dict],
    user_query: str,
) -> list[dict]:
    """Re-rank semantic results using ColBERT embeddings and MaxSim."""
    query_encoding = tokenizer(user_query, return_tensors="pt")
    query_embedding = model(**query_encoding).last_hidden_state.mean(dim=1)

    scores = []
    for result in semantic_results:
        table_name_encoding = tokenizer(
            result.get("document", ""),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        table_description_encoding = tokenizer(
            result.get("table_description", ""),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        table_name_embedding = model(**table_name_encoding).last_hidden_state
        table_description_embedding = model(
            **table_description_encoding,
        ).last_hidden_state

        name_score = maxsim(query_embedding.unsqueeze(0), table_name_embedding)
        description_score = maxsim(
            query_embedding.unsqueeze(0),
            table_description_embedding,
        )

        final_score = 0.7 * name_score + 0.3 * description_score
        scores.append({"score": final_score.item(), "document": result["document"]})

    return sorted(scores, key=lambda x: x["score"], reverse=True)


def weighted_rerank(
    semantic_results: list[dict],
    keyword_results: list[dict],
    user_query: str,
) -> list[dict]:
    """Combine and re-rank results from semantic and keyword searches."""
    weight_semantic = 0.7
    doc_scores = {}

    for sem_result in semantic_results:
        doc_id = sem_result["document"]
        doc_scores[doc_id] = weight_semantic * sem_result["score"]

    for kw_result in keyword_results:
        doc_id = kw_result["document"]
        if doc_id in doc_scores:
            doc_scores[doc_id] += (1 - weight_semantic) * kw_result["score"]
        else:
            doc_scores[doc_id] = (1 - weight_semantic) * kw_result["score"]

    combined_results = [
        {"document": doc, "combined_score": score} for doc, score in doc_scores.items()
    ]
    return sorted(combined_results, key=lambda x: x["combined_score"], reverse=True)


async def query_kpi_knowledge_base(
    user_query: str,
    similarity_threshold: float,
) -> list[dict]:
    """Query KPIs/knowledge base with re-ranking."""
    embedded_query = await get_embedding(user_query)
    if not embedded_query:
        return []

    semantic_search_task = vector_search(query_embedding=embedded_query, k=10)
    keyword_search_task = keyword_search(user_query, k=10)

    try:
        semantic_results = await semantic_search_task
        keyword_results = await keyword_search_task

        reranked_semantic_results = await rerank_with_colbert(
            semantic_results,
            user_query,
        )
        reranked_results = weighted_rerank(
            reranked_semantic_results,
            keyword_results,
            user_query,
        )

        return [
            result
            for result in reranked_results
            if result.get("combined_score", 0) >= similarity_threshold
        ]

    except ValueError:
        logging.exception("Error during query execution")
        return []


@app.get("/query_kpi")
async def query_kpi_endpoint(
    user_query: str = Query(..., description="User query"),
) -> dict:
    """Provide API endpoint for querying the knowledge base with query parameters."""
    similarity_threshold = SIMILARITY_THRESHOLD_PRESET
    start_time = time.time()
    results = await query_kpi_knowledge_base(user_query, similarity_threshold)
    end_time = time.time()
    if not results:
        raise HTTPException(status_code=404, detail="No relevant results found")

    return {
        "execution_time": f"{end_time - start_time:.2f} seconds",
        "results": results,
    }
