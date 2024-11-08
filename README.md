# Knowledge Base Query and Re-Ranking API 
### Hybrid Search [*Semantic + Keyword*] + Reranking

## Overview

This repository provides a FastAPI application for querying and re-ranking search results in a knowledge base. It integrates semantic and keyword search functionalities using embeddings for similarity and a ColBERT-based model for re-ranking. The application uses ChromaDB, Redis, and caching for optimized response times.

## Features

- **Semantic Search**: Uses vector embeddings to find similar results.
- **Keyword Search**: Utilizes Whoosh for precise keyword matches.
- **ColBERT Re-ranking**: Leverages ColBERT-based model for enhanced ranking.
- **Caching**: Redis caching for faster retrieval of embeddings.
- **API Endpoint**: `/query_kpi` endpoint for querying the knowledge base with re-ranked results based on similarity.

## Technologies Used

- **FastAPI**: API framework.
- **ChromaDB**: Manages and queries the knowledge base data.
- **Redis**: Embedding caching.
- **ColBERT**: Re-ranking based on semantic similarity.
- **Whoosh**: Keyword search.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Redis server (running on localhost)
- ChromaDB credentials
- ColBERT model (e.g., `colbert-ir/colbertv2.0` for embeddings)

### Configuration

1. **Create a Configuration File**:
   Set up a `config.json` file in the root directory with the following structure:
   ```json
   {
     "CHROMA_DB_USERNAME": "your_chromadb_username",
     "CHROMA_DB_PASSWORD": "your_chromadb_password",
     "CHROMA_DB_HOST": "your_chromadb_host",
     "CHROMA_DB_PORT": "your_chromadb_port",
     "EMBEDDING_MODEL_URL": "your_embedding_model_url",
     "EMBEDDING_MODEL_NAME": "your_embedding_model_name"
   }

2. **Redis**: Ensure Redis is running on localhost:6379.

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
### Running the Application
Start the FastAPI application:
    ```bash
    uvicorn main:app --reload
    ```

The API will be available at `http://127.0.0.1:8000`.


## API Usage
`/query_kpi` **Endpoint**

Query the knowledge base using a user prompt.

**Parameters**:
- `user_query`: The query text (string).

**Response**:
- `execution_time`: Time taken to process the query.
- `results`: List of matched documents and their scores.

### Example Request:
```bash
curl -X GET "http://127.0.0.1:8000/query_kpi?user_query=example+query"

```

or

Use `http://127.0.0.1:8000/docs` to use the **Swagger UI**

### Example Response:

```json
{
  "execution_time": "0.42 seconds",
  "results": [
    {
      "document": "Document1",
      "combined_score": 0.85
    },
    {
      "document": "Document2",
      "combined_score": 0.78
    }
  ]
}

```

## Code Structure
- get_embedding: Retrieves the embedding for a query from cache or the model.
- vector_search: Executes a vector search in ChromaDB based on embeddings.
- keyword_search: Uses Whoosh to perform keyword-based searches.
- rerank_with_colbert: Re-ranks results using the ColBERT model.
- weighted_rerank: Combines semantic and keyword results with weighted scores.
- query_kpi_knowledge_base: Coordinates the complete search and re-ranking - workflow.

## Disclaimer

This project was created for fun! Feel free to use, modify, or share it as you like – enjoy experimenting and learning!
