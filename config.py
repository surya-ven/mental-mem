# config.py

import os

# --- API Keys ---
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# --- Model Definitions for the Study (Using OpenRouter Identifiers) ---

# Models used for high-level reasoning tasks (dataset generation, reflection, evaluation)
DATA_GEN_MODEL = "openai/gpt-4.1-mini"
REFLECTOR_MODEL = "openai/gpt-4.1"
JUDGE_MODEL = "openai/gpt-4.1"

# "Local-runnable" open model (called via OpenRouter)
LOCAL_MODEL_NAME = "google/gemma-3-12b-it"

# Closed-source model being tested in the evaluation cohort
CLOSED_MODEL_NAME = "openai/gpt-4.1"

# --- NEW: Mem0 Configuration to use OpenRouter for its internal LLM & Embedder ---
# This tells mem0's default "openai" provider for LLM and Embedder to use OpenRouter.

# Embedding model available on OpenRouter (OpenAI's text-embedding-3-small is a common one)
# Its dimension is 1536.
OPENAI_EMBEDDING_MODEL_FOR_MEM0 = "text-embedding-3-small"
EMBEDDING_MODEL_DIMS = 1536

# A model for mem0's internal inference tasks (if infer=True is used).
# This should be an OpenAI API compatible model available on OpenRouter.
# GPT-4o-Mini is a good default if available and cost-effective.
MEM0_INTERNAL_LLM_MODEL = "gpt-4o-mini"


MEM0_CONFIG = {
    "vector_store": {
        # Default, ensure Qdrant is running or change to "chroma" for in-memory
        "provider": "qdrant",
        "config": {
            "host": "localhost",        # Default Qdrant host
            "port": 6333,             # Default Qdrant gRPC port
            "embedding_model_dims": EMBEDDING_MODEL_DIMS,
            # "collection_name": "mem0_collection" # Optional: specify a collection name
            # For a persistent Qdrant, ensure your Qdrant server is configured for persistence.
            # If running Qdrant in Docker, use a volume mount for the /qdrant/storage directory.
        }
    },
    "llm": {  # For mem0's internal summarization/inference (if infer_memories=True on add())
        "provider": "openai",  # Use mem0's built-in OpenAI LLM provider
        "config": {
            "model": MEM0_INTERNAL_LLM_MODEL,
            "api_key": OPENAI_API_KEY,       # Pass OpenRouter API key
        }
    },
    "embedder": {  # For creating embeddings for memories
        "provider": "openai",  # Use mem0's built-in OpenAI Embedder provider
        "config": {
            "model": OPENAI_EMBEDDING_MODEL_FOR_MEM0,
            "api_key": OPENAI_API_KEY,       # Pass OpenAI API key
        }
    }
    # Optional: history_db_path if you don't want the default ./mem0_history.db
    # "history_db_path": "./mem0_data/history.db"
}

LOG_DIR = os.path.join("output", "run_logs")
