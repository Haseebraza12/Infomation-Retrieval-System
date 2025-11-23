"""
Configuration module for Cortex IR System
Loads environment variables and provides centralized configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
INDEXES_DIR = BASE_DIR / "indexes"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
INDEXES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Dataset Configuration
DATA_PATH = os.getenv("DATA_PATH", str(ROOT_DIR / "Articles.csv"))

# Index Paths
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", str(INDEXES_DIR / "bm25_index"))
COLBERT_INDEX_PATH = os.getenv("COLBERT_INDEX_PATH", str(INDEXES_DIR / "colbert_index"))
METADATA_DB_PATH = os.getenv("METADATA_DB_PATH", str(INDEXES_DIR / "metadata.db"))
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", str(INDEXES_DIR / "processed_articles.pkl"))
TOPIC_MODEL_PATH = os.getenv("TOPIC_MODEL_PATH", str(MODELS_DIR / "topic_model"))

# Model Configuration
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# BM25 Parameters
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
TITLE_BOOST = float(os.getenv("TITLE_BOOST", "3.0"))
ENTITY_BOOST = float(os.getenv("ENTITY_BOOST", "2.0"))

# Retrieval Parameters
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "100"))
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "100"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "20"))

# Reranking Parameters
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "32"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "400"))

# Post-processing Parameters
DIVERSITY_LAMBDA_FACTUAL = float(os.getenv("DIVERSITY_LAMBDA_FACTUAL", "0.9"))
DIVERSITY_LAMBDA_EXPLORATORY = float(os.getenv("DIVERSITY_LAMBDA_EXPLORATORY", "0.6"))
ENTITY_CONFIDENCE_THRESHOLD = float(os.getenv("ENTITY_CONFIDENCE_THRESHOLD", "0.85"))
ENTITY_SIMILARITY_THRESHOLD = float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.8"))

# UI Configuration
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "False").lower() == "true"

# Performance
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"

# Query Types
QUERY_TYPES = {
    "BREAKING": "breaking",
    "HISTORICAL": "historical",
    "FACTUAL": "factual",
    "ANALYTICAL": "analytical"
}

# Categories
CATEGORIES = ["Business", "Sports"]

# Temporal Intelligence Settings
TEMPORAL_SETTINGS = {
    "breaking": {"recency_days": 7, "boost_factor": 2.0},
    "historical": {"date_matching": True, "tolerance_days": 30},
    "analytical": {"temporal_bias": False},
    "factual": {"recency_days": 30, "boost_factor": 1.5}
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
