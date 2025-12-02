"""
Configuration Module for Cortex IR System
Contains all system parameters and paths
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "implementation" / "indexes"
LOG_DIR = BASE_DIR / "implementation" / "logs"
MODEL_DIR = BASE_DIR / "implementation" / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, INDEX_DIR, LOG_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data paths
DATA_PATH = BASE_DIR / "Articles.csv"
PROCESSED_DATA_PATH = INDEX_DIR / "processed_articles.pkl"

# Model configurations
SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # For neural reranking

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75
BM25_TITLE_BOOST = 3  # Repeat title tokens 3x for boosting

# Retrieval parameters
TOP_K_RETRIEVAL = 100
NUM_DOCS_TO_RERANK = 20 # Limit input to reranker for speed
TOP_K_RERANK = 20  # Reduced for faster processing (was 50)
TOP_K_FINAL = 10
RRF_K = 60  # Reciprocal Rank Fusion constant

# Reranking parameters
RERANK_BATCH_SIZE = 32  # Batch size for neural reranking
RERANK_TOP_K = 50  # Number of documents to rerank

# Entity extraction
ENTITY_CONFIDENCE_THRESHOLD = 0.85
ENTITY_BOOST = 1.2  # Boost factor for entity matches

# Post-processing - Base parameter
MMR_LAMBDA = 0.7  # Balance between relevance and diversity (default)

# Query-type specific diversity parameters
DIVERSITY_LAMBDA_BREAKING = 0.3      # High diversity for breaking news
DIVERSITY_LAMBDA_FACTUAL = 0.5       # Medium diversity for factual queries
DIVERSITY_LAMBDA_ANALYTICAL = 0.7    # Higher diversity for analytical queries
DIVERSITY_LAMBDA_HISTORICAL = 0.6    # Medium-high diversity for historical queries
DIVERSITY_LAMBDA_EXPLORATORY = 0.8   # Highest diversity for exploratory queries
DIVERSITY_LAMBDA_DEFAULT = 0.6       # Default fallback value

# Temporal parameters
TEMPORAL_DECAY_DAYS = 365  # Days for temporal decay
TEMPORAL_BOOST_BREAKING = 2.0  # Boost recent articles for breaking news
TEMPORAL_BOOST_FACTUAL = 1.0   # No temporal boost for factual queries
TEMPORAL_BOOST_DEFAULT = 1.2   # Default temporal boost

# Content processing
MAX_CONTENT_LENGTH = 512  # Maximum tokens for content processing
MAX_QUERY_LENGTH = 128    # Maximum tokens for query

# Deduplication
ENTITY_SIMILARITY_THRESHOLD = 0.7  # Threshold for entity-based deduplication

# Dense retrieval (alternative to ColBERT)
USE_DENSE_RETRIEVAL = True  # Use sentence-transformers instead of ColBERT
DENSE_TOP_K = 100  # Number of results from dense retrieval

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance
BATCH_SIZE = 32
NUM_WORKERS = 4


@dataclass
class Config:
    """Configuration class with all system parameters"""
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    INDEX_DIR: Path = INDEX_DIR
    LOG_DIR: Path = LOG_DIR
    MODEL_DIR: Path = MODEL_DIR
    DATA_PATH: Path = DATA_PATH
    PROCESSED_DATA_PATH: Path = PROCESSED_DATA_PATH
    
    # Models
    SPACY_MODEL: str = SPACY_MODEL
    EMBEDDING_MODEL: str = EMBEDDING_MODEL
    RERANKER_MODEL: str = RERANKER_MODEL
    CROSS_ENCODER_MODEL: str = CROSS_ENCODER_MODEL
    
    # BM25
    BM25_K1: float = BM25_K1
    BM25_B: float = BM25_B
    BM25_TITLE_BOOST: int = BM25_TITLE_BOOST
    
    # Retrieval
    TOP_K_RETRIEVAL: int = TOP_K_RETRIEVAL
    NUM_DOCS_TO_RERANK: int = NUM_DOCS_TO_RERANK
    TOP_K_RERANK: int = TOP_K_RERANK
    TOP_K_FINAL: int = TOP_K_FINAL
    RRF_K: int = RRF_K
    
    # Reranking
    RERANK_BATCH_SIZE: int = RERANK_BATCH_SIZE
    RERANK_TOP_K: int = RERANK_TOP_K
    
    # Entity extraction
    ENTITY_CONFIDENCE_THRESHOLD: float = ENTITY_CONFIDENCE_THRESHOLD
    ENTITY_BOOST: float = ENTITY_BOOST
    
    # Post-processing
    MMR_LAMBDA: float = MMR_LAMBDA
    DIVERSITY_LAMBDA_BREAKING: float = DIVERSITY_LAMBDA_BREAKING
    DIVERSITY_LAMBDA_FACTUAL: float = DIVERSITY_LAMBDA_FACTUAL
    DIVERSITY_LAMBDA_ANALYTICAL: float = DIVERSITY_LAMBDA_ANALYTICAL
    DIVERSITY_LAMBDA_HISTORICAL: float = DIVERSITY_LAMBDA_HISTORICAL
    DIVERSITY_LAMBDA_EXPLORATORY: float = DIVERSITY_LAMBDA_EXPLORATORY
    DIVERSITY_LAMBDA_DEFAULT: float = DIVERSITY_LAMBDA_DEFAULT
    
    # Temporal
    TEMPORAL_DECAY_DAYS: int = TEMPORAL_DECAY_DAYS
    TEMPORAL_BOOST_BREAKING: float = TEMPORAL_BOOST_BREAKING
    TEMPORAL_BOOST_FACTUAL: float = TEMPORAL_BOOST_FACTUAL
    TEMPORAL_BOOST_DEFAULT: float = TEMPORAL_BOOST_DEFAULT
    
    # Content processing
    MAX_CONTENT_LENGTH: int = MAX_CONTENT_LENGTH
    MAX_QUERY_LENGTH: int = MAX_QUERY_LENGTH
    
    # Deduplication
    ENTITY_SIMILARITY_THRESHOLD: float = ENTITY_SIMILARITY_THRESHOLD
    
    # Dense retrieval
    USE_DENSE_RETRIEVAL: bool = USE_DENSE_RETRIEVAL
    DENSE_TOP_K: int = DENSE_TOP_K
    
    # Logging
    LOG_LEVEL: str = LOG_LEVEL
    LOG_FORMAT: str = LOG_FORMAT
    
    # Performance
    BATCH_SIZE: int = BATCH_SIZE
    NUM_WORKERS: int = NUM_WORKERS
    
    def __post_init__(self):
        """Ensure all directories exist"""
        for directory in [self.DATA_DIR, self.INDEX_DIR, self.LOG_DIR, self.MODEL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
