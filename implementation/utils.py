"""
Utility functions for Cortex IR System
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List
from pathlib import Path
import json
import pickle

import config

# Setup logging
def setup_logging(name: str) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(handler)
    
    return logger

# Performance timing decorator
def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000  # Convert to ms
        logger = setup_logging(func.__module__)
        logger.info(f"{func.__name__} completed in {elapsed:.2f}ms")
        return result
    return wrapper

# File operations
def save_pickle(data: Any, filepath: str) -> None:
    """Save data to pickle file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger = setup_logging(__name__)
    logger.info(f"Saved data to {filepath}")

def load_pickle(filepath: str) -> Any:
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger = setup_logging(__name__)
    logger.info(f"Loaded data from {filepath}")
    return data

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    logger = setup_logging(__name__)
    logger.info(f"Saved JSON to {filepath}")

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger = setup_logging(__name__)
    logger.info(f"Loaded JSON from {filepath}")
    return data

# Text processing utilities
def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (keep alphanumeric, punctuation, and basic symbols)
    text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', text)
    return text.strip()

# Score normalization
def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
    """Normalize scores using specified method"""
    if not scores:
        return []
    
    if method == "minmax":
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    elif method == "zscore":
        import numpy as np
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return [0.0] * len(scores)
        return [(s - mean) / std for s in scores]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# Date parsing
def parse_date(date_string: str):
    """Parse date string to datetime object"""
    from datetime import datetime
    import pandas as pd
    
    try:
        return pd.to_datetime(date_string)
    except:
        return None

# Snippet generation
def generate_snippet(text: str, query: str, max_length: int = 200) -> str:
    """Generate a snippet from text highlighting query terms"""
    words = text.split()
    query_terms = query.lower().split()
    
    # Find position of first query term
    positions = []
    for i, word in enumerate(words):
        if any(term in word.lower() for term in query_terms):
            positions.append(i)
    
    if not positions:
        # No query terms found, return beginning
        snippet = ' '.join(words[:max_length // 5])
        return truncate_text(snippet, max_length)
    
    # Start from first occurrence
    start_pos = max(0, positions[0] - 10)
    end_pos = min(len(words), start_pos + max_length // 5)
    
    snippet = ' '.join(words[start_pos:end_pos])
    
    if start_pos > 0:
        snippet = "..." + snippet
    if end_pos < len(words):
        snippet = snippet + "..."
    
    return snippet

# Batch processing
def batch_iterator(items: List[Any], batch_size: int):
    """Yield batches from items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# Check if index exists
def index_exists(index_type: str) -> bool:
    """Check if specified index exists"""
    if index_type == "bm25":
        return Path(config.BM25_INDEX_PATH).exists()
    elif index_type == "colbert":
        return Path(config.COLBERT_INDEX_PATH).exists()
    elif index_type == "metadata":
        return Path(config.METADATA_DB_PATH).exists()
    elif index_type == "processed":
        return Path(config.PROCESSED_DATA_PATH).exists()
    else:
        return False

# Progress tracking
class ProgressTracker:
    """Track progress of operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        
    def get_eta(self) -> str:
        """Get estimated time remaining"""
        if self.current == 0:
            return "Unknown"
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        remaining = (self.total - self.current) / rate
        
        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining / 60:.0f}m"
        else:
            return f"{remaining / 3600:.1f}h"
    
    def __str__(self) -> str:
        pct = (self.current / self.total) * 100
        return f"{self.description}: {self.current}/{self.total} ({pct:.1f}%) - ETA: {self.get_eta()}"
