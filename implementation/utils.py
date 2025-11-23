"""
Utility functions and helpers for Cortex IR System
"""

import logging
import sys
import pickle
import functools
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Callable
import numpy as np

from config import Config

# Initialize config
config = Config()

# Setup logging
def setup_logger(name: str = __name__, level: str = None) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger
    """
    if level is None:
        level = config.LOG_LEVEL
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = config.LOG_DIR / f"cortex_ir_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logger(__name__)


def load_stopwords() -> set:
    """Load English stopwords"""
    # Basic English stopwords
    stopwords = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
        'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
        'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'down',
        'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
        'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
        'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me',
        'might', 'more', 'most', 'must', 'my', 'myself', 'no', 'nor', 'not', 'now',
        'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves',
        'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than',
        'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
        'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
        'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while',
        'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yours',
        'yourself', 'yourselves'
    }
    return stopwords


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity score
    """
    if not set1 and not set2:
        return 1.0
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1] range using min-max normalization
    
    Args:
        scores: List of scores
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if min_score == max_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


def batch_data(data: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split data into batches
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches


def format_time_ms(time_seconds: float) -> str:
    """
    Format time in seconds to milliseconds string
    
    Args:
        time_seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "123.45ms")
    """
    return f"{time_seconds * 1000:.2f}ms"


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_snippet(content: str, query_tokens: List[str], window_size: int = 150) -> str:
    """
    Extract relevant snippet from content based on query tokens
    
    Args:
        content: Full content text
        query_tokens: List of query tokens
        window_size: Size of snippet window
        
    Returns:
        Extracted snippet
    """
    if not content or not query_tokens:
        return truncate_text(content, window_size)
    
    content_lower = content.lower()
    
    # Find first occurrence of any query token
    best_position = -1
    for token in query_tokens:
        pos = content_lower.find(token.lower())
        if pos != -1 and (best_position == -1 or pos < best_position):
            best_position = pos
    
    if best_position == -1:
        # No match found, return beginning
        return truncate_text(content, window_size)
    
    # Extract window around match
    start = max(0, best_position - window_size // 2)
    end = min(len(content), start + window_size)
    
    snippet = content[start:end]
    
    # Add ellipsis if needed
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    
    return snippet


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def calculate_entity_overlap(entities1: List[Dict], entities2: List[Dict]) -> float:
    """
    Calculate entity overlap between two entity lists
    
    Args:
        entities1: First entity list
        entities2: Second entity list
        
    Returns:
        Overlap score (0-1)
    """
    if not entities1 or not entities2:
        return 0.0
    
    # Extract entity texts
    set1 = {e['text'].lower() if isinstance(e, dict) else str(e).lower() for e in entities1}
    set2 = {e['text'].lower() if isinstance(e, dict) else str(e).lower() for e in entities2}
    
    return jaccard_similarity(set1, set2)


def get_file_size_mb(filepath: Path) -> float:
    """
    Get file size in megabytes
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in MB
    """
    if not filepath.exists():
        return 0.0
    
    return filepath.stat().st_size / (1024 * 1024)


def check_indices_exist(config: Config) -> Dict[str, bool]:
    """
    Check which indices exist
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with index existence status
    """
    return {
        'bm25': (config.INDEX_DIR / "bm25_index").exists(),
        'processed_data': config.PROCESSED_DATA_PATH.exists(),
        'metadata': (config.INDEX_DIR / "metadata.db").exists(),
        'dense_embeddings': (config.INDEX_DIR / "dense_embeddings.npy").exists()
    }


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object to pickle file
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.debug(f"Saved pickle to {filepath} ({get_file_size_mb(filepath):.2f} MB)")


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.debug(f"Loaded pickle from {filepath} ({get_file_size_mb(filepath):.2f} MB)")
    return obj


def timer(func: Callable) -> Callable:
    """
    Decorator for timing functions
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} completed in {format_time_ms(elapsed)}")
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {format_time_ms(self.elapsed)}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


# Export commonly used functions
__all__ = [
    'logger',
    'setup_logger',
    'load_stopwords',
    'cosine_similarity',
    'jaccard_similarity',
    'normalize_scores',
    'batch_data',
    'format_time_ms',
    'truncate_text',
    'extract_snippet',
    'merge_dicts',
    'calculate_entity_overlap',
    'get_file_size_mb',
    'check_indices_exist',
    'save_pickle',
    'load_pickle',
    'timer',
    'Timer',
    'safe_divide'
]
