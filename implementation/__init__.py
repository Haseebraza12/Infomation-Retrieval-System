"""
Package initialization for Cortex IR System
"""

__version__ = "1.0.0"
__author__ = "Cortex IR Team"
__description__ = "Advanced Hybrid Information Retrieval System for News Articles"

from .config import *
from .utils import setup_logging

# Initialize default logger
logger = setup_logging(__name__)
logger.info(f"Cortex IR System v{__version__} initialized")
