"""
Metadata Manager Module
Handles filtering of results by category, date, and entities using SQLite
"""

import sqlite3
from typing import List, Dict, Optional, Union, Set
from pathlib import Path
from datetime import datetime

from config import Config
from utils import logger, timer

class MetadataManager:
    """
    Manages metadata filtering using SQLite
    """
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        self.db_path = self.config.INDEX_DIR / "metadata.db"
        
    def _get_connection(self):
        """Get SQLite connection with row factory"""
        if not self.db_path.exists():
            logger.error(f"Metadata database not found at {self.db_path}")
            return None
            
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def get_all_doc_ids(self) -> Set[int]:
        """Get all document IDs"""
        conn = self._get_connection()
        if not conn:
            return set()
            
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM articles")
            return set(row['id'] for row in cursor.fetchall())
        finally:
            conn.close()
            
    def filter_by_category(self, category: str) -> Set[int]:
        """
        Filter documents by category
        
        Args:
            category: Category name (e.g. "Business", "Sports")
            
        Returns:
            Set of document IDs
        """
        if not category or category.lower() == "all":
            return self.get_all_doc_ids()
            
        conn = self._get_connection()
        if not conn:
            return set()
            
        try:
            cursor = conn.cursor()
            print(f"DEBUG: Filtering category: '{category.lower()}'")
            cursor.execute("SELECT id FROM articles WHERE LOWER(category) = ?", (category.lower(),))
            res = cursor.fetchall()
            print(f"DEBUG: Found {len(res)} docs")
            return set(row['id'] for row in res)
        finally:
            conn.close()
            
    def filter_by_date_range(self, start_date: str = None, end_date: str = None) -> Set[int]:
        """
        Filter documents by date range
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Set of document IDs
        """
        if not start_date and not end_date:
            return self.get_all_doc_ids()
            
        conn = self._get_connection()
        if not conn:
            return set()
            
        try:
            query = "SELECT id FROM articles WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND parsed_date >= ?"
                params.append(start_date)
                
            if end_date:
                query += " AND parsed_date <= ?"
                params.append(end_date)
                
            cursor = conn.cursor()
            cursor.execute(query, params)
            return set(row['id'] for row in cursor.fetchall())
        finally:
            conn.close()
            
    def filter_by_entity(self, entity_text: str) -> Set[int]:
        """
        Filter documents by entity
        
        Args:
            entity_text: Entity text to match (partial match supported)
            
        Returns:
            Set of document IDs
        """
        if not entity_text:
            return self.get_all_doc_ids()
            
        conn = self._get_connection()
        if not conn:
            return set()
            
        try:
            cursor = conn.cursor()
            # Use LIKE for partial matching
            cursor.execute("SELECT DISTINCT article_id FROM entities WHERE text LIKE ?", (f"%{entity_text}%",))
            return set(row['article_id'] for row in cursor.fetchall())
        finally:
            conn.close()
            
    def get_metadata(self, doc_ids: List[int]) -> Dict[int, Dict]:
        """
        Get metadata for specific documents
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Dictionary mapping doc_id to metadata dict
        """
        if not doc_ids:
            return {}
            
        conn = self._get_connection()
        if not conn:
            return {}
            
        try:
            # SQLite has a limit on variables, so chunk if necessary
            # For now assuming reasonable size
            placeholders = ','.join(['?'] * len(doc_ids))
            query = f"SELECT * FROM articles WHERE id IN ({placeholders})"
            
            cursor = conn.cursor()
            cursor.execute(query, list(doc_ids))
            
            results = {}
            for row in cursor.fetchall():
                results[row['id']] = dict(row)
                
            return results
        finally:
            conn.close()

if __name__ == "__main__":
    # Test
    manager = MetadataManager()
    
    cats = manager.filter_by_category("Business")
    print(f"Business docs: {len(cats)}")
    
    dates = manager.filter_by_date_range("2023-01-01", "2023-12-31")
    print(f"2023 docs: {len(dates)}")
