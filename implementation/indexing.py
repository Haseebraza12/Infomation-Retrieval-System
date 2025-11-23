"""
Indexing Module - Builds BM25 and dense retrieval indices
"""

import pickle
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import bm25s
from sentence_transformers import SentenceTransformer

from config import Config
from utils import logger, timer, save_pickle, load_pickle

config = Config()


class BM25Indexer:
    """BM25 index builder using bm25s library"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        self.index = None
        
    @timer
    def build_index(self, processed_articles: List[Dict]):
        """
        Build BM25 index from processed articles
        
        Args:
            processed_articles: List of preprocessed articles with tokens
        """
        logger.info(f"Building BM25 index for {len(processed_articles)} articles...")
        
        # Prepare corpus as raw text (bm25s will tokenize internally)
        corpus_texts = []
        for article in tqdm(processed_articles, desc="Preparing corpus"):
            # Create text by repeating title for boosting
            title = article['title']
            content = article['content']
            
            # Repeat title for boosting
            boosted_text = ' '.join([title] * self.config.BM25_TITLE_BOOST) + ' ' + content
            corpus_texts.append(boosted_text)
        
        # Tokenize corpus with bm25s
        logger.info("Tokenizing corpus for BM25...")
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords='en', show_progress=True)
        
        # Build index
        logger.info("Building BM25 index...")
        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)
        
        logger.info(f"BM25 index built with {len(corpus_texts)} documents")
        
        return self.index
    
    def save_index(self):
        """Save BM25 index to disk"""
        index_path = self.config.INDEX_DIR / "bm25_index"
        index_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving BM25 index to {index_path}")
        self.index.save(str(index_path))
        logger.info("BM25 index saved successfully")
    
    def load_index(self):
        """Load BM25 index from disk"""
        index_path = self.config.INDEX_DIR / "bm25_index"
        
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {index_path}")
        
        logger.info(f"Loading BM25 index from {index_path}")
        self.index = bm25s.BM25.load(str(index_path), mmap=True)
        logger.info("BM25 index loaded successfully")
        
        return self.index


class DenseEmbeddingIndexer:
    """Dense embedding index builder using sentence-transformers"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        self.model = None
        self.embeddings = None
        
    @timer
    def build_index(self, processed_articles: List[Dict]):
        """
        Build dense embedding index
        
        Args:
            processed_articles: List of preprocessed articles
        """
        logger.info(f"Building dense embeddings for {len(processed_articles)} articles...")
        
        # Load model
        logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Prepare texts (title + snippet of content)
        texts = []
        for article in tqdm(processed_articles, desc="Preparing texts"):
            title = article['title']
            content = article['content'][:500]  # First 500 chars
            text = f"{title}. {content}"
            texts.append(text)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.config.BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        logger.info(f"Dense embeddings created: shape={self.embeddings.shape}")
        
        return self.embeddings
    
    def save_index(self):
        """Save embeddings to disk"""
        embeddings_path = self.config.INDEX_DIR / "dense_embeddings.npy"
        
        logger.info(f"Saving dense embeddings to {embeddings_path}")
        np.save(embeddings_path, self.embeddings)
        logger.info("Dense embeddings saved successfully")
    
    def load_index(self):
        """Load embeddings from disk"""
        embeddings_path = self.config.INDEX_DIR / "dense_embeddings.npy"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Dense embeddings not found at {embeddings_path}")
        
        logger.info(f"Loading dense embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        logger.info(f"Dense embeddings loaded: shape={self.embeddings.shape}")
        
        # Load model for query encoding
        self.model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        return self.embeddings


class MetadataIndexer:
    """SQLite-based metadata indexer for fast lookups"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        self.db_path = self.config.INDEX_DIR / "metadata.db"
        self.conn = None
        
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Drop existing tables to start fresh
        cursor.execute('DROP TABLE IF EXISTS entities')
        cursor.execute('DROP TABLE IF EXISTS articles')
        
        # Articles table
        cursor.execute('''
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                date TEXT,
                parsed_date TEXT,
                category TEXT,
                doc_length INTEGER
            )
        ''')
        
        # Entities table
        cursor.execute('''
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                text TEXT,
                label TEXT,
                confidence REAL,
                in_title INTEGER,
                FOREIGN KEY (article_id) REFERENCES articles(id)
            )
        ''')
        
        # Create indices for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_article_category ON articles(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_article_date ON articles(parsed_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_article ON entities(article_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_text ON entities(text)')
        
        self.conn.commit()
    
    @timer
    def build_index(self, processed_articles: List[Dict]):
        """
        Build metadata index
        
        Args:
            processed_articles: List of preprocessed articles
        """
        logger.info(f"Building metadata index for {len(processed_articles)} articles...")
        
        # Remove old database if exists
        if self.db_path.exists():
            logger.info(f"Removing old database: {self.db_path}")
            self.db_path.unlink()
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        
        cursor = self.conn.cursor()
        
        # Insert articles and entities
        for article in tqdm(processed_articles, desc="Indexing metadata"):
            # Convert parsed_date to string if it exists
            parsed_date_str = None
            if article.get('parsed_date'):
                try:
                    parsed_date_str = str(article['parsed_date'])
                except:
                    pass
            
            # Insert article
            cursor.execute('''
                INSERT INTO articles (id, title, content, date, parsed_date, category, doc_length)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['id'],
                article['title'],
                article['content'],
                article['date'],
                parsed_date_str,
                article['category'],
                article['doc_length']
            ))
            
            # Insert entities
            for entity in article.get('entities', []):
                cursor.execute('''
                    INSERT INTO entities (article_id, text, label, confidence, in_title)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    article['id'],
                    entity['text'],
                    entity['label'],
                    entity['confidence'],
                    1 if entity.get('in_title', False) else 0
                ))
        
        self.conn.commit()
        logger.info("Metadata index built successfully")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class IndexBuilder:
    """Main index builder orchestrator"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        
    def build_all_indices(self, processed_articles: List[Dict]):
        """Build all indices"""
        logger.info("="*70)
        logger.info("BUILDING ALL INDICES")
        logger.info("="*70)
        
        # 1. Build BM25 index
        logger.info("\n1. Building BM25 Index...")
        bm25_indexer = BM25Indexer(self.config)
        bm25_indexer.build_index(processed_articles)
        bm25_indexer.save_index()
        
        # 2. Build dense embeddings
        logger.info("\n2. Building Dense Embeddings...")
        dense_indexer = DenseEmbeddingIndexer(self.config)
        dense_indexer.build_index(processed_articles)
        dense_indexer.save_index()
        
        # 3. Build metadata index
        logger.info("\n3. Building Metadata Index...")
        metadata_indexer = MetadataIndexer(self.config)
        metadata_indexer.build_index(processed_articles)
        metadata_indexer.close()
        
        logger.info("\n" + "="*70)
        logger.info("ALL INDICES BUILT SUCCESSFULLY")
        logger.info("="*70)
    
    def load_bm25_index(self):
        """Load BM25 index"""
        bm25_indexer = BM25Indexer(self.config)
        return bm25_indexer.load_index()
    
    def load_dense_embeddings(self):
        """Load dense embeddings"""
        dense_indexer = DenseEmbeddingIndexer(self.config)
        return dense_indexer.load_index()
    
    def load_metadata(self):
        """Load metadata (articles list)"""
        return load_pickle(self.config.PROCESSED_DATA_PATH)


def main():
    """Main indexing pipeline"""
    logger.info("Starting indexing pipeline...")
    
    cfg = Config()
    
    # Load processed articles
    logger.info(f"Loading processed articles from {cfg.PROCESSED_DATA_PATH}")
    
    if not cfg.PROCESSED_DATA_PATH.exists():
        logger.error("Preprocessed data not found. Please run preprocessing first.")
        return
    
    processed_articles = load_pickle(cfg.PROCESSED_DATA_PATH)
    logger.info(f"Loaded {len(processed_articles)} processed articles")
    
    # Build all indices
    builder = IndexBuilder(cfg)
    builder.build_all_indices(processed_articles)
    
    logger.info("\nIndexing complete!")


if __name__ == "__main__":
    main()
