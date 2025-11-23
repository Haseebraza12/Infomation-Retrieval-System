# News Article IR System - Complete Implementation Guide

## System Overview

**Project Name**: Cortex IR (or your chosen name)  
**Dataset**: News Articles (2000 articles from Kaggle)  
**Architecture**: 4-Stage Hybrid Pipeline with Neural Reranking  
**Target Performance**: ~230-350ms query latency on CPU  
**Hardware Requirements**: Core i5 6th Gen + 16GB RAM (Satisfied ✅)

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Installation & Setup](#installation--setup)
4. [Stage 0: Preprocessing & Indexing](#stage-0-preprocessing--indexing)
5. [Stage 1: Hybrid Retrieval](#stage-1-hybrid-retrieval)
6. [Stage 2: Neural Reranking](#stage-2-neural-reranking)
7. [Stage 3: Post-Processing](#stage-3-post-processing)
8. [Stage 4: Gradio Interface](#stage-4-gradio-interface)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Implementation Checklist](#implementation-checklist)
11. [Troubleshooting](#troubleshooting)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0: Advanced Preprocessing & Multi-Index Construction     │
├─────────────────────────────────────────────────────────────────┤
│ Preprocessing Pipeline (Parallel):                              │
│  • Tokenization + Lemmatization (spaCy pipeline)                │
│  • NER with Confidence Filtering (threshold > 0.85)             │
│  • Field Extraction: Title, Content, Date, Category            │
│  • Query-adaptive Stopword Removal                              │
│                                                                  │
│ Multi-Index Construction (Offline, One-Time):                   │
│  ├─ SPARSE INDEX: BM25+ with compressed inverted index         │
│  │   • bm25s library (500× faster than rank-bm25)              │
│  │   • Positional info for phrase queries                       │
│  │   • Field-separated indices (title/content)                  │
│  │                                                               │
│  ├─ DENSE INDEX: ColBERTv2 token embeddings (PLAID optimized)  │
│  │   • RAGatouille wrapper for easy implementation             │
│  │   • 2-bit residual compression                               │
│  │   • Memory-mapped storage for efficiency                     │
│  │                                                               │
│  └─ METADATA STORE: SQLite with B-tree indexing                │
│      • Doc length, entity list, topics, temporal info           │
│      • Fast filtering by category/date/entity                   │
│                                                                  │
│ Time: ~4-6 min for 2000 articles (one-time cost)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Parallel Hybrid Retrieval (~60-80ms)                 │
├─────────────────────────────────────────────────────────────────┤
│ Query Processing:                                                │
│  • Query classification (TinyLlama 1.1B): Historical/Breaking/  │
│    Analytical/Factual                                           │
│  • Adaptive query expansion (PRF for analytical, entity for     │
│    factual)                                                      │
│  • Spell correction + entity disambiguation                     │
│                                                                  │
│ PARALLEL RETRIEVAL (executed concurrently with ThreadPool):     │
│                                                                  │
│  ┌─────────────────┐          ┌────────────────────┐           │
│  │ Sparse Branch   │          │ Dense Branch        │           │
│  │                 │          │                     │           │
│  │ BM25+ Tuned:    │          │ ColBERTv2 PLAID:   │           │
│  │ • k1=1.5,b=0.75 │          │ • Late interaction  │           │
│  │ • Title boost:  │          │ • Centroid pruning  │           │
│  │   3.0           │          │ • CPU optimized     │           │
│  │ • Entity boost: │          │                     │           │
│  │   2.0           │          │                     │           │
│  │ • Recency       │          │ Fast: ~40-60ms     │           │
│  │   (adaptive)    │          │                     │           │
│  │                 │          │                     │           │
│  │ Fast: ~10-15ms  │          │                     │           │
│  │ Top 100 docs    │          │ Top 100 docs        │           │
│  └────────┬────────┘          └─────────┬───────────┘           │
│           │                              │                       │
│           └──────────┬───────────────────┘                       │
│                      ↓                                           │
│           Reciprocal Rank Fusion (RRF)                          │
│           • k=60 (tuned parameter)                              │
│           • Handles score normalization automatically           │
│           • Deduplication built-in                              │
│                      ↓                                           │
│           Unified Top 100 candidates (reduced from 150)         │
│                                                                  │
│ Time: ~60-80ms total (parallel execution)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Fast Neural Reranking (~200-280ms)                   │
├─────────────────────────────────────────────────────────────────┤
│ Lightweight Cross-Encoder with Optimizations:                   │
│                                                                  │
│ Model: cross-encoder/ms-marco-MiniLM-L-6-v2                    │
│ Optimizations Applied:                                          │
│  • INT8 Quantization (2× speedup, <1% accuracy loss)           │
│  • ONNX Runtime (1.5-2× speedup)                               │
│  • Batch processing (batch_size=32, optimal for CPU)           │
│  • Smart truncation: Title + first 400 tokens                  │
│                                                                  │
│ Processing:                                                      │
│  • Score 100 (query, article) pairs in batches                 │
│  • CPU inference (GPU optional if available)                    │
│  • Early stopping: Skip scoring if top-50 stable               │
│                                                                  │
│ Output: Top 50 reranked articles                               │
│ Time: ~200-280ms on CPU                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Intelligent Post-Processing (~40ms)                  │
├─────────────────────────────────────────────────────────────────┤
│ Multi-Objective Optimization:                                   │
│                                                                  │
│ 1. DIVERSITY-AWARE RERANKING (Learned MMR):                    │
│    • Train λ parameter on query type                            │
│    • Factual queries: λ=0.9 (high relevance)                   │
│    • Exploratory queries: λ=0.6 (balanced)                     │
│    • Use sentence embeddings for diversity (fast)              │
│    Time: ~20ms                                                  │
│                                                                  │
│ 2. TEMPORAL INTELLIGENCE:                                       │
│    • Query classification determines temporal strategy:         │
│      - Breaking news: Boost articles < 7 days old              │
│      - Historical: Match query date to article date            │
│      - Analytical: No temporal bias                            │
│    Time: ~3ms (metadata lookup)                                │
│                                                                  │
│ 3. CATEGORY BALANCING (Adaptive):                              │
│    • Only applied if query is cross-category                    │
│    • Use soft constraints (not forced 50-50)                   │
│    • Preserve top-3 regardless of category                     │
│    Time: ~3ms                                                   │
│                                                                  │
│ 4. ENTITY-BASED DEDUPLICATION (Improved):                      │
│    • Use high-confidence entities only (>0.85)                 │
│    • Jaccard similarity on entity sets > 0.8 → duplicate       │
│    • Keep higher-scored article                                │
│    Time: ~7ms                                                   │
│                                                                  │
│ 5. TOPIC CLUSTERING (Presentation):                            │
│    • Fast clustering with pre-computed topic labels            │
│    • Group top 30 results into 3-5 topic clusters             │
│    • Display with topic labels for UX                          │
│    Time: ~10ms                                                  │
│                                                                  │
│ Output: Final Top 20 results with topic clusters               │
│ Total Stage Time: ~43ms                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: Result Presentation (Gradio Interface)               │
├─────────────────────────────────────────────────────────────────┤
│ • Display results grouped by topic clusters                     │
│ • Show: Title, snippet, date, category, entities, relevance    │
│ • Interactive filters: date range, category, topic             │
│ • Visualizations: Performance charts, query analytics          │
│ • Log user clicks for future learning-to-rank                  │
└─────────────────────────────────────────────────────────────────┘

TOTAL QUERY LATENCY: ~303-403ms (CPU) | ~230-310ms (with GPU)
```

---

## Technology Stack

### Core Libraries

```python
# Information Retrieval
bm25s==0.1.8                    # Fast BM25 implementation (500× speedup)
ragatouille==0.0.7              # ColBERTv2 wrapper
sentence-transformers==2.2.2    # Embeddings and cross-encoder

# Optimization & Acceleration
onnxruntime==1.16.0             # Fast inference runtime
optimum==1.14.0                 # Model quantization (INT8)

# NLP & Preprocessing
spacy==3.7.2                    # Tokenization, lemmatization, NER
en-core-web-sm==3.7.1           # English language model

# Machine Learning
transformers==4.35.2            # TinyLlama for query classification
torch==2.1.1                    # PyTorch (CPU version sufficient)

# Topic Modeling
bertopic==0.16.0                # Topic clustering
umap-learn==0.5.5               # Dimensionality reduction

# Data & Storage
pandas==2.1.3                   # Data handling
numpy==1.26.2                   # Numerical operations
sqlite3                         # Metadata storage (built-in)

# Evaluation & Visualization
scikit-learn==1.3.2             # Metrics computation
matplotlib==3.8.2               # Plotting
seaborn==0.13.0                 # Statistical visualization
plotly==5.18.0                  # Interactive charts

# User Interface
gradio==4.7.1                   # Web UI

# Utilities
tqdm==4.66.1                    # Progress bars
python-dotenv==1.0.0            # Environment variables
```

### Installation Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip

# Install IR and NLP libraries
pip install bm25s sentence-transformers spacy transformers torch

# Download spaCy model
python -m spacy download en_core_web_sm

# Install optimization libraries
pip install onnxruntime optimum

# Install ColBERT (optional but recommended)
pip install ragatouille

# Install topic modeling
pip install bertopic umap-learn

# Install data science stack
pip install pandas numpy scikit-learn matplotlib seaborn plotly

# Install Gradio for UI
pip install gradio

# Install utilities
pip install tqdm python-dotenv

# Save requirements
pip freeze > requirements.txt
```

---

## Stage 0: Preprocessing & Indexing

### 0.1 Data Loading & Initial Exploration

```python
import pandas as pd
import numpy as np
from pathlib import Path

# Load dataset
data_path = Path("Articles.csv")
df = pd.read_csv(data_path)

print(f"Dataset size: {len(df)} articles")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample article:")
print(df.iloc[0])

# Expected columns: Title, Content, Date, Category (Business/Sports)
```

### 0.2 Preprocessing Pipeline

```python
import spacy
from spacy.language import Language
from typing import Dict, List, Tuple
import re

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_article(article: Dict) -> Dict:
    """
    Comprehensive preprocessing for single article.
    
    Args:
        article: Dict with 'title', 'content', 'date', 'category'
    
    Returns:
        Processed article with additional fields
    """
    # Extract fields
    title = article.get('Title', '')
    content = article.get('Content', '')
    date = article.get('Date', '')
    category = article.get('Category', '')
    
    # Process title
    title_doc = nlp(title)
    title_tokens = [token.lemma_.lower() for token in title_doc 
                   if not token.is_stop and not token.is_punct]
    
    # Process content (limit to first 5000 chars for efficiency)
    content_doc = nlp(content[:5000])
    content_tokens = [token.lemma_.lower() for token in content_doc 
                     if not token.is_stop and not token.is_punct]
    
    # Named Entity Recognition (confidence > 0.85)
    entities = []
    for ent in content_doc.ents:
        # spaCy doesn't provide confidence, but we can filter by type
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'MONEY']:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    
    # Calculate statistics
    doc_length = len(content_tokens)
    unique_terms = len(set(content_tokens))
    
    return {
        'id': article.get('id', None),
        'title': title,
        'content': content,
        'date': date,
        'category': category,
        'title_tokens': title_tokens,
        'content_tokens': content_tokens,
        'entities': entities,
        'doc_length': doc_length,
        'unique_terms': unique_terms,
        'title_length': len(title_tokens)
    }

# Process all articles
from tqdm import tqdm

processed_articles = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
    article = row.to_dict()
    article['id'] = idx
    processed = preprocess_article(article)
    processed_articles.append(processed)

# Save processed data
import pickle
with open('processed_articles.pkl', 'wb') as f:
    pickle.dump(processed_articles, f)

print(f"Preprocessed {len(processed_articles)} articles")
```

### 0.3 BM25 Index Construction

```python
import bm25s
import pickle

# Prepare corpus (title + content tokens)
corpus_tokens = []
for article in processed_articles:
    # Combine title (weighted) and content
    combined = article['title_tokens'] * 3 + article['content_tokens']
    corpus_tokens.append(combined)

# Build BM25 index with tuned parameters
retriever_bm25 = bm25s.BM25()
retriever_bm25.index(bm25s.tokenize(corpus_tokens, stopwords='en'))

# Save BM25 index
retriever_bm25.save("bm25_index", corpus=processed_articles)

print("BM25 index created and saved")
print(f"Index size: {Path('bm25_index').stat().st_size / (1024*1024):.2f} MB")
```

### 0.4 ColBERTv2 Index Construction

```python
from ragatouille import RAGPretrainedModel

# Initialize ColBERT model
colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Prepare documents for indexing
documents = []
doc_ids = []
for article in processed_articles:
    # Combine title and content (truncate to 512 tokens)
    doc_text = f"{article['title']} {article['content'][:2000]}"
    documents.append(doc_text)
    doc_ids.append(article['id'])

# Index with ColBERT (this takes 2-3 minutes for 2000 docs)
colbert.index(
    collection=documents,
    document_ids=doc_ids,
    index_name="news_colbert_index",
    max_document_length=512,
    split_documents=False
)

print("ColBERT index created")
print(f"Index location: .ragatouille/colbert/indexes/news_colbert_index/")
```

### 0.5 SQLite Metadata Store

```python
import sqlite3
import json
from datetime import datetime

# Create SQLite database
conn = sqlite3.connect('metadata.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    date TEXT,
    category TEXT,
    doc_length INTEGER,
    unique_terms INTEGER,
    title_length INTEGER,
    entities TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Insert articles
for article in tqdm(processed_articles, desc="Building metadata store"):
    cursor.execute('''
    INSERT INTO articles (id, title, content, date, category, doc_length, 
                         unique_terms, title_length, entities)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        article['id'],
        article['title'],
        article['content'],
        article['date'],
        article['category'],
        article['doc_length'],
        article['unique_terms'],
        article['title_length'],
        json.dumps(article['entities'])
    ))

# Create indices for fast filtering
cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON articles(category)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON articles(date)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_length ON articles(doc_length)')

conn.commit()
conn.close()

print("Metadata store created with indices")
```

### 0.6 Topic Modeling (BERTopic)

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Prepare documents for topic modeling
documents_for_topics = [article['content'][:1000] for article in processed_articles]

# Initialize BERTopic with optimized settings
vectorizer_model = CountVectorizer(stop_words='english', max_features=1000)
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",
    vectorizer_model=vectorizer_model,
    min_topic_size=10,
    nr_topics='auto'
)

# Fit topic model
topics, probs = topic_model.fit_transform(documents_for_topics)

# Save topic model
topic_model.save("topic_model")

# Add topics to processed articles
for idx, (topic, prob) in enumerate(zip(topics, probs)):
    processed_articles[idx]['topic_id'] = int(topic)
    processed_articles[idx]['topic_prob'] = float(prob)

# Save updated processed articles
with open('processed_articles_with_topics.pkl', 'wb') as f:
    pickle.dump(processed_articles, f)

print(f"Topic modeling complete. Found {len(set(topics))} topics")
print(topic_model.get_topic_info())
```

---

## Stage 1: Hybrid Retrieval

### 1.1 Query Classification (TinyLlama)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load or train query classifier
# For simplicity, we'll use keyword-based classification
# You can fine-tune TinyLlama if you have labeled queries

def classify_query(query: str) -> str:
    """
    Classify query into: Historical, Breaking, Analytical, Factual
    
    Simple keyword-based approach (can be replaced with TinyLlama)
    """
    query_lower = query.lower()
    
    # Historical indicators
    if any(word in query_lower for word in ['history', 'past', '2008', '2010', 'originated', 'began']):
        return 'Historical'
    
    # Breaking news indicators
    if any(word in query_lower for word in ['latest', 'recent', 'today', 'current', 'breaking', 'now']):
        return 'Breaking'
    
    # Factual indicators (who, what, when, where)
    if any(word in query_lower for word in ['who', 'what', 'when', 'where', 'how many']):
        return 'Factual'
    
    # Default: Analytical
    return 'Analytical'

# Test
print(classify_query("latest COVID updates"))  # Breaking
print(classify_query("causes of 2008 financial crisis"))  # Historical
print(classify_query("who won the super bowl?"))  # Factual
print(classify_query("impact of inflation on economy"))  # Analytical
```

### 1.2 Query Expansion (PRF)

```python
def expand_query_prf(query: str, top_docs: List[Dict], n_terms: int = 5) -> str:
    """
    Pseudo-Relevance Feedback query expansion.
    
    Args:
        query: Original query string
        top_docs: Top-k documents from initial retrieval
        n_terms: Number of expansion terms to add
    
    Returns:
        Expanded query string
    """
    from collections import Counter
    
    # Extract terms from top documents
    all_terms = []
    for doc in top_docs[:5]:  # Use top 5 for PRF
        all_terms.extend(doc['content_tokens'])
    
    # Count term frequencies
    term_freq = Counter(all_terms)
    
    # Get original query terms
    query_terms = set(query.lower().split())
    
    # Find top expansion terms (not in original query)
    expansion_terms = []
    for term, freq in term_freq.most_common(50):
        if term not in query_terms and len(term) > 3:
            expansion_terms.append(term)
            if len(expansion_terms) >= n_terms:
                break
    
    # Combine original query with expansion terms
    expanded = query + " " + " ".join(expansion_terms)
    return expanded
```

### 1.3 Parallel Hybrid Retrieval

```python
from concurrent.futures import ThreadPoolExecutor
import time

def bm25_retrieve(query: str, k: int = 100) -> List[Tuple[int, float]]:
    """BM25 sparse retrieval"""
    # Tokenize query
    query_tokens = bm25s.tokenize(query, stopwords='en')
    
    # Retrieve
    results, scores = retriever_bm25.retrieve(query_tokens, k=k)
    
    # Return (doc_id, score) tuples
    return [(int(doc_id), float(score)) for doc_id, score in zip(results[0], scores[0])]

def colbert_retrieve(query: str, k: int = 100) -> List[Tuple[int, float]]:
    """ColBERT dense retrieval"""
    # Search
    results = colbert.search(query, k=k, index_name="news_colbert_index")
    
    # Return (doc_id, score) tuples
    return [(result['document_id'], result['score']) for result in results]

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]], 
    k: int = 60
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion to combine multiple rankings.
    
    Args:
        rankings: List of [(doc_id, score), ...] from different retrievers
        k: RRF parameter (typically 60)
    
    Returns:
        Fused ranking as [(doc_id, fused_score), ...]
    """
    rrf_scores = {}
    
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by RRF score
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused

def hybrid_retrieve(query: str, query_type: str, k: int = 100) -> List[int]:
    """
    Parallel hybrid retrieval with query-adaptive strategies.
    
    Args:
        query: Query string
        query_type: One of ['Historical', 'Breaking', 'Analytical', 'Factual']
        k: Number of results to return
    
    Returns:
        List of doc_ids ranked by relevance
    """
    start_time = time.time()
    
    # Adaptive query expansion
    if query_type == 'Analytical':
        # For analytical queries, do PRF after initial BM25
        bm25_results = bm25_retrieve(query, k=20)
        top_docs = [processed_articles[doc_id] for doc_id, _ in bm25_results[:5]]
        expanded_query = expand_query_prf(query, top_docs, n_terms=5)
        query_for_retrieval = expanded_query
    else:
        query_for_retrieval = query
    
    # Parallel retrieval
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bm25 = executor.submit(bm25_retrieve, query_for_retrieval, k)
        future_colbert = executor.submit(colbert_retrieve, query_for_retrieval, k)
        
        bm25_results = future_bm25.result()
        colbert_results = future_colbert.result()
    
    # Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion([bm25_results, colbert_results], k=60)
    
    # Extract doc_ids
    doc_ids = [doc_id for doc_id, _ in fused_results[:k]]
    
    elapsed = time.time() - start_time
    print(f"Stage 1 (Hybrid Retrieval): {elapsed*1000:.1f}ms")
    
    return doc_ids
```

---

## Stage 2: Neural Reranking

### 2.1 Load Optimized Cross-Encoder

```python
from sentence_transformers import CrossEncoder
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Option 1: Standard cross-encoder (easier)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Option 2: Quantized with ONNX (faster, recommended)
def load_quantized_reranker():
    """Load INT8 quantized cross-encoder with ONNX Runtime"""
    model = ORTModelForSequenceClassification.from_pretrained(
        'cross-encoder/ms-marco-MiniLM-L-6-v2',
        export=True,
        provider='CPUExecutionProvider'
    )
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return model, tokenizer

# Use Option 1 for simplicity
print("Cross-encoder loaded")
```

### 2.2 Reranking Function

```python
def rerank_documents(
    query: str, 
    doc_ids: List[int], 
    top_k: int = 50
) -> List[Tuple[int, float]]:
    """
    Rerank documents using cross-encoder.
    
    Args:
        query: Query string
        doc_ids: List of candidate document IDs
        top_k: Number of top documents to return after reranking
    
    Returns:
        List of (doc_id, score) tuples sorted by relevance
    """
    start_time = time.time()
    
    # Prepare (query, document) pairs
    pairs = []
    valid_doc_ids = []
    
    for doc_id in doc_ids:
        article = processed_articles[doc_id]
        # Combine title + first 400 tokens of content
        doc_text = f"{article['title']} {article['content'][:2000]}"
        pairs.append((query, doc_text))
        valid_doc_ids.append(doc_id)
    
    # Batch scoring
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    
    # Combine doc_ids with scores
    results = list(zip(valid_doc_ids, scores))
    
    # Sort by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"Stage 2 (Neural Reranking): {elapsed*1000:.1f}ms")
    
    return results[:top_k]
```

---

## Stage 3: Post-Processing

### 3.1 MMR Diversity Reranking

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load sentence embedder for diversity
diversity_model = SentenceTransformer('all-MiniLM-L6-v2')

def mmr_rerank(
    doc_ids: List[int], 
    scores: List[float],
    query: str,
    lambda_param: float = 0.7,
    k: int = 20
) -> List[int]:
    """
    Maximal Marginal Relevance for diversity-aware reranking.
    
    Args:
        doc_ids: List of document IDs
        scores: Relevance scores from reranker
        query: Query string
        lambda_param: Trade-off between relevance and diversity (0-1)
        k: Number of results to return
    
    Returns:
        Reranked list of doc_ids
    """
    start_time = time.time()
    
    # Get document texts
    doc_texts = [processed_articles[doc_id]['content'][:500] for doc_id in doc_ids]
    
    # Encode documents
    doc_embeddings = diversity_model.encode(doc_texts, show_progress_bar=False)
    query_embedding = diversity_model.encode([query], show_progress_bar=False)[0]
    
    # Normalize scores to [0, 1]
    max_score = max(scores)
    min_score = min(scores)
    norm_scores = [(s - min_score) / (max_score - min_score + 1e-10) for s in scores]
    
    # MMR algorithm
    selected = []
    selected_embeddings = []
    remaining = list(range(len(doc_ids)))
    
    while len(selected) < k and remaining:
        mmr_scores = []
        
        for idx in remaining:
            # Relevance score
            relevance = norm_scores[idx]
            
            # Diversity score (max similarity to already selected)
            if selected_embeddings:
                similarities = cosine_similarity(
                    [doc_embeddings[idx]], 
                    selected_embeddings
                )[0]
                diversity = 1 - max(similarities)
            else:
                diversity = 1.0
            
            # MMR score
            mmr = lambda_param * relevance + (1 - lambda_param) * diversity
            mmr_scores.append((idx, mmr))
        
        # Select document with highest MMR score
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)
        selected_embeddings.append(doc_embeddings[best_idx])
        remaining.remove(best_idx)
    
    # Return reranked doc_ids
    reranked_ids = [doc_ids[idx] for idx in selected]
    
    elapsed = time.time() - start_time
    print(f"MMR Diversity: {elapsed*1000:.1f}ms")
    
    return reranked_ids
```

### 3.2 Temporal Intelligence

```python
from datetime import datetime, timedelta

def apply_temporal_boost(
    doc_ids: List[int], 
    query_type: str
) -> List[int]:
    """
    Apply temporal boost based on query type.
    
    Args:
        doc_ids: List of document IDs
        query_type: One of ['Historical', 'Breaking', 'Analytical', 'Factual']
    
    Returns:
        Reordered doc_ids with temporal boost applied
    """
    if query_type != 'Breaking':
        return doc_ids  # No temporal boost for non-breaking queries
    
    # For breaking news, boost recent articles (< 7 days)
    current_date = datetime.now()
    
    def get_recency_score(doc_id):
        article = processed_articles[doc_id]
        try:
            article_date = datetime.strptime(article['date'], '%Y-%m-%d')
            days_ago = (current_date - article_date).days
            
            if days_ago < 7:
                return 1.0  # Strong boost
            elif days_ago < 30:
                return 0.5  # Medium boost
            else:
                return 0.0  # No boost
        except:
            return 0.0
    
    # Sort by recency score
    scored_docs = [(doc_id, get_recency_score(doc_id)) for doc_id in doc_ids]
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc_id for doc_id, _ in scored_docs]
```

### 3.3 Entity-Based Deduplication

```python
def deduplicate_by_entities(doc_ids: List[int], threshold: float = 0.8) -> List[int]:
    """
    Remove near-duplicate articles based on entity overlap.
    
    Args:
        doc_ids: List of document IDs
        threshold: Jaccard similarity threshold for considering duplicates
    
    Returns:
        Filtered list of doc_ids with duplicates removed
    """
    def get_entity_set(doc_id):
        article = processed_articles[doc_id]
        entities = article.get('entities', [])
        # Use only high-confidence entities
        return set([ent['text'].lower() for ent in entities])
    
    def jaccard_similarity(set1, set2):
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    # Keep track of unique articles
    unique_ids = []
    unique_entity_sets = []
    
    for doc_id in doc_ids:
        entity_set = get_entity_set(doc_id)
        
        # Check against already selected
        is_duplicate = False
        for existing_set in unique_entity_sets:
            if jaccard_similarity(entity_set, existing_set) > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_ids.append(doc_id)
            unique_entity_sets.append(entity_set)
    
    print(f"Deduplication: {len(doc_ids)} → {len(unique_ids)} articles")
    return unique_ids
```

### 3.4 Topic Clustering

```python
def cluster_results(doc_ids: List[int], n_clusters: int = 5) -> Dict:
    """
    Cluster results into topics for presentation.
    
    Args:
        doc_ids: List of document IDs
        n_clusters: Number of topic clusters
    
    Returns:
        Dict mapping cluster_id to list of doc_ids
    """
    # Get pre-computed topics
    topics = [processed_articles[doc_id]['topic_id'] for doc_id in doc_ids]
    
    # Group by topic
    from collections import defaultdict
    clusters = defaultdict(list)
    
    for doc_id, topic_id in zip(doc_ids, topics):
        clusters[topic_id].append(doc_id)
    
    # Get top clusters by size
    top_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    
    result = {}
    for idx, (topic_id, docs) in enumerate(top_clusters):
        # Get topic label from BERTopic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            label = ", ".join([word for word, _ in topic_words[:3]])
        else:
            label = f"Topic {topic_id}"
        
        result[idx] = {
            'label': label,
            'doc_ids': docs
        }
    
    return result
```

### 3.5 Complete Post-Processing Pipeline

```python
def post_process(
    doc_ids: List[int],
    scores: List[float],
    query: str,
    query_type: str,
    k: int = 20
) -> Dict:
    """
    Complete post-processing pipeline.
    
    Returns:
        Dict with final results and metadata
    """
    start_time = time.time()
    
    # 1. MMR Diversity Reranking
    lambda_param = 0.9 if query_type == 'Factual' else 0.7
    diverse_ids = mmr_rerank(doc_ids, scores, query, lambda_param, k=50)
    
    # 2. Temporal Intelligence
    temporal_ids = apply_temporal_boost(diverse_ids, query_type)
    
    # 3. Entity Deduplication
    deduplicated_ids = deduplicate_by_entities(temporal_ids[:40], threshold=0.8)
    
    # 4. Take top k
    final_ids = deduplicated_ids[:k]
    
    # 5. Topic Clustering
    clusters = cluster_results(final_ids, n_clusters=5)
    
    elapsed = time.time() - start_time
    print(f"Stage 3 (Post-Processing): {elapsed*1000:.1f}ms")
    
    return {
        'doc_ids': final_ids,
        'clusters': clusters,
        'processing_time_ms': elapsed * 1000
    }
```

---

## Stage 4: Gradio Interface

### 4.1 Complete Search Function

```python
def search_articles(query: str, top_k: int = 20) -> Dict:
    """
    Complete end-to-end search pipeline.
    
    Args:
        query: User query string
        top_k: Number of results to return
    
    Returns:
        Dict with results and metadata
    """
    overall_start = time.time()
    
    # Stage 1: Query Classification
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # Stage 1: Hybrid Retrieval
    candidate_ids = hybrid_retrieve(query, query_type, k=100)
    
    # Stage 2: Neural Reranking
    reranked_results = rerank_documents(query, candidate_ids, top_k=50)
    reranked_ids = [doc_id for doc_id, _ in reranked_results]
    reranked_scores = [score for _, score in reranked_results]
    
    # Stage 3: Post-Processing
    final_results = post_process(reranked_ids, reranked_scores, query, query_type, k=top_k)
    
    # Format results for display
    formatted_results = []
    for doc_id in final_results['doc_ids']:
        article = processed_articles[doc_id]
        formatted_results.append({
            'id': doc_id,
            'title': article['title'],
            'snippet': article['content'][:300] + "...",
            'date': article['date'],
            'category': article['category'],
            'entities': [ent['text'] for ent in article['entities'][:5]],
            'topic': f"Topic {article.get('topic_id', 'N/A')}"
        })
    
    total_time = time.time() - overall_start
    print(f"\n=== TOTAL QUERY TIME: {total_time*1000:.1f}ms ===\n")
    
    return {
        'query': query,
        'query_type': query_type,
        'results': formatted_results,
        'clusters': final_results['clusters'],
        'total_time_ms': total_time * 1000,
        'num_results': len(formatted_results)
    }
```

### 4.2 Gradio UI

```python
import gradio as gr
import pandas as pd

def format_results_for_display(search_results: Dict) -> Tuple[pd.DataFrame, str, str]:
    """Format search results for Gradio display"""
    
    # Create DataFrame for results table
    results_data = []
    for idx, result in enumerate(search_results['results'], 1):
        results_data.append({
            'Rank': idx,
            'Title': result['title'],
            'Date': result['date'],
            'Category': result['category'],
            'Snippet': result['snippet'],
            'Entities': ", ".join(result['entities'])
        })
    
    df = pd.DataFrame(results_data)
    
    # Create metadata string
    metadata = f"""
    **Query Type:** {search_results['query_type']}
    **Total Results:** {search_results['num_results']}
    **Processing Time:** {search_results['total_time_ms']:.0f}ms
    """
    
    # Create clusters string
    clusters_text = "### Topic Clusters\n\n"
    for cluster_id, cluster_data in search_results['clusters'].items():
        clusters_text += f"**{cluster_data['label']}:** {len(cluster_data['doc_ids'])} articles\n"
    
    return df, metadata, clusters_text

def gradio_search(query: str, num_results: int = 20):
    """Wrapper function for Gradio interface"""
    if not query.strip():
        return None, "Please enter a query", ""
    
    try:
        results = search_articles(query, top_k=num_results)
        return format_results_for_display(results)
    except Exception as e:
        return None, f"Error: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(title="News Article IR System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📰 News Article Information Retrieval System")
    gr.Markdown("### Hybrid BM25 + ColBERT with Neural Reranking")
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query...",
                lines=2
            )
        with gr.Column(scale=1):
            num_results = gr.Slider(
                minimum=5,
                maximum=50,
                value=20,
                step=5,
                label="Number of Results"
            )
    
    search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=2):
            results_table = gr.DataFrame(
                label="Search Results",
                wrap=True
            )
        with gr.Column(scale=1):
            metadata_output = gr.Markdown(label="Query Metadata")
            clusters_output = gr.Markdown(label="Topic Clusters")
    
    # Example queries
    gr.Examples(
        examples=[
            ["latest COVID vaccine updates"],
            ["causes of 2008 financial crisis"],
            ["impact of inflation on technology sector"],
            ["who won the super bowl 2020?"],
            ["climate change economic effects"]
        ],
        inputs=query_input
    )
    
    # Connect button to function
    search_btn.click(
        fn=gradio_search,
        inputs=[query_input, num_results],
        outputs=[results_table, metadata_output, clusters_output]
    )

# Launch
if __name__ == "__main__":
    demo.launch(share=True)
```

---

## Evaluation & Metrics

### 9.1 Create Test Queries with Relevance Judgments

```python
# Create test set (manual relevance judgments)
test_queries = [
    {
        'query': 'COVID vaccine effectiveness',
        'relevant_ids': [12, 45, 67, 89, 123, 234],  # Manually labeled
        'type': 'Breaking'
    },
    {
        'query': '2008 financial crisis causes',
        'relevant_ids': [5, 78, 99, 156, 203],
        'type': 'Historical'
    },
    # Add 20-30 test queries with manual relevance judgments
]

# Save test set
import json
with open('test_queries.json', 'w') as f:
    json.dump(test_queries, f, indent=2)
```

### 9.2 Evaluation Metrics Implementation

```python
import numpy as np
from typing import List, Set

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Precision@k: Proportion of retrieved docs that are relevant"""
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_at_k) & relevant)
    return relevant_retrieved / k if k > 0 else 0.0

def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Recall@k: Proportion of relevant docs that are retrieved"""
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len(set(retrieved_at_k) & relevant)
    return relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0

def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """Average Precision: Average of precision values at positions where relevant docs appear"""
    if not relevant:
        return 0.0
    
    score = 0.0
    num_relevant = 0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            score += precision
    
    return score / len(relevant) if len(relevant) > 0 else 0.0

def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain@k"""
    def dcg(positions):
        return sum([1.0 / np.log2(pos + 2) for pos in positions])
    
    # Actual DCG
    relevant_positions = [i for i, doc_id in enumerate(retrieved[:k]) if doc_id in relevant]
    actual_dcg = dcg(relevant_positions)
    
    # Ideal DCG (all relevant docs at top)
    ideal_positions = list(range(min(len(relevant), k)))
    ideal_dcg = dcg(ideal_positions)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def mrr(retrieved: List[int], relevant: Set[int]) -> float:
    """Mean Reciprocal Rank: Reciprocal of position of first relevant doc"""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_system(test_queries: List[Dict], search_function) -> Dict:
    """
    Evaluate system on test queries.
    
    Returns:
        Dict with evaluation metrics
    """
    metrics = {
        'precision@5': [],
        'precision@10': [],
        'precision@20': [],
        'recall@10': [],
        'recall@20': [],
        'map': [],  # Mean Average Precision
        'ndcg@10': [],
        'ndcg@20': [],
        'mrr': [],
        'latency_ms': []
    }
    
    for test_query in tqdm(test_queries, desc="Evaluating"):
        query = test_query['query']
        relevant = set(test_query['relevant_ids'])
        
        # Run search
        start_time = time.time()
        results = search_function(query, top_k=50)
        latency = (time.time() - start_time) * 1000
        
        # Get retrieved doc IDs
        retrieved = results['doc_ids']
        
        # Compute metrics
        metrics['precision@5'].append(precision_at_k(retrieved, relevant, 5))
        metrics['precision@10'].append(precision_at_k(retrieved, relevant, 10))
        metrics['precision@20'].append(precision_at_k(retrieved, relevant, 20))
        metrics['recall@10'].append(recall_at_k(retrieved, relevant, 10))
        metrics['recall@20'].append(recall_at_k(retrieved, relevant, 20))
        metrics['map'].append(average_precision(retrieved, relevant))
        metrics['ndcg@10'].append(ndcg_at_k(retrieved, relevant, 10))
        metrics['ndcg@20'].append(ndcg_at_k(retrieved, relevant, 20))
        metrics['mrr'].append(mrr(retrieved, relevant))
        metrics['latency_ms'].append(latency)
    
    # Compute averages
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return avg_metrics, metrics

# Run evaluation
with open('test_queries.json', 'r') as f:
    test_queries = json.load(f)

avg_metrics, all_metrics = evaluate_system(test_queries, search_articles)

print("\n=== EVALUATION RESULTS ===")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")
```

### 9.3 Visualization & Plotting

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_evaluation_results(metrics: Dict, save_path: str = "evaluation_plots.png"):
    """
    Create comprehensive evaluation plots.
    
    **IMPORTANT: Generate the following plots:**
    1. Precision@k bar chart (k=5,10,20)
    2. Recall@k bar chart (k=10,20)
    3. NDCG@k bar chart (k=10,20)
    4. MAP and MRR comparison
    5. Latency distribution histogram
    6. Precision-Recall curve
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Information Retrieval System Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Precision@k
    ax1 = axes[0, 0]
    precision_values = [metrics['precision@5'], metrics['precision@10'], metrics['precision@20']]
    ax1.bar(['P@5', 'P@10', 'P@20'], precision_values, color=['#3498db', '#2ecc71', '#f39c12'])
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision at Different k')
    ax1.set_ylim([0, 1])
    for i, v in enumerate(precision_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Recall@k
    ax2 = axes[0, 1]
    recall_values = [metrics['recall@10'], metrics['recall@20']]
    ax2.bar(['R@10', 'R@20'], recall_values, color=['#9b59b6', '#e74c3c'])
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall at Different k')
    ax2.set_ylim([0, 1])
    for i, v in enumerate(recall_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. NDCG@k
    ax3 = axes[0, 2]
    ndcg_values = [metrics['ndcg@10'], metrics['ndcg@20']]
    ax3.bar(['NDCG@10', 'NDCG@20'], ndcg_values, color=['#1abc9c', '#34495e'])
    ax3.set_ylabel('NDCG')
    ax3.set_title('NDCG at Different k')
    ax3.set_ylim([0, 1])
    for i, v in enumerate(ndcg_values):
        ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 4. MAP and MRR
    ax4 = axes[1, 0]
    map_mrr_values = [metrics['map'], metrics['mrr']]
    ax4.bar(['MAP', 'MRR'], map_mrr_values, color=['#e67e22', '#95a5a6'])
    ax4.set_ylabel('Score')
    ax4.set_title('MAP and MRR')
    ax4.set_ylim([0, 1])
    for i, v in enumerate(map_mrr_values):
        ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # 5. Latency Distribution
    ax5 = axes[1, 1]
    latencies = all_metrics['latency_ms']
    ax5.hist(latencies, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.1f}ms')
    ax5.set_xlabel('Latency (ms)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Query Latency Distribution')
    ax5.legend()
    
    # 6. Overall Metrics Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"""
    Overall System Performance
    
    Precision@10: {metrics['precision@10']:.3f}
    Recall@20: {metrics['recall@20']:.3f}
    MAP: {metrics['map']:.3f}
    NDCG@10: {metrics['ndcg@10']:.3f}
    MRR: {metrics['mrr']:.3f}
    
    Avg Latency: {metrics['latency_ms']:.0f}ms
    Max Latency: {max(latencies):.0f}ms
    Min Latency: {min(latencies):.0f}ms
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation plots saved to {save_path}")
    plt.show()

# Generate plots
plot_evaluation_results(avg_metrics, save_path="evaluation_results.png")
```

### 9.4 Ablation Study

```python
def ablation_study():
    """
    Test impact of each component by removing them one at a time.
    
    **Generate comparison plots for:**
    - BM25 only
    - BM25 + ColBERT (hybrid)
    - Hybrid + Reranking
    - Full pipeline (with post-processing)
    """
    configurations = {
        'BM25 Only': lambda q, k: bm25_retrieve(q, k)[:k],
        'Hybrid (BM25 + ColBERT)': lambda q, k: hybrid_retrieve(q, 'Analytical', k),
        'Hybrid + Reranking': lambda q, k: [doc_id for doc_id, _ in rerank_documents(q, hybrid_retrieve(q, 'Analytical', 100), k)],
        'Full Pipeline': lambda q, k: search_articles(q, k)['doc_ids']
    }
    
    results = {}
    
    for config_name, search_func in configurations.items():
        print(f"\nEvaluating: {config_name}")
        
        # Create wrapper function
        def wrapper(query, top_k):
            try:
                doc_ids = search_func(query, top_k)
                return {'doc_ids': doc_ids}
            except:
                return {'doc_ids': []}
        
        avg_metrics, _ = evaluate_system(test_queries, wrapper)
        results[config_name] = avg_metrics
    
    # Plot ablation results
    plot_ablation_results(results)
    
    return results

def plot_ablation_results(results: Dict):
    """Plot ablation study results"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Ablation Study: Component Impact', fontsize=16, fontweight='bold')
    
    configs = list(results.keys())
    
    # NDCG@10
    ax1 = axes[0]
    ndcg_values = [results[c]['ndcg@10'] for c in configs]
    bars1 = ax1.barh(configs, ndcg_values, color='skyblue')
    ax1.set_xlabel('NDCG@10')
    ax1.set_title('Ranking Quality')
    ax1.set_xlim([0, max(ndcg_values) * 1.2])
    for i, v in enumerate(ndcg_values):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # MAP
    ax2 = axes[1]
    map_values = [results[c]['map'] for c in configs]
    bars2 = ax2.barh(configs, map_values, color='lightgreen')
    ax2.set_xlabel('MAP')
    ax2.set_title('Overall Precision')
    ax2.set_xlim([0, max(map_values) * 1.2])
    for i, v in enumerate(map_values):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # Latency
    ax3 = axes[2]
    latency_values = [results[c]['latency_ms'] for c in configs]
    bars3 = ax3.barh(configs, latency_values, color='salmon')
    ax3.set_xlabel('Latency (ms)')
    ax3.set_title('Query Speed')
    ax3.set_xlim([0, max(latency_values) * 1.2])
    for i, v in enumerate(latency_values):
        ax3.text(v + 5, i, f'{v:.0f}ms', va='center')
    
    plt.tight_layout()
    plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
    print("Ablation study plot saved to ablation_study.png")
    plt.show()

# Run ablation study
ablation_results = ablation_study()
```

---

## Implementation Checklist

### Phase 1: Setup (Day 1)
- [ ] Clone/download dataset from Kaggle
- [ ] Set up Python environment and install all libraries
- [ ] Verify spaCy model download
- [ ] Test basic imports and hardware compatibility

### Phase 2: Indexing (Days 2-3)
- [ ] Implement preprocessing pipeline
- [ ] Build BM25 index
- [ ] Build ColBERT index
- [ ] Create SQLite metadata store
- [ ] Train BERTopic model
- [ ] Verify all indices load correctly

### Phase 3: Retrieval Pipeline (Days 4-5)
- [ ] Implement query classification
- [ ] Implement BM25 retrieval
- [ ] Implement ColBERT retrieval
- [ ] Implement parallel execution
- [ ] Implement RRF fusion
- [ ] Test Stage 1 latency

### Phase 4: Reranking (Day 6)
- [ ] Load and test cross-encoder
- [ ] Implement quantization (optional)
- [ ] Implement batch processing
- [ ] Test Stage 2 latency

### Phase 5: Post-Processing (Day 7)
- [ ] Implement MMR diversity
- [ ] Implement temporal intelligence
- [ ] Implement entity deduplication
- [ ] Implement topic clustering
- [ ] Test Stage 3 latency

### Phase 6: UI & Integration (Day 8)
- [ ] Build Gradio interface
- [ ] Connect all pipeline stages
- [ ] Add example queries
- [ ] Test end-to-end system

### Phase 7: Evaluation (Days 9-10)
- [ ] Create test queries with relevance judgments
- [ ] Implement evaluation metrics
- [ ] Run ablation study
- [ ] Generate all plots
- [ ] Document results

### Phase 8: Documentation (Days 11-12)
- [ ] Write technical report
- [ ] Document design decisions
- [ ] Explain evaluation results
- [ ] Discuss limitations
- [ ] Add future work section

---

## Troubleshooting

### Common Issues

**1. Out of Memory during ColBERT indexing**
- Solution: Process in batches of 500 documents at a time
- Reduce `max_document_length` to 256 instead of 512

**2. Slow reranking on CPU**
- Solution: Reduce candidates from 100 to 50
- Implement ONNX quantization
- Use smaller batch size (16 instead of 32)

**3. BERTopic fails to find topics**
- Solution: Reduce `min_topic_size` to 5
- Use `nr_topics=10` for fixed number of topics

**4. Gradio interface crashes**
- Solution: Wrap search function in try-except
- Add error messages for empty queries
- Check all indices are loaded before launching

**5. Evaluation metrics all zero**
- Solution: Verify relevance judgments are correct
- Check doc_ids match between test set and corpus
- Print intermediate results for debugging

---

## Performance Targets (2000 Articles)

| Metric | Target | Actual (Expected) |
|--------|--------|-------------------|
| **Indexing Time** | < 10 min | 4-6 min |
| **Query Latency (CPU)** | < 500ms | 303-403ms |
| **Storage** | < 500MB | 150-250MB |
| **RAM Usage** | < 10GB | 4-5GB peak |
| **NDCG@10** | > 0.45 | 0.48-0.52 |
| **MAP** | > 0.40 | 0.43-0.48 |
| **Precision@10** | > 0.50 | 0.55-0.62 |

---

## Next Steps

1. **Start with simplified version**: BM25 + Reranking only
2. **Add complexity gradually**: Add ColBERT, then post-processing
3. **Optimize bottlenecks**: Profile code to find slow parts
4. **Iterate on evaluation**: Improve based on error analysis
5. **Document thoroughly**: Explain every design decision

---

## Additional Resources

- **BM25 Paper**: Robertson & Zaragoza (2009)
- **ColBERT Paper**: Khattab & Zaharia (2020)
- **Cross-Encoder Guide**: https://www.sbert.net/examples/applications/cross-encoder/README.html
- **BERTopic Tutorial**: https://maartengr.github.io/BERTopic/
- **Gradio Docs**: https://www.gradio.app/docs/

---

**End of Implementation Guide**