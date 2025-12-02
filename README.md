# 🧠 Cortex IR - Advanced News Article Search Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art hybrid information retrieval system for news articles, combining BM25, ColBERT, neural reranking, and intelligent post-processing for superior search performance.

## 🌟 Features

- **🔍 Hybrid Search**: Combines sparse (BM25) and dense (ColBERT) retrieval
- **🧠 Neural Reranking**: Cross-encoder model for improved relevance
- **🎯 Query Intelligence**: Automatic classification, expansion, and spell correction
- **🌈 Diversity Aware**: MMR algorithm for diverse results
- **⏰ Temporal Intelligence**: Context-aware temporal boosting
- **🔗 Entity Deduplication**: Remove duplicates based on named entities
- **📊 Topic Clustering**: Organize results by themes
- **🎨 Modern UI**: Beautiful Gradio interface with animations and visualizations
- **⚡ Fast Performance**: ~10-15ms query latency on CPU
- **🚀 Auto-Setup**: Automatically preprocesses and indexes data on first run

## 📁 Project Structure

```
implementation/
├── config.py                 # Configuration management
├── utils.py                  # Utility functions
├── preprocessing.py          # Stage 0: Article preprocessing
├── indexing.py              # Stage 0: Index construction
├── query_processor.py        # Stage 1: Query processing
├── retrieval.py             # Stage 1: Hybrid retrieval
├── reranker.py              # Stage 2: Neural reranking
├── post_processor.py        # Stage 3: Post-processing
├── main.py                  # Complete pipeline orchestration
├── gradio_app.py            # Gradio web interface
├── evaluation.py            # Evaluation metrics
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  STAGE 0: Preprocessing & Indexing (Offline)                │
├──────────────────────────────────────────────────────────────┤
│  • spaCy NLP pipeline (tokenization, lemmatization, NER)     │
│  • BM25+ sparse index (bm25s library)                        │
│  • ColBERTv2 dense index (RAGatouille)                       │
│  • SQLite metadata store                                      │
│  • BERTopic clustering                                        │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: Hybrid Retrieval (~60-80ms)                       │
├──────────────────────────────────────────────────────────────┤
│  • Query classification (breaking/historical/factual/        │
│    analytical)                                                │
│  • Parallel BM25 + ColBERT retrieval                         │
│  • Reciprocal Rank Fusion (RRF)                              │
│  • Returns top 100 candidates                                │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: Neural Reranking (~200-280ms)                     │
├──────────────────────────────────────────────────────────────┤
│  • Cross-encoder (ms-marco-MiniLM-L-6-v2)                   │
│  • Batch processing for efficiency                           │
│  • Ensemble scoring with multiple signals                    │
│  • Returns top 50 candidates                                 │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: Post-Processing (~40ms)                           │
├──────────────────────────────────────────────────────────────┤
│  • Diversity reranking (MMR)                                 │
│  • Temporal intelligence boosting                            │
│  • Entity-based deduplication                                │
│  • Category balancing                                        │
│  • Topic clustering for presentation                         │
│  • Returns final 20 results                                  │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 16GB RAM recommended
- 5GB disk space for indices

### Installation

1. **Clone the repository** (or navigate to project directory):

```bash
cd implementation
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download spaCy model**:

```bash
python -m spacy download en_core_web_sm
```

5. **Configure environment** (optional):

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## 🚀 Quick Start

**Just run one command:**

```bash
python app.py
```

Or directly:

```bash
python gradio_app.py
```

The system will automatically:
- ✅ Check if data is preprocessed (if not, preprocess Articles.csv)
- ✅ Check if indices are built (if not, build BM25 and metadata indices)
- ✅ Launch the Gradio web interface at `http://localhost:7860`

**First run takes ~3-5 minutes for preprocessing and indexing. Subsequent runs start instantly.**

### Manual Component Usage

If you need to run components separately:

```bash
# Step 1: Preprocess articles
python preprocessing.py

# Step 2: Build indices (BM25, metadata)
python indexing.py

# Step 3: Launch UI
python gradio_app.py
```

### Command-Line Usage

```python
from main import CortexIRPipeline

# Initialize pipeline
pipeline = CortexIRPipeline()

# Execute search
results = pipeline.search(
    query="latest sports news",
    top_k=10,
    enable_reranking=True,
    enable_post_processing=True
)

# Display results
for i, doc in enumerate(results['results'], 1):
    print(f"{i}. {doc['title']}")
    print(f"   Score: {doc.get('ensemble_score', 0):.4f}")
    print(f"   {doc['snippet']}\n")
```

## 📊 Configuration

Edit `config.py` or create a `.env` file to customize:

### Key Parameters

```python
# BM25 Parameters
BM25_K1 = 1.5              # Term frequency saturation
BM25_B = 0.75              # Length normalization
TITLE_BOOST = 3.0          # Title field weight
ENTITY_BOOST = 2.0         # Entity weight

# Retrieval
TOP_K_SPARSE = 100         # BM25 candidates
TOP_K_DENSE = 100          # ColBERT candidates
RRF_K = 60                 # RRF fusion parameter

# Reranking
RERANK_BATCH_SIZE = 32     # Batch size for cross-encoder
RERANK_TOP_K = 50          # Documents to rerank

# Post-processing
FINAL_TOP_K = 20           # Final results
DIVERSITY_LAMBDA_FACTUAL = 0.9      # Relevance weight
DIVERSITY_LAMBDA_EXPLORATORY = 0.6   # Balanced weight
```

## 🧪 Evaluation

Evaluate system performance:

```python
from main import CortexIRPipeline
from evaluation import SystemEvaluator

# Initialize
pipeline = CortexIRPipeline()
evaluator = SystemEvaluator(pipeline)

# Define test queries and relevance judgments
test_queries = ["sports news", "economic trends"]
relevant_docs = [[1, 5, 10], [2, 8, 15]]

# Evaluate
evaluation = evaluator.evaluate_queries(test_queries, relevant_docs)

# Print report
evaluator.print_evaluation_report(evaluation)
```

### Metrics Calculated

- **MAP** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **Precision@K** (K = 5, 10, 20)
- **Recall@K** (K = 5, 10, 20)
- **NDCG@K** (Normalized Discounted Cumulative Gain)
- **Response Time** (Average and standard deviation)

## 🎨 Web Interface Features

### Search Tab
- 🔍 Intelligent search with auto-correction
- 🎛️ Configurable result count
- ⚙️ Toggle reranking and post-processing
- 📊 Real-time performance metrics
- 📈 Interactive visualizations
- 🏷️ Category and timeline distributions
- 💡 Example queries

### Analytics Tab
- 📉 Search history statistics
- ⏱️ Average response times
- 🎯 Query type distribution
- 📊 Usage patterns

### About Tab
- 📖 System documentation
- 🏗️ Architecture overview
- 🛠️ Technology stack
- 📈 Performance specifications

## 🔧 Module Details

### preprocessing.py
- **ArticlePreprocessor**: Tokenization, lemmatization, NER
- **QueryPreprocessor**: Query processing and entity extraction
- Uses spaCy for NLP tasks
- Filters entities by confidence (threshold: 0.85)

### indexing.py
- **BM25Indexer**: Fast sparse index with bm25s
- **ColBERTIndexer**: Dense retrieval with RAGatouille
- **MetadataIndexer**: SQLite database with B-tree indices
- **TopicIndexer**: BERTopic for document clustering

### query_processor.py
- **QueryClassifier**: Classifies queries into 4 types
- **QueryExpander**: PRF-based query expansion
- **SpellCorrector**: Common spelling corrections
- **QueryProcessor**: Complete query pipeline

### retrieval.py
- **BM25Retriever**: Sparse retrieval
- **ColBERTRetriever**: Dense retrieval
- **ReciprocalRankFusion**: Score fusion
- **HybridRetriever**: Parallel hybrid search

### reranker.py
- **NeuralReranker**: Cross-encoder reranking
- **EnsembleReranker**: Multi-signal ensemble
- Batch processing for efficiency
- Early stopping optimization

### post_processor.py
- **DiversityReranker**: MMR for diversity
- **TemporalProcessor**: Time-aware boosting
- **EntityDeduplicator**: Duplicate removal
- **CategoryBalancer**: Cross-category balance
- **ResultClusterer**: Topic-based clustering

### main.py
- **CortexIRPipeline**: Complete pipeline orchestration
- Timing and logging
- Batch search support
- Error handling

### gradio_app.py
- **CortexGradioApp**: Modern web interface
- Animated UI components
- Interactive visualizations
- Search analytics dashboard

## 📈 Performance

Typical performance on Core i5 6th Gen + 16GB RAM:

| Stage | Time (CPU) |
|-------|-----------|
| Query Processing | ~10ms |
| Hybrid Retrieval | ~60-80ms |
| Neural Reranking | ~200-280ms |
| Post-Processing | ~40ms |
| **Total** | **~300-400ms** |

## 🧰 Technology Stack

### Core IR
- **bm25s**: Fast BM25 implementation
- **RAGatouille**: ColBERTv2 wrapper
- **sentence-transformers**: Embeddings & cross-encoder

### NLP & ML
- **spaCy**: NLP pipeline
- **transformers**: Pre-trained models
- **BERTopic**: Topic modeling
- **PyTorch**: Deep learning

### Storage & Data
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **SQLite**: Metadata storage

### UI & Visualization
- **Gradio**: Web interface
- **Plotly**: Interactive charts
- **Matplotlib/Seaborn**: Static plots

## 🐛 Troubleshooting

### Indices Not Found
```bash
# Rebuild indices
python preprocessing.py
python indexing.py
```

### spaCy Model Missing
```bash
python -m spacy download en_core_web_sm
```

### Out of Memory
- Reduce batch size in config: `RERANK_BATCH_SIZE = 16`
- Disable ColBERT: Set `TOP_K_DENSE = 0`
- Reduce candidates: `TOP_K_SPARSE = 50`

### Slow Performance
- Disable reranking for faster results
- Use smaller cross-encoder model
- Reduce number of post-processing steps

## 📝 Example Queries

**Breaking News Queries:**
- "latest COVID updates"
- "recent sports results"
- "breaking business news today"

**Historical Queries:**
- "causes of 2008 financial crisis"
- "history of Olympic games"
- "origin of economic recession"

**Factual Queries:**
- "who won the super bowl?"
- "what is inflation rate?"
- "which team won championship?"

**Analytical Queries:**
- "impact of inflation on economy"
- "trends in sports performance"
- "effects of market volatility"

## 🤝 Contributing

This is an academic project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use for academic purposes.

## 👨‍💻 Author

Built as part of CS516 Information Retrieval Assignment , by Haseeb ur rehman

## 🙏 Acknowledgments

- **spaCy** for NLP capabilities
- **Hugging Face** for transformer models
- **RAGatouille** for simplified ColBERT integration
- **Gradio** for rapid UI development

---

**Built with ❤️ for advanced information retrieval**
