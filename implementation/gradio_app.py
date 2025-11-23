"""
Gradio Web Interface for Cortex IR System
Modern UI with animations, visualizations, and analytics
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

import config
from main import CortexIRPipeline
from utils import setup_logging

logger = setup_logging(__name__)


class CortexGradioApp:
    """
    Gradio web interface for Cortex IR
    """
    
    def __init__(self):
        """Initialize Gradio app"""
        logger.info("Initializing Gradio app...")
        
        # Initialize pipeline
        try:
            self.pipeline = CortexIRPipeline()
            self.pipeline_ready = True
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            self.pipeline_ready = False
            self.pipeline = None
        
        # Search history for analytics
        self.search_history = []
        
        logger.info("Gradio app initialized")
    
    def search(
        self,
        query: str,
        num_results: int,
        enable_reranking: bool,
        enable_post_processing: bool
    ):
        """
        Execute search and format results for UI
        
        Args:
            query: Search query
            num_results: Number of results to return
            enable_reranking: Whether to use reranking
            enable_post_processing: Whether to use post-processing
            
        Returns:
            Tuple of formatted components for UI
        """
        if not self.pipeline_ready:
            error_html = """
            <div style='padding: 20px; background: #fee; border: 2px solid #c33; border-radius: 8px;'>
                <h3>❌ Pipeline Not Ready</h3>
                <p>Please run preprocessing and indexing first:</p>
                <code>python preprocessing.py && python indexing.py</code>
            </div>
            """
            return error_html, "", None, None, None
        
        if not query or not query.strip():
            return "", "Please enter a query", None, None, None
        
        # Execute search
        try:
            result = self.pipeline.search(
                query=query,
                top_k=num_results,
                enable_reranking=enable_reranking,
                enable_post_processing=enable_post_processing
            )
            
            # Save to history
            self.search_history.append({
                'timestamp': datetime.now(),
                'query': query,
                'num_results': len(result['results']),
                'time_ms': result['metadata']['total_time_ms'],
                'query_type': result['query']['type']
            })
            
            # Format results
            results_html = self._format_results(result)
            query_info = self._format_query_info(result['query'])
            performance_chart = self._create_performance_chart(result['metadata'])
            category_chart = self._create_category_chart(result['results'])
            timeline_chart = self._create_timeline_chart(result['results'])
            
            return results_html, query_info, performance_chart, category_chart, timeline_chart
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            error_html = f"""
            <div style='padding: 20px; background: #fee; border: 2px solid #c33; border-radius: 8px;'>
                <h3>❌ Search Error</h3>
                <p>{str(e)}</p>
            </div>
            """
            return error_html, "", None, None, None
    
    def _format_results(self, result: dict) -> str:
        """Format search results as HTML"""
        results = result['results']
        query_type = result['query']['type']
        
        if not results:
            return """
            <div style='padding: 20px; text-align: center; color: #666;'>
                <h3>No results found</h3>
                <p>Try a different query</p>
            </div>
            """
        
        html = f"""
        <style>
            .result-container {{
                animation: fadeIn 0.5s;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .result-card {{
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 16px;
                transition: all 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            .result-card:hover {{
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                transform: translateY(-2px);
                border-color: #4285f4;
            }}
            .result-rank {{
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                width: 32px;
                height: 32px;
                border-radius: 50%;
                text-align: center;
                line-height: 32px;
                font-weight: bold;
                margin-right: 12px;
            }}
            .result-title {{
                font-size: 18px;
                font-weight: 600;
                color: #1a73e8;
                margin: 8px 0;
                cursor: pointer;
            }}
            .result-title:hover {{
                text-decoration: underline;
            }}
            .result-snippet {{
                color: #5f6368;
                line-height: 1.6;
                margin: 12px 0;
            }}
            .result-meta {{
                display: flex;
                gap: 16px;
                flex-wrap: wrap;
                margin-top: 12px;
                font-size: 13px;
            }}
            .meta-item {{
                display: flex;
                align-items: center;
                gap: 6px;
                color: #5f6368;
            }}
            .category-badge {{
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 500;
            }}
            .category-business {{
                background: #e8f5e9;
                color: #2e7d32;
            }}
            .category-sports {{
                background: #e3f2fd;
                color: #1565c0;
            }}
            .score-badge {{
                padding: 4px 10px;
                background: #fff3e0;
                color: #e65100;
                border-radius: 8px;
                font-weight: 600;
                font-size: 12px;
            }}
            .query-type-badge {{
                display: inline-block;
                padding: 6px 14px;
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 20px;
            }}
        </style>
        
        <div class="result-container">
            <div style="margin-bottom: 20px;">
                <span class="query-type-badge">🎯 {query_type.title()} Query</span>
                <span style="color: #666; margin-left: 12px;">
                    Found {len(results)} results in {result['metadata']['total_time_ms']:.0f}ms
                </span>
            </div>
        """
        
        for i, doc in enumerate(results, 1):
            category = doc.get('category', 'Unknown')
            category_class = f"category-{category.lower()}"
            
            # Determine score to display
            score = doc.get('ensemble_score',
                           doc.get('rerank_score',
                                  doc.get('retrieval_score', 0.0)))
            
            # Format date
            date_str = doc.get('date', 'N/A')
            
            # Get entities
            entities = doc.get('entities', [])[:3]
            entity_str = ', '.join([e['text'] for e in entities]) if entities else 'None'
            
            html += f"""
            <div class="result-card">
                <div style="display: flex; align-items: start;">
                    <span class="result-rank">{i}</span>
                    <div style="flex: 1;">
                        <div class="result-title">
                            {doc.get('title', 'Untitled')}
                        </div>
                        <div class="result-snippet">
                            {doc.get('snippet', doc.get('content', '')[:200])}...
                        </div>
                        <div class="result-meta">
                            <span class="category-badge {category_class}">
                                📁 {category}
                            </span>
                            <span class="meta-item">
                                📅 {date_str}
                            </span>
                            <span class="score-badge">
                                ⭐ {score:.3f}
                            </span>
                            <span class="meta-item" title="Named Entities">
                                🏷️ {entity_str}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        
        return html
    
    def _format_query_info(self, query_info: dict) -> str:
        """Format query information"""
        info = f"""
        **Original Query:** {query_info['original']}
        
        **Corrected Query:** {query_info['corrected']}
        
        **Query Type:** {query_info['type'].title()}
        
        **Tokens:** {', '.join(query_info['tokens'])}
        
        **Entities Detected:** {len(query_info['entities'])}
        """
        
        if query_info['entities']:
            entities_str = ', '.join([f"{e['text']} ({e['label']})" for e in query_info['entities']])
            info += f"\n\n**Entity Details:** {entities_str}"
        
        return info
    
    def _create_performance_chart(self, metadata: dict) -> go.Figure:
        """Create performance breakdown chart"""
        stages = ['Query\nProcessing', 'Retrieval', 'Reranking', 'Post\nProcessing']
        times = [
            metadata['stage_times']['query_processing'],
            metadata['stage_times']['retrieval'],
            metadata['stage_times']['reranking'],
            metadata['stage_times']['post_processing']
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=stages,
                y=times,
                text=[f"{t:.1f}ms" for t in times],
                textposition='auto',
                marker=dict(
                    color=times,
                    colorscale='Viridis',
                    showscale=False
                )
            )
        ])
        
        fig.update_layout(
            title={
                'text': f"Performance Breakdown (Total: {metadata['total_time_ms']:.1f}ms)",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Pipeline Stage",
            yaxis_title="Time (ms)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _create_category_chart(self, results: list) -> go.Figure:
        """Create category distribution chart"""
        categories = {}
        for doc in results:
            cat = doc.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(categories.keys()),
                values=list(categories.values()),
                hole=0.4,
                marker=dict(
                    colors=['#4CAF50', '#2196F3', '#FF9800'],
                    line=dict(color='white', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title={
                'text': "Results by Category",
                'x': 0.5,
                'xanchor': 'center'
            },
            template="plotly_white",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def _create_timeline_chart(self, results: list) -> go.Figure:
        """Create timeline distribution of results"""
        dates = []
        for doc in results:
            parsed_date = doc.get('parsed_date')
            if parsed_date:
                try:
                    if isinstance(parsed_date, str):
                        dates.append(pd.to_datetime(parsed_date))
                    else:
                        dates.append(parsed_date)
                except:
                    pass
        
        if not dates:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Timeline Distribution (No date data)",
                template="plotly_white",
                height=400
            )
            return fig
        
        # Create histogram
        fig = go.Figure(data=[
            go.Histogram(
                x=dates,
                marker=dict(
                    color='#1976D2',
                    line=dict(color='white', width=1)
                ),
                nbinsx=20
            )
        ])
        
        fig.update_layout(
            title={
                'text': "Results Timeline Distribution",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Date",
            yaxis_title="Number of Articles",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def get_analytics(self):
        """Get analytics from search history"""
        if not self.search_history:
            return "No search history yet. Try some queries first!"
        
        df = pd.DataFrame(self.search_history)
        
        analytics = f"""
        ## 📊 Search Analytics
        
        **Total Searches:** {len(self.search_history)}
        
        **Average Response Time:** {df['time_ms'].mean():.1f}ms
        
        **Average Results per Query:** {df['num_results'].mean():.1f}
        
        **Query Type Distribution:**
        """
        
        for qtype, count in df['query_type'].value_counts().items():
            analytics += f"\n- {qtype.title()}: {count} queries"
        
        return analytics
    
    def build_interface(self):
        """Build Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
        }
        .gr-button-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        }
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Cortex IR System", theme=gr.themes.Soft()) as app:
            
            # Header
            gr.Markdown("""
            # 🧠 Cortex IR - News Article Search Engine
            
            **Advanced Hybrid Information Retrieval System** with BM25, ColBERT, Neural Reranking, and Intelligent Post-Processing
            """)
            
            # Main search interface
            with gr.Tab("🔍 Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter your search query here... (e.g., 'latest sports news', 'economic impact of inflation')",
                            lines=2
                        )
                    
                    with gr.Column(scale=1):
                        num_results = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5,
                            label="Number of Results"
                        )
                
                with gr.Row():
                    reranking_check = gr.Checkbox(
                        label="Enable Neural Reranking",
                        value=True,
                        info="Use cross-encoder for better ranking"
                    )
                    post_processing_check = gr.Checkbox(
                        label="Enable Post-Processing",
                        value=True,
                        info="Apply diversity, temporal boost, and deduplication"
                    )
                
                search_btn = gr.Button("🚀 Search", variant="primary", size="lg")
                
                # Results
                gr.Markdown("## Results")
                
                results_output = gr.HTML(label="Search Results")
                
                # Analytics row
                with gr.Row():
                    with gr.Column():
                        query_info_output = gr.Markdown(label="Query Analysis")
                    
                    with gr.Column():
                        performance_chart = gr.Plot(label="Performance")
                
                with gr.Row():
                    category_chart = gr.Plot(label="Category Distribution")
                    timeline_chart = gr.Plot(label="Timeline")
                
                # Example queries
                gr.Markdown("### 💡 Example Queries")
                with gr.Row():
                    gr.Examples(
                        examples=[
                            ["latest business news"],
                            ["who won the championship?"],
                            ["impact of inflation on economy"],
                            ["sports team performance 2023"],
                            ["company merger announcements"]
                        ],
                        inputs=query_input
                    )
                
                # Connect search button
                search_btn.click(
                    fn=self.search,
                    inputs=[query_input, num_results, reranking_check, post_processing_check],
                    outputs=[results_output, query_info_output, performance_chart, 
                            category_chart, timeline_chart]
                )
            
            # Analytics tab
            with gr.Tab("📈 Analytics"):
                gr.Markdown("## System Analytics and Search History")
                
                analytics_output = gr.Markdown()
                refresh_btn = gr.Button("🔄 Refresh Analytics", variant="secondary")
                
                refresh_btn.click(
                    fn=self.get_analytics,
                    outputs=analytics_output
                )
            
            # About tab
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About Cortex IR System
                
                Cortex IR is a state-of-the-art hybrid information retrieval system designed for news article search.
                
                ### 🏗️ System Architecture
                
                **4-Stage Pipeline:**
                
                1. **Preprocessing & Indexing**
                   - spaCy-based NLP pipeline
                   - Named Entity Recognition (NER)
                   - BM25+ sparse index
                   - ColBERTv2 dense index
                   - SQLite metadata store
                
                2. **Hybrid Retrieval** (~60-80ms)
                   - Parallel BM25 and ColBERT retrieval
                   - Reciprocal Rank Fusion (RRF)
                   - Query classification and expansion
                
                3. **Neural Reranking** (~200-280ms)
                   - Cross-encoder model (MiniLM-L-6)
                   - Batch processing optimization
                   - Early stopping for efficiency
                
                4. **Post-Processing** (~40ms)
                   - Diversity-aware reranking (MMR)
                   - Temporal intelligence
                   - Entity-based deduplication
                   - Topic clustering
                
                ### 🎯 Features
                
                - **Hybrid Search**: Combines keyword (BM25) and semantic (ColBERT) search
                - **Neural Reranking**: Deep learning for improved relevance
                - **Query Intelligence**: Automatic query classification and expansion
                - **Diversity**: MMR algorithm for diverse results
                - **Temporal Awareness**: Boost recent articles for breaking news queries
                - **Deduplication**: Remove duplicate articles based on entity similarity
                - **Topic Clustering**: Organize results by themes
                
                ### 📊 Performance
                
                - **Query Latency**: ~300-400ms on CPU
                - **Indexed Articles**: 2000 news articles
                - **Categories**: Business, Sports
                - **Accuracy**: High precision and recall with neural reranking
                
                ### 🛠️ Technology Stack
                
                - **IR Libraries**: bm25s, ragatouille (ColBERT)
                - **NLP**: spaCy, transformers, sentence-transformers
                - **ML**: PyTorch, scikit-learn
                - **UI**: Gradio, Plotly
                - **Storage**: SQLite, pickle
                
                ---
                
                **Built with ❤️ for advanced information retrieval**
                """)
        
        return app
    
    def launch(
        self,
        share: bool = None,
        server_name: str = None,
        server_port: int = None
    ):
        """Launch Gradio app"""
        if share is None:
            share = config.GRADIO_SHARE
        if server_name is None:
            server_name = config.GRADIO_SERVER_NAME
        if server_port is None:
            server_port = config.GRADIO_SERVER_PORT
        
        app = self.build_interface()
        
        logger.info(f"Launching Gradio app on {server_name}:{server_port}")
        
        app.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )


def main():
    """Main entry point"""
    app = CortexGradioApp()
    app.launch()


if __name__ == "__main__":
    main()
