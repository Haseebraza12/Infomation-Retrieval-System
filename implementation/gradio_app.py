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
import sys
from pathlib import Path

from config import Config
from utils import logger, check_indices_exist


class CortexGradioApp:
    """
    Gradio web interface for Cortex IR
    """
    
    def __init__(self):
        """Initialize Gradio app"""
        logger.info("Initializing Cortex IR System...")
        
        # Initialize config
        self.config = Config()
        
        # Ensure system is ready
        self._ensure_system_ready()
        
        # Initialize pipeline (now guaranteed to have indices)
        from main import CortexIRPipeline
        try:
            logger.info("Loading IR pipeline...")
            self.pipeline = CortexIRPipeline()
            self.pipeline_ready = True
            logger.info("✅ Pipeline loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error initializing pipeline: {e}", exc_info=True)
            self.pipeline_ready = False
            self.pipeline = None
        
        # Search history for analytics
        self.search_history = []
        
        logger.info("🎉 Cortex IR System initialized and ready!")
    
    def _ensure_system_ready(self):
        """Ensure preprocessing and indexing are done - runs automatically if needed"""
        logger.info("Checking system readiness...")
        
        # Check if indices exist
        indices = check_indices_exist(self.config)
        
        needs_preprocessing = not indices['processed_data']
        needs_indexing = not indices['bm25'] or not indices['metadata']
        
        if needs_preprocessing or needs_indexing:
            logger.info("=" * 70)
            logger.info("🔧 FIRST TIME SETUP - Building search indices...")
            logger.info("=" * 70)
            logger.info("This will take 3-5 minutes on first run.")
            logger.info("Subsequent runs will be instant!")
            logger.info("")
        
        # Run preprocessing if needed
        if needs_preprocessing:
            logger.info("📚 Step 1/2: Preprocessing articles...")
            logger.info("-" * 70)
            try:
                self._run_preprocessing()
                logger.info("✅ Preprocessing completed!")
            except Exception as e:
                logger.error(f"❌ Preprocessing failed: {e}", exc_info=True)
                raise RuntimeError("Failed to preprocess data. Please check the logs.")
        else:
            logger.info("✅ Preprocessed data found")
        
        # Run indexing if needed
        if needs_indexing:
            logger.info("")
            logger.info("🔨 Step 2/2: Building search indices...")
            logger.info("-" * 70)
            try:
                self._run_indexing()
                logger.info("✅ Indexing completed!")
            except Exception as e:
                logger.error(f"❌ Indexing failed: {e}", exc_info=True)
                raise RuntimeError("Failed to build indices. Please check the logs.")
        else:
            logger.info("✅ Search indices found")
        
        if needs_preprocessing or needs_indexing:
            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ SETUP COMPLETE - System is ready!")
            logger.info("=" * 70)
            logger.info("")
        else:
            logger.info("✅ All components ready - launching UI...")
    
    def _run_preprocessing(self):
        """Run preprocessing pipeline"""
        from preprocessing import main as preprocess_main
        
        logger.info("Starting preprocessing pipeline...")
        preprocess_main()
        logger.info("Preprocessing completed successfully")
    
    def _run_indexing(self):
        """Run indexing pipeline"""
        from indexing import main as indexing_main
        
        logger.info("Starting indexing pipeline...")
        indexing_main()
        logger.info("Indexing completed successfully")
    
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
                <p>The system failed to initialize. Please check the logs.</p>
            </div>
            """
            return error_html, "Pipeline not ready", None, None, None
        
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
            
            logger.info(f"Search Query: '{query}'")
            logger.info(f"Corrected Query: '{result['query']['corrected']}'")
            logger.info(f"Correction diff: {query.strip() != result['query']['corrected'].strip()}")
            
            # Format results
            results_html = self._format_results(result)
            query_info = self._format_query_info(result['query'])
            performance_chart = self._create_performance_chart(result['metadata'])
            category_chart = self._create_category_chart(result['results'])
            timeline_chart = self._create_timeline_chart(result['results'])
            
            # Check for spelling correction
            original_query = result['query']['original']
            corrected_query = result['query']['corrected']
            
            did_you_mean_visible = False
            did_you_mean_val = ""
            did_you_mean_state_val = ""
            
            # Simple check: if they differ (ignoring case/space if needed, but strict is fine)
            if original_query.strip() != corrected_query.strip():
                did_you_mean_visible = True
                did_you_mean_val = corrected_query
                did_you_mean_state_val = corrected_query
            
            # Generate highlighted query
            highlighted_html = self._generate_highlighted_query(original_query, corrected_query)
            
            return (results_html, query_info, performance_chart, category_chart, 
                    timeline_chart, 
                    gr.Group(visible=did_you_mean_visible), 
                    gr.Button(value=did_you_mean_val),
                    highlighted_html)
            
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            error_html = f"""
            <div style='padding: 20px; background: #fee; border: 2px solid #c33; border-radius: 8px;'>
                <h3>❌ Search Error</h3>
                <p>{str(e)}</p>
            </div>
            """
            return error_html, "", None, None, None, gr.Group(visible=False), gr.Button(), ""
    
    def advanced_search(
        self,
        query: str,
        category: str,
        start_date: str,
        end_date: str,
        entity: str,
        num_results: int,
        enable_reranking: bool,
        enable_post_processing: bool
    ):
        """
        Execute advanced hybrid search
        """
        if not self.pipeline_ready:
            return "Pipeline not ready", "", None
            
        if not query or not query.strip():
            return "Please enter a query", "", None
            
        try:
            # Execute hybrid search
            result = self.pipeline.hybrid_search(
                query=query,
                category=category if category != "All" else None,
                start_date=start_date,
                end_date=end_date,
                entity=entity,
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
            
            return results_html, query_info, performance_chart
            
        except Exception as e:
            logger.error(f"Advanced search error: {e}", exc_info=True)
            error_html = f"""
            <div style='padding: 20px; background: #fee; border: 2px solid #c33; border-radius: 8px;'>
                <h3>❌ Search Error</h3>
                <p>{str(e)}</p>
            </div>
            """
            return error_html, "", None
    
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
            if entities and isinstance(entities[0], dict):
                entity_str = ', '.join([e['text'] for e in entities])
            else:
                entity_str = ', '.join(entities) if entities else 'None'
            
            # Get snippet
            snippet = doc.get('snippet', doc.get('content', '')[:200])
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            
            html += f"""
            <div class="result-card">
                <div style="display: flex; align-items: start;">
                    <span class="result-rank">{i}</span>
                    <div style="flex: 1;">
                        <div class="result-title">
                            {doc.get('title', 'Untitled')}
                        </div>
                        <div class="result-snippet">
                            {snippet}
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

**Tokens:** {', '.join(query_info['tokens'][:10])}

**Entities Detected:** {len(query_info['entities'])}
        """
        
        if query_info['entities']:
            entities_str = ', '.join([f"{e['text']} ({e['label']})" for e in query_info['entities'][:5]])
            info += f"\n\n**Entity Details:** {entities_str}"
        
        return info
    
    def _generate_highlighted_query(self, original: str, corrected: str) -> str:
        """Generate HTML with highlighted errors"""
        if original.strip() == corrected.strip():
            return ""
            
        orig_words = original.split()
        corr_words = corrected.split()
        
        if len(orig_words) != len(corr_words):
            # Fallback if token counts differ
            return f"<div style='color: #d32f2f; margin-top: 4px;'>Spelling errors detected</div>"
            
        html_parts = []
        for ow, cw in zip(orig_words, corr_words):
            if ow != cw:
                html_parts.append(
                    f"<span style='color: #d32f2f; text-decoration: underline; font-weight: bold; cursor: help;' "
                    f"title='Did you mean: {cw}'>{ow}</span>"
                )
            else:
                html_parts.append(ow)
                
        return f"<div style='margin-top: 4px; color: #666;'>Detected errors: {' '.join(html_parts)}</div>"
    
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
        
        if not categories:
            fig = go.Figure()
            fig.update_layout(
                title="Category Distribution (No data)",
                template="plotly_white",
                height=400
            )
            return fig
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(categories.keys()),
                values=list(categories.values()),
                hole=0.4,
                marker=dict(
                    colors=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'],
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
        
        with gr.Blocks(title="Cortex IR System") as app:
            
            # Custom CSS
            gr.HTML("""
            <style>
                .gradio-container {
                    font-family: 'Inter', 'Segoe UI', sans-serif;
                }
                h1 {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-weight: 800;
                }
            </style>
            """)
            
            # Header
            gr.Markdown("""
            # 🧠 Cortex IR - News Article Search Engine
            
            **Advanced Hybrid Information Retrieval System** with BM25, Dense Retrieval, Neural Reranking, and Intelligent Post-Processing
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
                        highlighted_query_output = gr.HTML()
                    
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
                        value=True
                    )
                    post_processing_check = gr.Checkbox(
                        label="Enable Post-Processing",
                        value=True
                    )
                
                search_btn = gr.Button("🚀 Search", variant="primary")
                
                # Did you mean component
                with gr.Group(visible=False) as did_you_mean_group:
                    with gr.Row():
                        gr.Markdown("### Did you mean:")
                        did_you_mean_btn = gr.Button("", variant="secondary", size="sm")
                
                did_you_mean_state = gr.State("")
                
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
                gr.Examples(
                    examples=[
                        ["latest business news"],
                        ["who won the championship?"],
                        ["impact of inflation on economy"],
                        ["sports team performance 2023"],
                        ["company merger announcements"],
                        ["explin machine lerning"],
                        ["sindh transport fares"],
                        ["asian markets 2015"],
                        ["china manufacturing growth"],
                        ["dollar exchange rate"],
                        ["gold price analysis"]
                    ],
                    inputs=query_input,
                    label="Example Queries"
                )
                
                # Connect search button
                search_btn.click(
                    fn=self.search,
                    inputs=[query_input, num_results, reranking_check, post_processing_check],
                    outputs=[results_output, query_info_output, performance_chart, 
                            category_chart, timeline_chart, did_you_mean_group, did_you_mean_btn,
                            highlighted_query_output]
                )
                
                # Did you mean handler
                def apply_correction(corrected_query):
                    return corrected_query
                
                did_you_mean_btn.click(
                    fn=apply_correction,
                    inputs=[did_you_mean_state],
                    outputs=[query_input]
                ).then(
                    fn=self.search,
                    inputs=[query_input, num_results, reranking_check, post_processing_check],
                    outputs=[results_output, query_info_output, performance_chart, 
                            category_chart, timeline_chart, did_you_mean_group, did_you_mean_btn,
                            highlighted_query_output]
                )

            # Advanced Search Tab
            with gr.Tab("🔬 Advanced Search"):
                gr.Markdown("### Boolean Search & Filters")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        adv_query_input = gr.Textbox(
                            label="Boolean Query",
                            placeholder="e.g. 'economy AND (growth OR inflation) NOT recession'",
                            lines=2
                        )
                    
                    with gr.Column(scale=1):
                        adv_num_results = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5,
                            label="Number of Results"
                        )
                
                with gr.Row():
                    category_filter = gr.Dropdown(
                        choices=["All", "Business", "Sports", "Technology"],
                        label="Category",
                        value="All"
                    )
                    entity_filter = gr.Textbox(
                        label="Entity Filter",
                        placeholder="e.g. Tesla, Apple"
                    )
                
                with gr.Row():
                    start_date = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        placeholder="2023-01-01"
                    )
                    end_date = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        placeholder="2023-12-31"
                    )
                
                with gr.Row():
                    adv_reranking_check = gr.Checkbox(
                        label="Enable Neural Reranking",
                        value=True
                    )
                    adv_post_processing_check = gr.Checkbox(
                        label="Enable Post-Processing",
                        value=True
                    )
                
                adv_search_btn = gr.Button("🚀 Advanced Search", variant="primary")
                
                # Results
                gr.Markdown("## Results")
                adv_results_output = gr.HTML(label="Search Results")
                
                # Analytics
                with gr.Row():
                    with gr.Column():
                        adv_query_info_output = gr.Markdown(label="Query Analysis")
                    with gr.Column():
                        adv_performance_chart = gr.Plot(label="Performance")
                
                # Example Boolean Queries
                gr.Markdown("### 💡 Example Boolean Queries")
                gr.Examples(
                    examples=[
                        ["technology AND innovation"],
                        ["business AND (merger OR acquisition)"],
                        ["sports NOT football"],
                        ["economy AND inflation NOT recession"]
                    ],
                    inputs=adv_query_input
                )
                
                # Connect advanced search button
                adv_search_btn.click(
                    fn=self.advanced_search,
                    inputs=[adv_query_input, category_filter, start_date, end_date, 
                           entity_filter, adv_num_results, adv_reranking_check, adv_post_processing_check],
                    outputs=[adv_results_output, adv_query_info_output, adv_performance_chart]
                )
            
            # Analytics tab
            with gr.Tab("📈 Analytics"):
                gr.Markdown("## System Analytics and Search History")
                
                analytics_output = gr.Markdown()
                refresh_btn = gr.Button("🔄 Refresh Analytics")
                
                refresh_btn.click(
                    fn=self.get_analytics,
                    outputs=analytics_output
                )
            
            # System Metrics Tab
            with gr.Tab("System Metrics"):
                gr.Markdown("## System Evaluation Metrics")
                
                try:
                    import json
                    report_path = self.config.INDEX_DIR / "evaluation_report.json"
                    if report_path.exists():
                        with open(report_path, 'r') as f:
                            report = json.load(f)
                        
                        summary = report['summary']
                        
                        with gr.Row():
                            gr.Number(value=summary.get('map', 0), label="MAP (Mean Average Precision)")
                            gr.Number(value=summary.get('mrr', 0), label="MRR (Mean Reciprocal Rank)")
                            gr.Number(value=summary.get('avg_response_time_ms', 0), label="Avg Response Time (ms)")
                        
                        gr.Markdown("### Precision, Recall & F1-Score @ K")
                        
                        # Create data for dataframe
                        metrics_data = []
                        for k in [5, 10, 20]:
                            metrics_data.append([
                                f"@{k}",
                                f"{summary.get(f'precision@{k}', 0):.4f}",
                                f"{summary.get(f'recall@{k}', 0):.4f}",
                                f"{summary.get(f'f1@{k}', 0):.4f}"
                            ])
                        
                        gr.Dataframe(
                            headers=["K", "Precision", "Recall", "F1-Score"],
                            value=metrics_data,
                            interactive=False
                        )
                    else:
                        gr.Markdown("*Evaluation report not found. Run evaluation to generate metrics.*")
                except Exception as e:
                    gr.Markdown(f"Error loading metrics: {e}")

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
                   - Dense embeddings index
                   - SQLite metadata store
                
                2. **Hybrid Retrieval** (~60-80ms)
                   - Parallel BM25 and dense retrieval
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
                
                - **Hybrid Search**: Combines keyword (BM25) and semantic search
                - **Neural Reranking**: Deep learning for improved relevance
                - **Query Intelligence**: Automatic query classification and expansion
                - **Diversity**: MMR algorithm for diverse results
                - **Temporal Awareness**: Boost recent articles for breaking news queries
                - **Deduplication**: Remove duplicate articles based on entity similarity
                - **Topic Clustering**: Organize results by themes
                
                ### 📊 Performance
                
                - **Query Latency**: ~300-400ms on CPU
                - **Indexed Articles**: 2692 news articles
                - **Categories**: Business, Sports
                - **Accuracy**: High precision and recall with neural reranking
                
                ### 🛠️ Technology Stack
                
                - **IR Libraries**: bm25s, sentence-transformers
                - **NLP**: spaCy, transformers
                - **ML**: PyTorch, scikit-learn
                - **UI**: Gradio, Plotly
                - **Storage**: SQLite, pickle
                
                ---
                
                **Built with ❤️ for advanced information retrieval**
                """)
        
        return app
    
    def launch(
        self,
        share: bool = True,
    ):
        """Launch Gradio app"""
        app = self.build_interface()
        
        
        app.launch(
            share=share,
            show_error=True
        )


def main():
    """Main entry point"""
    try:
        app = CortexGradioApp()
        app.launch()
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        print("\n" + "="*70)
        print("❌ APPLICATION STARTUP FAILED")
        print("="*70)
        print(f"Error: {e}")
        print("\nPlease check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
