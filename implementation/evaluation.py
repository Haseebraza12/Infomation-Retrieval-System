"""
Evaluation Module for Cortex IR System
Metrics calculation and system evaluation
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

from utils import setup_logging

logger = setup_logging(__name__)


class IRMetrics:
    """
    Calculate IR evaluation metrics
    """
    
    def __init__(self):
        """Initialize metrics calculator"""
        logger.info("IRMetrics initialized")
    
    def precision_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cut-off rank
            
        Returns:
            Precision@K score
        """
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not retrieved_at_k:
            return 0.0
        
        return len(retrieved_at_k & relevant_set) / k
    
    def recall_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cut-off rank
            
        Returns:
            Recall@K score
        """
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        return len(retrieved_at_k & relevant_set) / len(relevant_set)
    
    def average_precision(
        self,
        retrieved: List[int],
        relevant: List[int]
    ) -> float:
        """
        Calculate Average Precision (AP)
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: List of relevant document IDs
            
        Returns:
            Average Precision score
        """
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        precisions = []
        num_relevant = 0
        
        for k, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                num_relevant += 1
                precisions.append(num_relevant / k)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_set)
    
    def mean_average_precision(
        self,
        retrieved_lists: List[List[int]],
        relevant_lists: List[List[int]]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP)
        
        Args:
            retrieved_lists: List of retrieved document lists (one per query)
            relevant_lists: List of relevant document lists (one per query)
            
        Returns:
            MAP score
        """
        aps = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            ap = self.average_precision(retrieved, relevant)
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def ndcg_at_k(
        self,
        retrieved: List[int],
        relevant: List[int],
        relevance_scores: Dict[int, float],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: List of relevant document IDs
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Cut-off rank
            
        Returns:
            NDCG@K score
        """
        # DCG calculation
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += (2**rel - 1) / np.log2(i + 1)
        
        # IDCG calculation (ideal DCG)
        ideal_docs = sorted(
            relevant,
            key=lambda x: relevance_scores.get(x, 0.0),
            reverse=True
        )
        
        idcg = 0.0
        for i, doc_id in enumerate(ideal_docs[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            idcg += (2**rel - 1) / np.log2(i + 1)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def mrr(
        self,
        retrieved_lists: List[List[int]],
        relevant_lists: List[List[int]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            retrieved_lists: List of retrieved document lists
            relevant_lists: List of relevant document lists
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)
            
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


class SystemEvaluator:
    """
    Evaluate complete IR system
    """
    
    def __init__(self, pipeline):
        """
        Initialize evaluator
        
        Args:
            pipeline: CortexIRPipeline instance
        """
        self.pipeline = pipeline
        self.metrics = IRMetrics()
        logger.info("SystemEvaluator initialized")
    
    def evaluate_queries(
        self,
        queries: List[str],
        relevant_docs: List[List[int]],
        k_values: List[int] = None
    ) -> Dict:
        """
        Evaluate system on multiple queries
        
        Args:
            queries: List of test queries
            relevant_docs: List of relevant document IDs for each query
            k_values: List of k values for metrics (default: [5, 10, 20])
            
        Returns:
            Dictionary with evaluation results
        """
        if k_values is None:
            k_values = [5, 10, 20]
        
        logger.info(f"Evaluating {len(queries)} queries")
        
        results = {
            'precision': {k: [] for k in k_values},
            'recall': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'map': [],
            'mrr': [],
            'response_times': []
        }
        
        retrieved_lists = []
        
        for query, relevant in zip(queries, relevant_docs):
            # Execute search
            result = self.pipeline.search(query, top_k=max(k_values))
            
            # Extract retrieved document IDs
            retrieved = [doc['id'] for doc in result['results']]
            retrieved_lists.append(retrieved)
            
            # Record response time
            results['response_times'].append(result['metadata']['total_time_ms'])
            
            # Calculate metrics for different k values
            for k in k_values:
                precision = self.metrics.precision_at_k(retrieved, relevant, k)
                recall = self.metrics.recall_at_k(retrieved, relevant, k)
                
                results['precision'][k].append(precision)
                results['recall'][k].append(recall)
        
        # Calculate aggregate metrics
        results['map'] = self.metrics.mean_average_precision(
            retrieved_lists,
            relevant_docs
        )
        
        results['mrr'] = self.metrics.mrr(retrieved_lists, relevant_docs)
        
        # Calculate averages
        summary = {
            'num_queries': len(queries),
            'avg_response_time_ms': np.mean(results['response_times']),
            'std_response_time_ms': np.std(results['response_times']),
            'map': results['map'],
            'mrr': results['mrr']
        }
        
        for k in k_values:
            summary[f'precision@{k}'] = np.mean(results['precision'][k])
            summary[f'recall@{k}'] = np.mean(results['recall'][k])
        
        logger.info("Evaluation complete")
        logger.info(f"MAP: {summary['map']:.4f}")
        logger.info(f"MRR: {summary['mrr']:.4f}")
        logger.info(f"Avg Response Time: {summary['avg_response_time_ms']:.2f}ms")
        
        return {
            'summary': summary,
            'detailed': results
        }
    
    def print_evaluation_report(self, evaluation: Dict):
        """Print formatted evaluation report"""
        summary = evaluation['summary']
        
        print("\n" + "="*70)
        print(" "*20 + "EVALUATION REPORT")
        print("="*70 + "\n")
        
        print(f"Number of Queries: {summary['num_queries']}")
        print(f"\nPerformance:")
        print(f"  Average Response Time: {summary['avg_response_time_ms']:.2f}ms")
        print(f"  Std Response Time: {summary['std_response_time_ms']:.2f}ms")
        
        print(f"\nEffectiveness Metrics:")
        print(f"  MAP (Mean Average Precision): {summary['map']:.4f}")
        print(f"  MRR (Mean Reciprocal Rank): {summary['mrr']:.4f}")
        
        print(f"\nPrecision@K:")
        for k in [5, 10, 20]:
            if f'precision@{k}' in summary:
                print(f"  P@{k}: {summary[f'precision@{k}']:.4f}")
        
        print(f"\nRecall@K:")
        for k in [5, 10, 20]:
            if f'recall@{k}' in summary:
                print(f"  R@{k}: {summary[f'recall@{k}']:.4f}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Example evaluation"""
    from main import CortexIRPipeline
    
    # Initialize pipeline
    pipeline = CortexIRPipeline()
    evaluator = SystemEvaluator(pipeline)
    
    # Example test queries with mock relevant docs
    test_queries = [
        "latest sports news",
        "business economic trends",
        "championship winners"
    ]
    
    # Mock relevant document IDs (in practice, these would come from human annotations)
    relevant_docs = [
        [10, 15, 23, 45, 67],  # Relevant docs for query 1
        [5, 12, 34, 56, 78],   # Relevant docs for query 2
        [8, 16, 24, 32, 40]    # Relevant docs for query 3
    ]
    
    print("\n=== System Evaluation ===\n")
    print("Note: Using mock relevance judgments for demonstration")
    print("In production, use actual human-annotated relevance judgments\n")
    
    # Evaluate
    evaluation = evaluator.evaluate_queries(test_queries, relevant_docs)
    
    # Print report
    evaluator.print_evaluation_report(evaluation)


if __name__ == "__main__":
    main()
