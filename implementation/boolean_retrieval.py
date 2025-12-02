"""
Boolean Retrieval Module
Handles boolean queries (AND, OR, NOT) using an inverted index
Supports Wildcard (*), Phrasal ("..."), and standard boolean operators
"""

import re
import fnmatch
from typing import List, Set, Dict, Union, Optional, Tuple
from collections import defaultdict
import pickle
from pathlib import Path
import spacy

from config import Config
from utils import logger, timer, load_pickle

class BooleanRetriever:
    """
    Boolean retrieval system using inverted index
    Supports AND, OR, NOT operators, parentheses, wildcards, and phrase queries
    """
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        # Inverted index: term -> doc_id -> list of positions
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.doc_ids = set()
        self.is_built = False
        self.all_terms = [] # Sorted list of all terms for wildcard search
        
        # Load spaCy for lemmatization
        try:
            self.nlp = spacy.load(self.config.SPACY_MODEL)
        except OSError:
            logger.warning(f"Model {self.config.SPACY_MODEL} not found. Downloading...")
            import os
            os.system(f"python -m spacy download {self.config.SPACY_MODEL}")
            self.nlp = spacy.load(self.config.SPACY_MODEL)
            
        # Load preprocessed data and build index
        self._load_and_build_index()
        
    def _load_and_build_index(self):
        """Load processed articles and build inverted index"""
        try:
            processed_path = self.config.PROCESSED_DATA_PATH
            if not processed_path.exists():
                logger.warning(f"Processed data not found at {processed_path}. Boolean retrieval will be empty.")
                return
                
            logger.info("Loading processed data for Boolean Retrieval...")
            articles = load_pickle(processed_path)
            
            self.build_index(articles)
            
        except Exception as e:
            logger.error(f"Error initializing BooleanRetriever: {e}")
    
    @timer
    def build_index(self, articles: List[Dict]):
        """
        Build positional inverted index from articles
        
        Args:
            articles: List of processed articles
        """
        logger.info(f"Building positional inverted index for {len(articles)} articles...")
        
        self.inverted_index.clear()
        self.doc_ids.clear()
        
        for article in articles:
            doc_id = article['id']
            self.doc_ids.add(doc_id)
            
            # Get all tokens (preserving order for positional indexing)
            # Assuming 'all_tokens' is a list of tokens in order
            tokens = article.get('all_tokens', [])
            
            for pos, token in enumerate(tokens):
                self.inverted_index[token][doc_id].append(pos)
                
        self.all_terms = sorted(list(self.inverted_index.keys()))
        self.is_built = True
        logger.info(f"Inverted index built with {len(self.inverted_index)} terms")
        
    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize query string into terms, phrases, and operators
        Preserves operators: AND, OR, NOT, (, )
        Handles phrases in quotes: "machine learning"
        """
        # Normalize query
        # Add spaces around parentheses
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        
        # Regex to capture phrases in quotes or standard tokens
        pattern = r'"([^"]+)"|(\S+)'
        
        tokens = []
        for match in re.finditer(pattern, query):
            phrase, word = match.groups()
            
            if phrase:
                # Lemmatize phrase content
                # We need to process the phrase to match index tokens
                # Index tokens are lemmatized, lowercased, no stopwords (mostly)
                # But here we should probably just lemmatize and lowercase
                # We should NOT remove stopwords inside a phrase if the index has them?
                # The index REMOVES stopwords. So "machine learning" -> "machine", "learn".
                # "The machine" -> "machine".
                # So we should also remove stopwords from the phrase query to match the index sequence.
                
                phrase_doc = self.nlp(phrase.lower())
                phrase_tokens = [
                    token.lemma_ 
                    for token in phrase_doc 
                    if not token.is_stop and not token.is_punct and token.is_alpha
                ]
                
                if phrase_tokens:
                    # Reconstruct phrase with lemmatized tokens
                    clean_phrase = " ".join(phrase_tokens)
                    tokens.append(f"_PHRASE_{clean_phrase}") 
                    
            elif word:
                upper_word = word.upper()
                if upper_word in ('AND', 'OR', 'NOT', '(', ')'):
                    tokens.append(upper_word)
                else:
                    # Clean and lemmatize term
                    # Handle wildcards: don't lemmatize if it has *
                    if '*' in word:
                        clean_term = word.lower() # Just lowercase wildcard
                        tokens.append(clean_term)
                    else:
                        # Lemmatize single term
                        # Pass original word to preserve case for 'US' detection
                        term_doc = self.nlp(word)
                        # Take the first token's lemma (should be only one)
                        if term_doc:
                             # Filter stopwords? 
                             # Whitelist 'US'
                             valid_tokens = [
                                 t.lemma_.lower() 
                                 for t in term_doc 
                                 if (t.text == 'US' or (not t.is_stop and not t.is_punct and t.is_alpha))
                             ]
                             if valid_tokens:
                                 tokens.append(valid_tokens[0])
                             else:
                                 # If it was a stopword, maybe we ignore it?
                                 pass
                    
        return tokens

    def _clean_text(self, text: str) -> str:
        """Clean text to match index terms (lowercase, remove special chars except *)"""
        # Allow * for wildcards
        text = text.lower()
        # Remove characters that are not alphanumeric or * or - or space
        text = re.sub(r'[^\w\s\-\*]', '', text)
        return text.strip()
    
    def retrieve(self, query: str) -> Set[int]:
        """
        Execute boolean query
        
        Args:
            query: Boolean query string (e.g. "apple AND banana", "machine learning", "comp*")
            
        Returns:
            Set of document IDs
        """
        if not self.is_built:
            logger.warning("Inverted index not built. Returning empty results.")
            return set()
            
        if not query or not query.strip():
            return set()
            
        try:
            tokens = self.tokenize_query(query)
            if not tokens:
                return set()
                
            # Parse and evaluate
            result_ids = self._evaluate_expression(tokens)
            return result_ids
            
        except Exception as e:
            logger.error(f"Error evaluating boolean query '{query}': {e}")
            return set()
            
    def _evaluate_expression(self, tokens: List[str]) -> Set[int]:
        """
        Evaluate boolean expression
        """
        # If single term
        if len(tokens) == 1:
            return self._get_postings(tokens[0])
            
        # Use Shunting-yard algorithm to convert to RPN, then evaluate
        rpn = self._to_rpn(tokens)
        return self._evaluate_rpn(rpn)

    def _to_rpn(self, tokens: List[str]) -> List[Union[str, Set[int]]]:
        """Convert infix tokens to Reverse Polish Notation"""
        output_queue = []
        operator_stack = []
        
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        
        for token in tokens:
            if token in ('AND', 'OR', 'NOT'):
                while (operator_stack and operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0)):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()
            else:
                # Operand (term or phrase)
                output_queue.append(token)
                
        while operator_stack:
            output_queue.append(operator_stack.pop())
            
        return output_queue

    def _evaluate_rpn(self, rpn: List[Union[str, Set[int]]]) -> Set[int]:
        """Evaluate RPN expression"""
        stack = []
        
        for token in rpn:
            if token == 'AND':
                if len(stack) < 2: continue
                right = stack.pop()
                left = stack.pop()
                stack.append(left & right)
            elif token == 'OR':
                if len(stack) < 2: continue
                right = stack.pop()
                left = stack.pop()
                stack.append(left | right)
            elif token == 'NOT':
                if len(stack) < 1: continue
                operand = stack.pop()
                stack.append(self.doc_ids - operand)
            else:
                # Term or Phrase
                stack.append(self._get_postings(token))
                
        return stack[0] if stack else set()

    def _get_postings(self, term: str) -> Set[int]:
        """Get postings list for a term, phrase, or wildcard"""
        
        # 1. Handle Phrase
        if term.startswith("_PHRASE_"):
            phrase_content = term[8:] # Remove prefix
            return self._get_phrase_postings(phrase_content)
            
        # 2. Handle Wildcard
        if '*' in term:
            return self._get_wildcard_postings(term)
            
        # 3. Exact Match
        if term in self.inverted_index:
            return set(self.inverted_index[term].keys())
            
        return set()

    def _get_wildcard_postings(self, wildcard_term: str) -> Set[int]:
        """Get union of postings for all terms matching wildcard"""
        matching_terms = fnmatch.filter(self.all_terms, wildcard_term)
        
        result_docs = set()
        for term in matching_terms:
            result_docs.update(self.inverted_index[term].keys())
            
        return result_docs

    def _get_phrase_postings(self, phrase: str) -> Set[int]:
        """
        Get documents containing the exact phrase
        Uses positional index to check adjacency
        """
        words = phrase.split()
        if not words:
            return set()
            
        # Get docs containing all words in the phrase
        # Start with docs containing the first word
        first_word = words[0]
        if first_word not in self.inverted_index:
            return set()
            
        candidate_docs = set(self.inverted_index[first_word].keys())
        
        # Intersect with docs containing other words
        for word in words[1:]:
            if word not in self.inverted_index:
                return set()
            candidate_docs &= set(self.inverted_index[word].keys())
            
        if not candidate_docs:
            return set()
            
        # Verify positions for candidate docs
        final_docs = set()
        
        for doc_id in candidate_docs:
            # Check if words appear in sequence in this doc
            # We need to find if there exists a sequence of positions p, p+1, p+2...
            # corresponding to words[0], words[1], words[2]...
            
            # Get positions for all words in this doc
            # positions_list = [ [pos1, pos2...], [pos1, pos2...] ... ]
            positions_list = [self.inverted_index[word][doc_id] for word in words]
            
            if self._has_phrase_match(positions_list):
                final_docs.add(doc_id)
                
        return final_docs
        
    def _has_phrase_match(self, positions_list: List[List[int]]) -> bool:
        """
        Check if a sequence of words exists based on their position lists
        positions_list[i] contains positions of i-th word in the phrase
        """
        if not positions_list:
            return False
            
        # We can use a recursive approach or iterative
        # Iterative:
        # Current valid positions for the phrase prefix
        # Start with positions of the first word
        current_positions = positions_list[0]
        
        for i in range(1, len(positions_list)):
            next_positions = positions_list[i]
            valid_next_positions = []
            
            # For each position 'p' in current_positions, check if 'p+1' exists in next_positions
            # Optimization: could use sets or sorted lists for faster lookup
            next_positions_set = set(next_positions)
            
            for pos in current_positions:
                if (pos + 1) in next_positions_set:
                    valid_next_positions.append(pos + 1)
            
            if not valid_next_positions:
                return False
                
            current_positions = valid_next_positions
            
        return True

if __name__ == "__main__":
    # Test
    retriever = BooleanRetriever()
    print("Index size:", len(retriever.inverted_index))
    
    # Simple test
    if len(retriever.inverted_index) > 0:
        term = list(retriever.inverted_index.keys())[0]
        print(f"Testing term: {term}")
        results = retriever.retrieve(term)
        print(f"Results for '{term}': {len(results)}")
        
        # Wildcard test
        wildcard = term[:2] + "*"
        print(f"Testing wildcard: {wildcard}")
        results = retriever.retrieve(wildcard)
        print(f"Results for '{wildcard}': {len(results)}")
