"""
Semantic Engine: The Intent Signal.

Uses sentence-transformers to calculate cosine similarity between a user's query
and the files in the codebase to answer: "What files actually match the user's intent?"
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SemanticEngine:
    """
    Analyzes codebase files against a user query using text embeddings
    to determine semantic relevance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Semantic Engine.

        Args:
            model_name: The sentence-transformers model to use.
                        Default 'all-MiniLM-L6-v2' is small and fast.
        """
        self.model_name = model_name
        self.model = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            # We initialize lazily to save memory if not immediately needed
            self._initialized = False
        else:
            print("Warning: sentence-transformers or scikit-learn not installed.")
            print("Run: pip install sentence-transformers scikit-learn")
            self._initialized = False

    def _initialize_model(self) -> None:
        """Load the embedding model lazily."""
        if not self._initialized and HAS_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True

    def _get_file_summaries(self, repo_path: str) -> Dict[str, str]:
        """
        Extract a brief text summary from each Python file for embedding.
        Currently extracts the first docstring or the first few lines.
        
        Args:
            repo_path: The root directory to scan.
            
        Returns:
            A dictionary mapping absolute file paths to text summaries.
        """
        repo_dir = Path(repo_path).resolve()
        py_files = list(repo_dir.rglob("*.py"))
        
        summaries: Dict[str, str] = {}
        
        for file in py_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read(2048) # Read just the top portion
                    
                # A simple heuristic: grab the first block of text/comments
                lines = content.splitlines()
                summary_lines = []
                for line in lines[:20]: # Look at first 20 lines max
                    if line.strip():
                        summary_lines.append(line.strip())
                
                # Combine into a single text block
                summary_text = " ".join(summary_lines)
                if summary_text:
                    summaries[str(file)] = summary_text
            except (UnicodeDecodeError, IOError):
                continue
                
        return summaries

    def get_semantic_scores(self, repo_path: str, query: str) -> Dict[str, float]:
        """
        Calculate semantic relevance scores for files against a user query.

        Args:
            repo_path: The root directory of the codebase.
            query: The user's instruction or bug description.

        Returns:
            A dictionary mapping absolute file paths to semantic scores (0.0 to 1.0).
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            # Fallback: simple text matching
            return self._fallback_keyword_scoring(repo_path, query)

        self._initialize_model()
        
        file_summaries = self._get_file_summaries(repo_path)
        if not file_summaries:
            return {}

        paths = list(file_summaries.keys())
        texts = [f"Filename: {Path(p).name}. Content: {file_summaries[p]}" for p in paths]
        
        # Calculate embeddings
        query_embedding = self.model.encode([query])
        document_embeddings = self.model.encode(texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, document_embeddings)[0]
        
        # Normalize scores to 0.0 - 1.0 range if needed, though cosine is usually -1 to 1
        # ReLU-like approach: discard negative correlations
        scores: Dict[str, float] = {}
        for path, score in zip(paths, similarities):
            scores[path] = max(0.0, float(score))
            
        return scores

    def _fallback_keyword_scoring(self, repo_path: str, query: str) -> Dict[str, float]:
        """Simple keyword overlap scoring if embeddings are unavailable."""
        query_words = set(query.lower().replace(".", " ").split())
        if not query_words:
            return {}
            
        file_summaries = self._get_file_summaries(repo_path)
        scores: Dict[str, float] = {}
        
        for path, content in file_summaries.items():
            content_lower = f"{Path(path).name} {content}".lower()
            matches = sum(1 for word in query_words if word in content_lower)
            # Simple score: matches / num_query_words
            score = min(1.0, matches / len(query_words))
            scores[path] = float(score)
            
        return scores
