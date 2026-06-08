"""
Assembler Engine: The Master Weighting Algorithm.

Combines the Structural (Dependency), Temporal (Git), and Intent (Semantic) signals
to produce a mathematically ranked list of the exact files needed for the LLM prompt.
"""

from typing import Dict, List, Set, Tuple
from pathlib import Path

from .dependency_engine import DependencyEngine
from .git_engine import GitEngine
from .semantic_engine import SemanticEngine


class ContextAssembler:
    """
    The master orchestrator that combines signals from the three core engines
    to determine the optimal context payload for an LLM.
    """

    def __init__(self, repo_path: str):
        """
        Initialize the Assembler with all three engines.

        Args:
            repo_path: The root directory of the codebase.
        """
        self.repo_path = str(Path(repo_path).resolve())
        self.dependency_engine = DependencyEngine(self.repo_path)
        self.git_engine = GitEngine(half_life_days=30.0)
        self.semantic_engine = SemanticEngine()
        
        # Cache the git scores since they don't change per query
        self._git_scores = self.git_engine.get_recency_scores(self.repo_path)

    def assemble_context(
        self, 
        query: str, 
        target_file: str = None, 
        top_k: int = 5,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ) -> List[Tuple[str, float]]:
        """
        Rank all files in the repository based on the combined weighted signals.

        The final score is a weighted sum:
        Score = (W_semantic * Semantic_Score) + 
                (W_structural * Structural_Score) + 
                (W_temporal * Temporal_Score)

        Args:
            query: The user's natural language instruction.
            target_file: Optional absolute path to a specific file the user is focused on.
            top_k: Number of top-ranked files to return.
            weights: Tuple of (Semantic Weight, Structural Weight, Temporal Weight).
                     Must sum to 1.0.

        Returns:
            A sorted list of tuples (file_path, total_score), highest score first.
        """
        w_semantic, w_structural, w_temporal = weights
        
        # 1. Get Intent Signal (Semantic Score)
        semantic_scores = self.semantic_engine.get_semantic_scores(self.repo_path, query)
        
        # 2. Get Structural Signal (Dependency Score)
        structural_scores: Dict[str, float] = {}
        
        # If the user specified a target file, heavily weight its dependencies
        if target_file:
            deps, dependents = self.dependency_engine.get_dependencies(target_file, depth=1)
            # The target file itself is structurally crucial
            structural_scores[target_file] = 1.0
            for d in deps:
                structural_scores[d] = 0.8  # Direct dependencies are very important
            for d in dependents:
                structural_scores[d] = 0.5  # Files that use the target might be impacted
        else:
            # If no target file, we might infer it from the query/semantic scores,
            # or we assign 0 structural score initially.
            # Let's boost structural scores for files that have high semantic scores.
            top_semantic = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
            if top_semantic:
                best_guess_target = top_semantic[0][0]
                deps, dependents = self.dependency_engine.get_dependencies(best_guess_target, depth=1)
                structural_scores[best_guess_target] = 1.0
                for d in deps:
                    structural_scores[d] = 0.8
                for d in dependents:
                    structural_scores[d] = 0.5

        # 3. Combine Signals
        final_scores: Dict[str, float] = {}
        
        # Get all unique files seen across any engine
        all_files: Set[str] = set()
        all_files.update(semantic_scores.keys())
        all_files.update(structural_scores.keys())
        all_files.update(self._git_scores.keys())
        
        for file in all_files:
            s_sem = semantic_scores.get(file, 0.0)
            s_str = structural_scores.get(file, 0.0)
            s_tmp = self._git_scores.get(file, 0.0)
            
            # Master equation
            total_score = (w_semantic * s_sem) + (w_structural * s_str) + (w_temporal * s_tmp)
            final_scores[file] = total_score

        # 4. Sort and return top K
        ranked_files = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_files[:top_k]
