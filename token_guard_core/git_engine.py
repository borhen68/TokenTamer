"""
Git Engine: The Temporal Signal.

Uses git log to calculate file recency and generate decay scores.
"What was edited recently?"
"""

import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


class GitEngine:
    """
    Analyzes git history to generate recency scores for files.
    """

    def __init__(self, half_life_days: float = 30.0):
        """
        Initialize the Git Engine.

        Args:
            half_life_days: The time in days for a file's recency score to halve.
                            Default is 30 days.
        """
        self.half_life_days = half_life_days

    def _get_last_modified_dates(self, repo_path: str) -> Dict[str, datetime]:
        """
        Use subprocess to run git log and extract the last modified date for all files.

        Args:
            repo_path: The root directory of the git repository.

        Returns:
            A dictionary mapping relative file paths to their last modified datetime.
        """
        repo_dir = Path(repo_path).resolve()
        
        # Ensure it's a git repository
        if not (repo_dir / ".git").exists():
            # Fallback for non-git repos: just return current time for all files
            # or raise an exception based on requirements. For now, empty dict.
            return {}

        # Run git log to get the latest commit date for each file
        # Format: %ct is commit timestamp (Unix epoch)
        # We process files one by one to get their specific last commit,
        # but a more efficient approach for large repos is parsing the full log.
        # Let's use `git ls-tree` and `git log` combined for efficiency if possible,
        # or `git log --name-only --format="%ct"` and process the stream.
        
        cmd = [
            "git", "log", 
            "--name-only", 
            "--format=COMMIT|%ct"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=str(repo_dir), 
                capture_output=True, 
                text=True, 
                check=True
            )
        except subprocess.CalledProcessError:
            return {}

        file_dates: Dict[str, datetime] = {}
        current_timestamp = None

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("COMMIT|"):
                try:
                    ts_str = line.split("|")[1]
                    current_timestamp = int(ts_str)
                except (IndexError, ValueError):
                    current_timestamp = None
            elif current_timestamp is not None:
                # It's a filename
                if line not in file_dates:
                    # We only care about the *most recent* commit for each file,
                    # and git log outputs in reverse chronological order.
                    file_dates[line] = datetime.fromtimestamp(current_timestamp, tz=timezone.utc)

        return file_dates

    def get_recency_scores(self, repo_path: str) -> Dict[str, float]:
        """
        Calculate an exponential decay score for all files based on their last edit date.

        Edited today = ~1.0
        Edited `half_life_days` ago = 0.5
        Edited long ago -> approaches 0.0

        Args:
            repo_path: The root directory of the git repository.

        Returns:
            A dictionary mapping absolute file paths to a recency score (0.0 to 1.0).
        """
        repo_dir = Path(repo_path).resolve()
        file_dates = self._get_last_modified_dates(repo_path)
        
        scores: Dict[str, float] = {}
        now = datetime.now(timezone.utc)
        
        # Calculate the decay constant lambda
        # N(t) = N0 * e^(-lambda * t)
        # 0.5 = 1.0 * e^(-lambda * half_life)
        # ln(0.5) = -lambda * half_life
        # lambda = -ln(0.5) / half_life
        decay_constant = -math.log(0.5) / self.half_life_days

        for rel_path, last_mod_date in file_dates.items():
            abs_path = str(repo_dir / rel_path)
            
            # Calculate age in days
            age_td = now - last_mod_date
            age_days = age_td.total_seconds() / (24 * 3600)
            
            # Ensure age is not negative (e.g. due to clock skew)
            age_days = max(0.0, age_days)
            
            # Calculate exponential decay score
            score = math.exp(-decay_constant * age_days)
            
            scores[abs_path] = score

        return scores
