"""
Context analyzer for TokenTamer.

Scans the message payload to determine which code blocks are "active"
(mentioned in the user's prompt and should be left intact) vs "background"
(can be safely skeletonized to save tokens).

Safety-first: if we're uncertain about a code block, we leave it intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .skeletonizer import Skeletonizer, SkeletonizeResult


# Common code file extensions
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb",
    ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".sql", ".yaml", ".yml",
    ".toml", ".json", ".xml", ".html", ".css", ".scss", ".less",
}

# Language detection by extension
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
}

# Regex to find filename references in text
FILE_REFERENCE_PATTERNS = [
    # Explicit file paths: /path/to/file.py, src/utils.py, ./file.py
    re.compile(r'(?:^|[\s`"\'\(])([.\w/\\-]+\.(?:' + "|".join(
        ext.lstrip(".") for ext in CODE_EXTENSIONS
    ) + r'))(?:[\s`"\'\),:;]|$)', re.MULTILINE),
    # Markdown-style file references: ## File: payment.py
    re.compile(r'#+\s*(?:File|Module|Script):\s*(\S+)', re.IGNORECASE),
]

# Regex to find fenced code blocks with optional file annotations
# Supports # File: (Python/Ruby/Shell), // File: (JS/TS/Go/C), /* File: */ (C/JS block)
CODE_BLOCK_PATTERN = re.compile(
    r'(?:(?:#|//)\s*(?:File|file|MODULE|module):\s*(\S+)\n|'
    r'(?:/\*\s*(?:File|file|MODULE|module):\s*(\S+)\s*\*/\n|'
    r'<!--\s*file:\s*(\S+)\s*-->\n))?'                   # HTML annotation above
    r'```(\w*)\n(.*?)```',                                   # The fenced code block
    re.DOTALL,
)

# Pattern for inline file annotations within code blocks
INLINE_FILE_ANNOTATION = re.compile(
    r'^(?:#|//)\s*(?:File|file|MODULE|module):\s*(\S+)',
    re.MULTILINE,
)


@dataclass
class CodeBlock:
    """A code block found in the message payload."""
    content: str
    language: str
    filename: Optional[str]
    start_pos: int
    end_pos: int
    is_active: bool = False
    skeleton_result: Optional[SkeletonizeResult] = None


@dataclass
class AnalysisResult:
    """Result of analyzing a message payload."""
    active_files: Set[str] = field(default_factory=set)
    code_blocks: List[CodeBlock] = field(default_factory=list)
    modified_content: str = ""
    total_blocks: int = 0
    skeletonized_blocks: int = 0


class ContextAnalyzer:
    """
    Analyzes LLM message payloads to identify active vs background code blocks,
    then selectively compresses background blocks using the Skeletonizer.

    Optionally uses a SemanticEngine for intent-based active-file detection
    when a repo_path is provided.
    """

    def __init__(self, skeletonizer: Skeletonizer, repo_path: Optional[str] = None):
        self.skeletonizer = skeletonizer
        self._repo_path = repo_path
        self._semantic_engine: Optional[object] = None
        if repo_path:
            try:
                from token_tamer_core.semantic_engine import SemanticEngine
                self._semantic_engine = SemanticEngine()
            except ImportError:
                pass

    def extract_active_files(self, messages: List[dict]) -> Set[str]:
        """
        Scan messages (especially the last user message) to find file references
        that indicate which files the user is actively working on.

        If a repo_path was provided and sentence-transformers is available,
        also boost files that are semantically similar to the user's query.
        """
        active_files: Set[str] = set()

        # Focus on the last user message — that's where the intent is
        user_messages = [
            m for m in messages
            if m.get("role") == "user"
        ]

        last_user_text = ""
        # Process user and system messages for explicit file references
        for msg in messages:
            content = self._get_text_content(msg)
            if not content:
                continue

            # Strip out code blocks so we only look at natural language text.
            text_only = CODE_BLOCK_PATTERN.sub("", content)

            role = msg.get("role", "")
            is_user = role == "user"
            is_system = role == "system"
            is_last_user = bool(
                user_messages and len(messages) > 0
                and msg == user_messages[-1]
            )

            if is_user or is_system or is_last_user:
                if is_last_user:
                    last_user_text = text_only
                for pattern in FILE_REFERENCE_PATTERNS:
                    for match in pattern.finditer(text_only):
                        filename = match.group(1)
                        if filename:
                            active_files.add(self._normalize_filename(filename))

        # Semantic boost: if we have a repo and a user query, add highly
        # semantically relevant files to the active set.
        if self._semantic_engine and self._repo_path and last_user_text:
            try:
                scores = self._semantic_engine.get_semantic_scores(
                    self._repo_path, last_user_text
                )
                # Boost files with similarity >= 0.5 into active set
                for filepath, score in scores.items():
                    if score >= 0.5:
                        active_files.add(self._normalize_filename(filepath))
            except Exception:
                pass

        return active_files

    def analyze_and_compress(
        self,
        messages: List[dict],
        all_messages: Optional[List[dict]] = None,
    ) -> Tuple[List[dict], AnalysisResult]:
        """
        Analyze the messages array, identify code blocks, and compress
        background files.

        Args:
            messages: The messages array from the API request payload.
            all_messages: Optional extended messages (e.g. including system prompt)
                           for active-file detection only.

        Returns:
            Tuple of (modified messages, analysis result).
        """
        active_files = self.extract_active_files(all_messages or messages)
        result = AnalysisResult(active_files=active_files)
        modified_messages = []

        for msg in messages:
            raw_content = msg.get("content", "")
            if not raw_content:
                modified_messages.append(msg.copy())
                continue

            # Create modified message
            new_msg = msg.copy()
            if isinstance(raw_content, str):
                modified_content = self._process_content(raw_content, active_files, result)
                new_msg["content"] = modified_content
            elif isinstance(raw_content, list):
                # Multi-part content: modify each text part individually
                new_parts = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        new_part = part.copy()
                        new_part["text"] = self._process_content(
                            part["text"], active_files, result
                        )
                        new_parts.append(new_part)
                    else:
                        new_parts.append(part)
                new_msg["content"] = new_parts

            modified_messages.append(new_msg)

        result.modified_content = str(modified_messages)
        return modified_messages, result

    def _process_content(
        self, content: str, active_files: Set[str], result: AnalysisResult
    ) -> str:
        """Process a single content string, compressing background code blocks."""
        # Find all code blocks
        modified = content
        offset = 0  # Track position offset from replacements

        for match in CODE_BLOCK_PATTERN.finditer(content):
            # Extract file annotation (#|//, /* */, or <!-- -->)
            filename = match.group(1) or match.group(2) or match.group(3)
            language = match.group(4) or ""
            code_content = match.group(5)

            # Check for inline file annotation within the code block
            if not filename:
                inline_match = INLINE_FILE_ANNOTATION.search(code_content)
                if inline_match:
                    filename = inline_match.group(1)

            # Determine language from filename if not specified
            if not language and filename:
                ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
                language = EXTENSION_TO_LANGUAGE.get(ext, "")

            # Normalize filename for comparison
            normalized = self._normalize_filename(filename) if filename else None

            # Decide: active or background?
            is_active = False
            if normalized and normalized in active_files:
                is_active = True
            elif not filename:
                # No filename detected — leave intact (safety-first)
                is_active = True

            block = CodeBlock(
                content=code_content,
                language=language,
                filename=filename,
                start_pos=match.start(),
                end_pos=match.end(),
                is_active=is_active,
            )

            result.total_blocks += 1

            if not is_active:
                # Try to skeletonize this background block
                skeleton_result = self.skeletonizer.skeletonize(
                    code_content, language=language
                )

                if skeleton_result.was_compressed:
                    block.skeleton_result = skeleton_result
                    result.skeletonized_blocks += 1

                    # Replace the code block content in the message
                    original_block = match.group(0)
                    compressed_block = original_block.replace(
                        code_content, skeleton_result.skeleton + "\n"
                    )

                    start = match.start() + offset
                    end = match.end() + offset
                    modified = modified[:start] + compressed_block + modified[end:]
                    offset += len(compressed_block) - len(original_block)

            result.code_blocks.append(block)

        return modified

    @staticmethod
    def _get_text_content(message: dict) -> str:
        """Extract text content from a message, handling both string and list formats."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        return ""

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        """
        Normalize a filename for consistent comparison.
        Strips paths down to basename for matching.
        """
        if not filename:
            return ""
        # Get the basename (last component of the path)
        name = filename.replace("\\", "/").rstrip("/")
        if "/" in name:
            name = name.rsplit("/", 1)[-1]
        return name.lower()
