"""
Multi-language code skeletonizer for TokenTamer.

Uses Python's native ast module for Python code and a robust regex/brace-balance
approach for C-style languages (JavaScript, TypeScript, Go, Rust, C, C++, Java, C#).

Strips function/method bodies while preserving structural signatures, imports,
and class definitions. This lets the LLM know *what* exists and *how* to call it,
without burning tokens on implementation details.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass


SKELETON_HEADER = "# [TOKEN-GUARD: Compressed — structural skeleton only]"

# Languages that use C-style brace syntax
C_STYLE_LANGUAGES = {
    "javascript", "js", "typescript", "ts", "tsx", "jsx",
    "go", "rust", "rs", "c", "cpp", "cxx", "h", "hpp",
    "java", "cs", "csharp", "kotlin", "kt", "swift",
    "scala", "php",
}

# Patterns that identify function/method signatures in C-style languages.
# We look for a line that ends with ) (optionally followed by -> type or : type)
# and then an opening brace.
_SIGNATURE_LINE_RE = re.compile(
    r"^(\s*)"                           # indent
    r"((?:export\s+|default\s+|public\s+|private\s+|protected\s+|static\s+|async\s+|func\s+|fn\s+|def\s+|var\s+|const\s+|let\s+)*"
    r"[A-Za-z_0-9\[\]\*\.<>\?\s]+"   # return type / qualifiers
    r"[A-Za-z_0-9]+\s*\([^)]*\)"       # name(args)
    r"(?:\s*:\s*[A-Za-z_0-9\[\]\*\.<>\?\s]+)?"  # optional : Type (TS, Kotlin)
    r"(?:\s*->\s*[A-Za-z_0-9\[\]\*\.<>\?\s]+)?" # optional -> Type (Swift, Rust, C#)
    r")"
    r"(\s*\{)",
    re.MULTILINE,
)

# Rust-specific: functions without braces (use `->` but may not have `{` on same line)
_RUST_FN_RE = re.compile(
    r"^(\s*(?:pub\s+)?fn\s+[A-Za-z_0-9]+\s*\([^)]*\)(?:\s*->\s*[^{]+)?)\s*\{",
    re.MULTILINE,
)

# Go-specific: func declarations
_GO_FUNC_RE = re.compile(
    r"^(\s*(?:func\s+(?:\([^)]*\)\s+)?[A-Za-z_0-9]+\s*\([^)]*\)(?:\s*[A-Za-z_0-9\[\]\*\s]+)?))\s*\{",
    re.MULTILINE,
)


@dataclass
class SkeletonizeResult:
    """Result of skeletonizing a code block."""
    original: str
    skeleton: str
    original_lines: int
    skeleton_lines: int
    was_compressed: bool

    @property
    def lines_saved(self) -> int:
        return self.original_lines - self.skeleton_lines

    @property
    def compression_ratio(self) -> float:
        if self.original_lines == 0:
            return 0.0
        return 1.0 - (self.skeleton_lines / self.original_lines)


class _BodyStripper(ast.NodeTransformer):
    """AST transformer that replaces function/method bodies with ellipsis."""

    def __init__(self, keep_docstrings: bool = False, keep_class_attrs: bool = True):
        super().__init__()
        self.keep_docstrings = keep_docstrings
        self.keep_class_attrs = keep_class_attrs

    def _strip_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        new_body: list[ast.stmt] = []
        if self.keep_docstrings and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                new_body.append(first)
        new_body.append(ast.Expr(value=ast.Constant(value=...)))
        node.body = new_body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        return self._strip_body(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        new_body: list[ast.stmt] = []
        if self.keep_docstrings and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                new_body.append(first)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                new_body.append(self._strip_body(child))
            elif self.keep_class_attrs and isinstance(child, (ast.Assign, ast.AnnAssign)):
                new_body.append(child)
            elif isinstance(child, ast.ClassDef):
                new_body.append(self.visit_ClassDef(child))
            elif isinstance(child, ast.Expr) and isinstance(getattr(child, "value", None), ast.Constant):
                if child not in new_body:
                    continue
            else:
                pass
        if not new_body:
            new_body.append(ast.Expr(value=ast.Constant(value=...)))
        node.body = new_body
        return node


def _find_matching_brace(text: str, open_idx: int) -> int:
    """Find index of matching closing brace starting from open_idx."""
    depth = 0
    in_string: str | None = None
    escape = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if ch == in_string:
                in_string = None
            continue
        if ch in ('"', "'"):
            in_string = ch
            continue
        if ch == "{" and not in_string:
            depth += 1
        elif ch == "}" and not in_string:
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _try_match(text: str) -> re.Match | None:
    """Try multiple patterns to find a function signature."""
    for pattern in (_GO_FUNC_RE, _RUST_FN_RE, _SIGNATURE_LINE_RE):
        match = pattern.search(text)
        if match:
            return match
    return None


def _strip_cstyle_scan(source: str) -> str:
    """Scan-based C-style body stripper that handles nested braces."""
    out_parts: list[str] = []
    i = 0
    n = len(source)

    while i < n:
        # Try to find a function signature ending with { on this "logical line"
        remaining = source[i:]
        match = _try_match(remaining)
        if not match:
            out_parts.append(remaining)
            break

        # Append everything before the match
        out_parts.append(remaining[:match.start()])

        # Append signature up to and including opening brace
        open_brace_idx = i + match.end() - 1
        out_parts.append(remaining[match.start():match.end()])

        # Find matching close brace
        close_brace_idx = _find_matching_brace(source, open_brace_idx)
        out_parts.append(" ... ")

        i = close_brace_idx + 1

    return "".join(out_parts)


class Skeletonizer:
    """
    Converts source code into structural skeletons by stripping function bodies
    while preserving signatures, imports, and class structures.
    Supports Python (AST) and C-style languages (brace-balance heuristic).
    """

    def __init__(self, keep_docstrings: bool = False, keep_class_attrs: bool = True):
        self.keep_docstrings = keep_docstrings
        self.keep_class_attrs = keep_class_attrs

    def skeletonize(self, source: str, language: str = "python") -> SkeletonizeResult:
        """Skeletonize source code, stripping function bodies."""
        original_lines = len(source.splitlines())
        lang = language.lower()

        if lang in ("python", "py", ""):
            return self._skeletonize_python(source, original_lines)

        if lang in C_STYLE_LANGUAGES:
            try:
                skeleton = self._skeletonize_cstyle(source)
                skeleton_lines = len(skeleton.splitlines())
                return SkeletonizeResult(
                    original=source,
                    skeleton=skeleton,
                    original_lines=original_lines,
                    skeleton_lines=skeleton_lines,
                    was_compressed=skeleton_lines < original_lines,
                )
            except Exception:
                return SkeletonizeResult(
                    original=source, skeleton=source,
                    original_lines=original_lines, skeleton_lines=original_lines,
                    was_compressed=False,
                )

        # Unknown language: return unchanged (safe fallback)
        return SkeletonizeResult(
            original=source, skeleton=source,
            original_lines=original_lines, skeleton_lines=original_lines,
            was_compressed=False,
        )

    def _skeletonize_python(self, source: str, original_lines: int) -> SkeletonizeResult:
        try:
            dedented = textwrap.dedent(source)
            tree = ast.parse(dedented)
            stripper = _BodyStripper(
                keep_docstrings=self.keep_docstrings,
                keep_class_attrs=self.keep_class_attrs,
            )
            new_tree = stripper.visit(tree)
            ast.fix_missing_locations(new_tree)
            skeleton = f"{SKELETON_HEADER}\n{ast.unparse(new_tree)}"
            skeleton_lines = len(skeleton.splitlines())
            return SkeletonizeResult(
                original=source, skeleton=skeleton,
                original_lines=original_lines, skeleton_lines=skeleton_lines,
                was_compressed=True,
            )
        except SyntaxError:
            return SkeletonizeResult(
                original=source, skeleton=source,
                original_lines=original_lines, skeleton_lines=original_lines,
                was_compressed=False,
            )

    def _skeletonize_cstyle(self, source: str) -> str:
        """Strip C-style function bodies while keeping imports and top-level declarations."""
        return f"{SKELETON_HEADER}\n{_strip_cstyle_scan(source)}"

    def skeletonize_if_python(self, source: str) -> SkeletonizeResult:
        """Attempt skeletonization; if parsing fails, return unchanged."""
        return self.skeletonize(source, language="python")
