"""
AST-based code skeletonizer for Token-Guard.

Parses Python source code into an Abstract Syntax Tree and strips function/method
bodies while preserving structural signatures, imports, and class definitions.
This lets the LLM know *what* exists and *how* to call it, without burning tokens
on the internal implementation details.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass


SKELETON_HEADER = "# [TOKEN-GUARD: Compressed — structural skeleton only]"


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
    """
    AST transformer that replaces function/method bodies with ellipsis,
    optionally preserving docstrings.
    """

    def __init__(self, keep_docstrings: bool = False, keep_class_attrs: bool = True):
        super().__init__()
        self.keep_docstrings = keep_docstrings
        self.keep_class_attrs = keep_class_attrs

    def _strip_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        """Replace a function body with an ellipsis, optionally keeping the docstring."""
        new_body: list[ast.stmt] = []

        if self.keep_docstrings and node.body:
            first = node.body[0]
            # Check if the first statement is a docstring
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, (ast.Constant,))
                and isinstance(first.value.value, str)
            ):
                new_body.append(first)

        # Add ellipsis as the body placeholder
        ellipsis_node = ast.Expr(value=ast.Constant(value=...))
        new_body.append(ellipsis_node)

        node.body = new_body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Strip regular function bodies."""
        # First, recurse into nested functions/classes
        self.generic_visit(node)
        return self._strip_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Strip async function bodies."""
        self.generic_visit(node)
        return self._strip_body(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """
        Process class definitions: strip method bodies but optionally
        keep class-level attributes and type annotations.
        """
        new_body: list[ast.stmt] = []

        # Optionally keep the class docstring
        if self.keep_docstrings and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, (ast.Constant,))
                and isinstance(first.value.value, str)
            ):
                new_body.append(first)

        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Strip the method body
                stripped = self._strip_body(child)
                new_body.append(stripped)
            elif self.keep_class_attrs and isinstance(child, (ast.Assign, ast.AnnAssign)):
                # Keep class-level attributes and type annotations
                new_body.append(child)
            elif isinstance(child, ast.ClassDef):
                # Recurse into nested classes
                visited = self.visit_ClassDef(child)
                new_body.append(visited)
            elif isinstance(child, ast.Expr) and isinstance(
                getattr(child, "value", None), ast.Constant
            ):
                # Skip standalone string expressions (non-first docstrings)
                if child not in new_body:
                    continue
            else:
                # Skip other statements in class body
                pass

        if not new_body:
            new_body.append(ast.Expr(value=ast.Constant(value=...)))

        node.body = new_body
        return node


class Skeletonizer:
    """
    Converts source code into structural skeletons by stripping function bodies
    while preserving signatures, imports, and class structures.
    """

    def __init__(self, keep_docstrings: bool = False, keep_class_attrs: bool = True):
        self.keep_docstrings = keep_docstrings
        self.keep_class_attrs = keep_class_attrs

    def skeletonize(self, source: str, language: str = "python") -> SkeletonizeResult:
        """
        Skeletonize source code, stripping function bodies.

        Args:
            source: The source code string to compress.
            language: Programming language (currently only 'python' is supported).

        Returns:
            SkeletonizeResult with original and compressed code.
        """
        original_lines = len(source.splitlines())

        if language.lower() != "python":
            # Non-Python: return unchanged (future: tree-sitter support)
            return SkeletonizeResult(
                original=source,
                skeleton=source,
                original_lines=original_lines,
                skeleton_lines=original_lines,
                was_compressed=False,
            )

        try:
            skeleton = self._skeletonize_python(source)
            skeleton_lines = len(skeleton.splitlines())
            return SkeletonizeResult(
                original=source,
                skeleton=skeleton,
                original_lines=original_lines,
                skeleton_lines=skeleton_lines,
                was_compressed=True,
            )
        except SyntaxError:
            # If we can't parse it, return unchanged
            return SkeletonizeResult(
                original=source,
                skeleton=source,
                original_lines=original_lines,
                skeleton_lines=original_lines,
                was_compressed=False,
            )

    def _skeletonize_python(self, source: str) -> str:
        """
        Parse Python source with the ast module, strip function bodies,
        and regenerate clean code.
        """
        # Dedent in case the source is indented (e.g., inside a code block)
        dedented = textwrap.dedent(source)
        tree = ast.parse(dedented)

        # Apply the body stripper transformation
        stripper = _BodyStripper(
            keep_docstrings=self.keep_docstrings,
            keep_class_attrs=self.keep_class_attrs,
        )
        new_tree = stripper.visit(tree)

        # Fix missing line numbers for new nodes
        ast.fix_missing_locations(new_tree)

        # Regenerate source code
        skeleton = ast.unparse(new_tree)

        # Add the header comment
        return f"{SKELETON_HEADER}\n{skeleton}"

    def skeletonize_if_python(self, source: str) -> SkeletonizeResult:
        """
        Attempt to skeletonize — if parsing fails, assume it's not Python
        and return unchanged.
        """
        return self.skeletonize(source, language="python")
