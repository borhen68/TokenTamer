"""
Dependency Engine: The Structural Signal.

Parses Python files to build a directed graph of imports to understand 
"What imports what?" to map out the codebase dependency structure.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple


class DependencyEngine:
    """
    Scans a directory for Python files and builds a directed graph of
    dependencies based on import statements.
    """

    def __init__(self, root_dir: str):
        """
        Initialize the Dependency Engine.

        Args:
            root_dir: The root directory of the codebase to scan.
        """
        self.root_dir = Path(root_dir).resolve()
        # Mapping from a module's absolute file path to a set of absolute paths it imports.
        self.dependencies: Dict[Path, Set[Path]] = {}
        # Mapping from a module's absolute file path to a set of absolute paths that import it.
        self.dependents: Dict[Path, Set[Path]] = {}
        
        # Build the graph upon initialization
        self._build_graph()

    def _build_graph(self) -> None:
        """Scan the directory and build the dependency and dependent graphs."""
        # Find all python files
        py_files = list(self.root_dir.rglob("*.py"))
        
        # Initialize graphs
        for file in py_files:
            self.dependencies[file] = set()
            self.dependents[file] = set()

        # Map python module names to file paths for resolving imports
        module_to_path: Dict[str, Path] = {}
        for file in py_files:
            # Calculate the module path relative to root
            try:
                rel_path = file.relative_to(self.root_dir)
                # Remove .py and replace / with .
                mod_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
                if mod_name.endswith(".__init__"):
                    mod_name = mod_name[:-9]
                module_to_path[mod_name] = file
                
                # Also map the absolute module name if applicable (simplification)
                # A more robust system would handle sys.path, but this works for simple repos
            except ValueError:
                pass

        # Parse each file and extract imports
        for file in py_files:
            self._parse_file_imports(file, module_to_path)

    def _parse_file_imports(self, filepath: Path, module_to_path: Dict[str, Path]) -> None:
        """
        Parse a single Python file, extract its imports, and resolve them to local files.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError):
            return

        imported_modules: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle relative imports
                    if node.level > 0:
                        # Calculate the base module path based on the current file's location
                        try:
                            rel_dir = filepath.parent.relative_to(self.root_dir)
                            parts = list(rel_dir.parts)
                            # Go up 'level - 1' directories
                            for _ in range(node.level - 1):
                                if parts:
                                    parts.pop()
                            
                            base_mod = ".".join(parts)
                            if base_mod:
                                mod_name = f"{base_mod}.{node.module}"
                            else:
                                mod_name = node.module
                            imported_modules.append(mod_name)
                        except ValueError:
                            pass
                    else:
                        imported_modules.append(node.module)

        # Resolve module names to actual file paths within the repo
        for mod_name in imported_modules:
            # Try exact match
            if mod_name in module_to_path:
                target_file = module_to_path[mod_name]
                self._add_edge(filepath, target_file)
                continue
                
            # Try to match submodules (e.g. 'utils.helpers' might refer to 'utils/helpers.py')
            # Check prefixes
            for known_mod, target_file in module_to_path.items():
                if mod_name.startswith(f"{known_mod}.") or mod_name == known_mod:
                     self._add_edge(filepath, target_file)

    def _add_edge(self, source: Path, target: Path) -> None:
        """Add a directed edge from source to target in the graph."""
        if source == target:
            return
            
        if source not in self.dependencies:
            self.dependencies[source] = set()
        self.dependencies[source].add(target)
        
        if target not in self.dependents:
            self.dependents[target] = set()
        self.dependents[target].add(source)

    def get_dependencies(self, filepath: str, depth: int = 1) -> Tuple[Set[str], Set[str]]:
        """
        Get files the target depends on, and files that depend on the target.

        Args:
            filepath: The absolute or relative path to the target Python file.
            depth: The depth of the dependency traversal (1 = direct imports).

        Returns:
            A tuple of (dependencies, dependents) as sets of string file paths.
        """
        target_path = Path(filepath).resolve()
        
        if target_path not in self.dependencies:
            # File might not exist or wasn't parsed
            return set(), set()

        deps = self._traverse_graph(target_path, self.dependencies, depth)
        deps.discard(target_path)  # Remove self if present
        
        dependents = self._traverse_graph(target_path, self.dependents, depth)
        dependents.discard(target_path)
        
        # Convert Paths back to strings for the return type
        return {str(p) for p in deps}, {str(p) for p in dependents}

    def _traverse_graph(self, start_node: Path, graph: Dict[Path, Set[Path]], max_depth: int) -> Set[Path]:
        """Traverse a graph up to a maximum depth using BFS."""
        visited: Set[Path] = set()
        queue: List[Tuple[Path, int]] = [(start_node, 0)]
        
        while queue:
            current_node, current_depth = queue.pop(0)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            if current_depth < max_depth:
                neighbors = graph.get(current_node, set())
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))
                        
        return visited
