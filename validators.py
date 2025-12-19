"""
Validators for generated Python code.

This module provides validation functions to ensure generated code:
1) Is syntactically valid Python (AST check)
2) Contains required imports and patterns for Delphes analysis
3) Does not contain anti-patterns like explicit event loops
4) Avoids template-echo placeholders (e.g., {collection}, {plotting_code})
"""

from __future__ import annotations

import ast
import re
from typing import List, Tuple, Set


# -----------------------------
# Regex-based checks (lightweight)
# -----------------------------

REQUIRED_IMPORTS = [
    ("uproot", r"(^|\n)\s*(import\s+uproot\b|from\s+uproot\b\s+import\b)"),
    ("awkward as ak", r"(^|\n)\s*import\s+awkward\s+as\s+ak\b"),
    ("vector", r"(^|\n)\s*(import\s+vector\b|from\s+vector\b\s+import\b)"),
]

# Anti-patterns that indicate incorrect code (explicit event loops)
FORBIDDEN_PATTERNS = [
    ("explicit event loop 'for event in'", r"\bfor\s+event\s+in\b"),
    ("explicit index loop 'for i in range(len('", r"\bfor\s+\w+\s+in\s+range\s*\(\s*len\s*\("),
]

# Common template placeholders that should NEVER appear in final code
# These are typical LLM "template echo" mistakes
# NOTE: We only flag these if they appear OUTSIDE of f-strings
KNOWN_BAD_PLACEHOLDERS = {
    # Template-style placeholders (these are almost never valid variable names)
    "{branch}", "{branches}", "{collection}", "{collections}",
    "{stage}", "{stages}", "{particle}", "{particles}", 
    "{object}", "{objects}", "{plot}", "{plots}", 
    "{plot_name}", "{plot_names}", "{cut}", "{cuts}", 
    "{mask}", "{masks}", "{output}", "{output_structure}",
    "{object_cut_code}", "{plotting_code}", "{selection_code}",
    # These COULD be variables but are suspicious in isolation
    "{n_total}", "{n_events}", "{n_after}",
}

# These are ONLY flagged if they appear outside f-strings AND outside function parameters
# "candidates" is a common legitimate variable name, so we need to be careful
CONTEXT_SENSITIVE_NAMES = {'candidate', 'candidates', 'count', 'counts'}


# -----------------------------
# AST helpers
# -----------------------------

class _CodeAnalyzer(ast.NodeVisitor):
    """Analyzes code to find f-string variables and function parameters."""
    
    def __init__(self):
        self.found_register_awkward = False
        self.uses_vector_like_ops = False
        self.uses_delphes_subscript = False
        self.fstring_variables: Set[str] = set()  # Variables used in f-strings
        self.all_variable_names: Set[str] = set()  # All defined variable names
        self.function_parameters: Set[str] = set()  # Function parameter names

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function parameter names."""
        for arg in node.args.args:
            self.function_parameters.add(arg.arg)
            self.all_variable_names.add(arg.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        """Track all variable names used."""
        self.all_variable_names.add(node.id)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        """Track variable names used in f-strings."""
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                self._extract_names_from_node(value.value)
        self.generic_visit(node)
    
    def _extract_names_from_node(self, node):
        """Recursively extract all Name nodes from an expression."""
        if isinstance(node, ast.Name):
            self.fstring_variables.add(node.id)
        elif isinstance(node, ast.Attribute):
            self._extract_names_from_node(node.value)
        elif isinstance(node, ast.Subscript):
            self._extract_names_from_node(node.value)
            self._extract_names_from_node(node.slice)
        elif isinstance(node, ast.BinOp):
            self._extract_names_from_node(node.left)
            self._extract_names_from_node(node.right)
        elif isinstance(node, ast.Call):
            self._extract_names_from_node(node.func)
            for arg in node.args:
                self._extract_names_from_node(arg)

    def visit_Call(self, node: ast.Call):
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "register_awkward":
            self.found_register_awkward = True
        elif isinstance(fn, ast.Name) and fn.id == "register_awkward":
            self.found_register_awkward = True

        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            if fn.value.id == "vector" and fn.attr in {"zip", "array"}:
                self.uses_vector_like_ops = True

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "vector":
            self.uses_vector_like_ops = True
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and slice_node.value == "Delphes":
            self.uses_delphes_subscript = True
        elif hasattr(ast, 'Index') and isinstance(slice_node, ast.Index):
            if isinstance(slice_node.value, ast.Constant) and slice_node.value.value == "Delphes":
                self.uses_delphes_subscript = True
        self.generic_visit(node)


def ast_validate(code: str) -> Tuple[List[str], ast.AST | None]:
    """Validate that code is syntactically valid Python and return AST."""
    errors: List[str] = []
    try:
        tree = ast.parse(code)
        return errors, tree
    except SyntaxError as e:
        errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        return errors, None


# -----------------------------
# Validation checks
# -----------------------------

def check_required_imports(code: str) -> List[str]:
    missing: List[str] = []
    for name, pattern in REQUIRED_IMPORTS:
        if not re.search(pattern, code, flags=re.MULTILINE):
            missing.append(f"Missing required import: {name}")
    return missing


def check_forbidden_patterns(code: str) -> List[str]:
    found: List[str] = []
    for name, pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            found.append(f"Forbidden pattern detected: {name}")
    return found


def check_placeholders(code: str, ast_tree: ast.AST | None) -> List[str]:
    """
    Check for unresolved template placeholders while respecting legitimate uses.
    
    This uses AST analysis to:
    1. Find all variables used in f-strings (these are legitimate)
    2. Find all function parameters (these are legitimate)
    3. Only flag {name} patterns that are NOT legitimate variable references
    """
    errors: List[str] = []
    
    # Get legitimate variable names from AST
    legitimate_names: Set[str] = set()
    if ast_tree is not None:
        analyzer = _CodeAnalyzer()
        analyzer.visit(ast_tree)
        # Any variable used in an f-string is legitimate
        legitimate_names = analyzer.fstring_variables
        # Function parameters are legitimate
        legitimate_names.update(analyzer.function_parameters)
        # All defined/used variables are legitimate
        legitimate_names.update(analyzer.all_variable_names)
    
    # Find all {identifier} patterns in the code
    placeholder_pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
    
    found_bad = []
    
    for match in placeholder_pattern.finditer(code):
        name = match.group(1)
        placeholder = "{" + name + "}"
        
        # Skip if it's a known legitimate variable
        if name in legitimate_names:
            continue
        
        # Check if this is a known bad placeholder
        if placeholder in KNOWN_BAD_PLACEHOLDERS:
            found_bad.append(placeholder)
        # Check for suspicious single-word template patterns
        # But only if they're NOT in legitimate_names
        elif name.lower() in {'branch', 'stage', 'particle', 'plot', 'cut', 
                               'mask', 'collection', 'object', 'output'}:
            found_bad.append(placeholder)
    
    if found_bad:
        unique_bad = sorted(set(found_bad))
        errors.append(f"Unresolved template placeholders found: {', '.join(unique_bad)}")
    
    return errors


def check_savefig(code: str) -> List[str]:
    """Require saving plots."""
    if not re.search(r"\.savefig\s*\(", code):
        return ["Missing required plot save call: *.savefig(...)"]
    return []


def check_tree_arrays_ak(code: str) -> List[str]:
    """Require using awkward as library='ak' in tree.arrays call."""
    pattern = r"\.arrays\s*\([^)]*library\s*=\s*['\"]ak['\"]"
    if not re.search(pattern, code):
        return ["Missing required pattern: tree.arrays(..., library='ak')"]
    return []


def check_delphes_access(ast_tree: ast.AST | None) -> List[str]:
    """Require that the code indexes the Delphes tree via ["Delphes"]."""
    if ast_tree is None:
        return []
    analyzer = _CodeAnalyzer()
    analyzer.visit(ast_tree)
    if not analyzer.uses_delphes_subscript:
        return ['Missing required ROOT access: file["Delphes"]']
    return []


def check_vector_registration(ast_tree: ast.AST | None, code: str) -> List[str]:
    """Require register_awkward() if vector operations are used."""
    if ast_tree is None:
        return []

    analyzer = _CodeAnalyzer()
    analyzer.visit(ast_tree)

    uses_vector_text = bool(re.search(r"\bvector\.", code))
    uses_vector = analyzer.uses_vector_like_ops or uses_vector_text

    if uses_vector and not analyzer.found_register_awkward:
        return ["Missing required call: vector.register_awkward()"]

    return []


def check_main_guard(code: str) -> List[str]:
    """Check for proper if __name__ == "__main__" guard."""
    if not re.search(r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', code):
        return ['Missing entry point: if __name__ == "__main__": block']
    return []


def check_plt_close(code: str) -> List[str]:
    """Check that plt.close() is called after savefig."""
    savefig_count = len(re.findall(r"\.savefig\s*\(", code))
    close_count = len(re.findall(r"plt\.close\s*\(", code))
    
    if savefig_count > 0 and close_count == 0:
        return ["Missing plt.close() calls after savefig"]
    return []


def validate_generated_code(code: str) -> None:
    """
    Perform all validations on generated code.
    Raises ValueError with detailed report if any validation fails.
    """
    all_errors: List[str] = []

    # 1) AST validation (most critical)
    ast_errors, ast_tree = ast_validate(code)
    all_errors.extend(ast_errors)

    # 2) Required imports
    all_errors.extend(check_required_imports(code))

    # 3) Placeholder detection (uses AST for accuracy)
    all_errors.extend(check_placeholders(code, ast_tree))

    # 4) Required patterns
    all_errors.extend(check_tree_arrays_ak(code))
    all_errors.extend(check_savefig(code))
    all_errors.extend(check_delphes_access(ast_tree))
    all_errors.extend(check_vector_registration(ast_tree, code))

    # 5) Best practices
    all_errors.extend(check_main_guard(code))
    all_errors.extend(check_plt_close(code))

    # 6) Forbidden patterns
    all_errors.extend(check_forbidden_patterns(code))

    # Deduplicate
    seen = set()
    unique_errors = []
    for err in all_errors:
        if err not in seen:
            seen.add(err)
            unique_errors.append(err)

    if unique_errors:
        report = "Generated code validation failed:\n"
        for i, err in enumerate(unique_errors, 1):
            report += f"  {i}. {err}\n"
        raise ValueError(report)


def get_validation_hints(errors: str) -> str:
    """Generate helpful hints based on validation errors."""
    hints = []
    
    if "placeholder" in errors.lower():
        hints.append("Replace ALL {placeholder} patterns with real Python variable names")
    if "register_awkward" in errors.lower():
        hints.append("Add 'vector.register_awkward()' after imports")
    if "library='ak'" in errors.lower():
        hints.append("Use: tree.arrays(branches, library='ak')")
    if "syntax error" in errors.lower():
        hints.append("Check indentation, brackets, and complete statements")
    if "savefig" in errors.lower():
        hints.append("Add plt.savefig('filename.png', dpi=150)")
    if "plt.close" in errors.lower():
        hints.append("Add plt.close() after savefig()")
    if "__main__" in errors.lower():
        hints.append("Add: if __name__ == '__main__': main()")
    
    return "\n".join(f"- {h}" for h in hints) if hints else ""