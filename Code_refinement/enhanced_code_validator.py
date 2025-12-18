"""
Enhanced Python Code Validator
==============================
A comprehensive tool to check Python code for syntax errors, bracket matching,
bad expressions, structural issues, security concerns, and best practices.
Useful for validating LLM-generated code.
"""


### I need to modify the imports to gurantee the needed packages are installed.
## Some of the imported packages can't be nstalled by pip install.
'''
 The ast module in Python allows programs to process trees of the Python abstract grammar, known as Abstract Syntax Trees (ASTs). Before the Python interpreter executes code, it first parses the source code into this hierarchical, tree-like data structure that represents the code's structure and meaning.
 
'''
import ast
import re
import sys
import tokenize
import io
import keyword
import builtins
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict, Any 
from enum import Enum
from collections import defaultdict


class ErrorSeverity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    STYLE = "STYLE"


@dataclass #dataclasses decorator 
class ValidationError:
    severity: ErrorSeverity
    category: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self):
        location = ""
        if self.line is not None:
            location = f" (line {self.line}"
            if self.column is not None:
                location += f", col {self.column}"
            location += ")"
        
        result = f"[{self.severity.value}][{self.category}]{location}: {self.message}"
        if self.suggestion:
            result += f"\n    ---> Suggestion: {self.suggestion}"
        return result


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    style: List[ValidationError] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        lines = []
        if self.is_valid:
            lines.append("Code is syntactically valid")
        else:
            lines.append("Code has errors")

        all_issues = [
            ("Errors", self.errors),
            ("Warnings", self.warnings),
            ("Info", self.info),
            ("Style", self.style)
        ]

        for label, issues in all_issues:
            if issues:
                lines.append(f"\n{label}:")
                for issue in issues:
                    lines.append(f"  {issue}")

        if self.stats:
            lines.append(f"\nCode Statistics:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def get_all_issues(self) -> List[ValidationError]:
        return self.errors + self.warnings + self.info + self.style


class CodeValidator:
    """Comprehensive Python code validator."""

    BRACKET_PAIRS = {'(': ')', '[': ']', '{': '}'}
    CLOSING_BRACKETS = set(BRACKET_PAIRS.values())
    BUILTIN_NAMES = set(dir(builtins))

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.info: List[ValidationError] = []
        self.style: List[ValidationError] = []
        self.stats: Dict[str, Any] = {}

    def _add_issue(self, severity: ErrorSeverity, category: str, message: str,
                   line: int = None, column: int = None, code_snippet: str = None,
                   suggestion: str = None):
        """Helper to add validation issues."""
        error = ValidationError(
            severity=severity,
            category=category,
            message=message,
            line=line,
            column=column,
            code_snippet=code_snippet,
            suggestion=suggestion
        )
        if severity == ErrorSeverity.ERROR:
            self.errors.append(error)
        elif severity == ErrorSeverity.WARNING:
            self.warnings.append(error)
        elif severity == ErrorSeverity.INFO:
            self.info.append(error)
        else:
            self.style.append(error)

    def validate(self, code: str) -> ValidationResult:
        """Run all validation checks on the code."""
        self.errors = []
        self.warnings = []
        self.info = []
        self.style = []
        self.stats = {}

        # Basic checks (always run)
        self._check_syntax(code)
        self._check_brackets(code)
        self._check_indentation(code)
        self._check_string_literals(code)

        # Type of checks to be considered (Optional)
        try:
            tree = ast.parse(code)
            self._check_structure(code, tree)
            self._check_naming_conventions(tree)
            self._check_imports(tree)
            self._check_variables(tree, code)
            self._check_functions(tree)
            self._check_classes(tree)
            self._check_exception_handling(tree)
            self._check_loops(tree)
            self._check_comparisons(tree)
            self._check_return_statements(tree)
            self._check_docstrings(tree)
            self._check_dead_code(tree)
            self._check_complexity(tree)
            self._collect_stats(tree, code)
        except SyntaxError:
            pass  # AST checks skipped due to syntax errors

        # Pattern-based checks (can run even with syntax errors)
        self._check_common_issues(code)
        self._check_security_issues(code)
        self._check_string_formatting(code)
        self._check_type_hints(code)
        
        # Do we need to change the structre of the return to be matched with the second LLM ?
        is_valid = len(self.errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=self.errors,
            warnings=self.warnings,
            info=self.info,
            style=self.style,
            stats=self.stats
        )

    # =========================================================================
    # BASIC CHECKS
    # =========================================================================

    def _check_syntax(self, code: str) -> None:
        """Check for Python syntax errors using ast module."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            self._add_issue(
                ErrorSeverity.ERROR, "Syntax",
                f"Syntax error: {e.msg}",
                line=e.lineno, column=e.offset,
                code_snippet=e.text.strip() if e.text else None
            )
        except Exception as e:
            self._add_issue(ErrorSeverity.ERROR, "Syntax", f"Parse error: {str(e)}")

    def _check_brackets(self, code: str) -> None:
        """Check for mismatched brackets, parentheses, and braces."""
        stack: List[Tuple[str, int, int]] = []
        lines = code.split('\n')
        in_string = False
        string_char = None
        in_multiline_string = False

        for line_num, line in enumerate(lines, 1):
            i = 0
            while i < len(line):
                char = line[i]

                if not in_string and char == '#':
                    break

                if i + 2 < len(line) and line[i:i+3] in ('"""', "'''"):
                    if in_multiline_string and line[i:i+3] == string_char:
                        in_multiline_string = False
                        string_char = None
                    elif not in_string:
                        in_multiline_string = True
                        string_char = line[i:i+3]
                    i += 3
                    continue

                if char in ('"', "'") and not in_multiline_string:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char and (i == 0 or line[i-1] != '\\'):
                        in_string = False
                        string_char = None
                    i += 1
                    continue

                if in_string or in_multiline_string:
                    i += 1
                    continue

                if char in self.BRACKET_PAIRS:
                    stack.append((char, line_num, i + 1))
                elif char in self.CLOSING_BRACKETS:
                    if not stack:
                        self._add_issue(
                            ErrorSeverity.ERROR, "Brackets",
                            f"Unmatched closing bracket '{char}'",
                            line=line_num, column=i + 1
                        )
                    else:
                        opening, _, _ = stack.pop()
                        expected = self.BRACKET_PAIRS[opening]
                        if char != expected:
                            self._add_issue(
                                ErrorSeverity.ERROR, "Brackets",
                                f"Mismatched brackets: expected '{expected}', found '{char}'",
                                line=line_num, column=i + 1
                            )
                i += 1

        for bracket, line_num, col in stack:
            self._add_issue(
                ErrorSeverity.ERROR, "Brackets",
                f"Unclosed bracket '{bracket}'",
                line=line_num, column=col
            )

    def _check_indentation(self, code: str) -> None:
        """Check for indentation issues."""
        lines = code.split('\n')
        prev_indent = 0
        indent_char = None
        indent_size = None

        for line_num, line in enumerate(lines, 1):
            if not line.strip() or line.strip().startswith('#'):
                continue

            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if indent > 0:
                first_char = line[0]
                if indent_char is None:
                    indent_char = first_char
                    if first_char == ' ':
                        # Detect indent size
                        indent_size = indent
                elif first_char != indent_char and first_char in (' ', '\t'):
                    self._add_issue(
                        ErrorSeverity.WARNING, "Indentation",
                        "Mixed tabs and spaces in indentation",
                        line=line_num,
                        suggestion="Use consistent indentation (preferably 4 spaces)"
                    )

            if indent - prev_indent > 8:
                self._add_issue(
                    ErrorSeverity.WARNING, "Indentation",
                    f"Large indentation jump ({indent - prev_indent} spaces)",
                    line=line_num
                )

            # Check for odd indentation (not multiple of 4 for spaces)
            if indent_char == ' ' and indent % 4 != 0 and indent > 0:
                self._add_issue(
                    ErrorSeverity.STYLE, "Indentation",
                    f"Non-standard indentation ({indent} spaces)",
                    line=line_num,
                    suggestion="Use 4 spaces per indentation level (PEP 8)"
                )

            prev_indent = indent

    def _check_string_literals(self, code: str) -> None:
        """Check for unterminated string literals."""
        try:
            list(tokenize.generate_tokens(io.StringIO(code).readline))
        except tokenize.TokenError as e:
            self._add_issue(
                ErrorSeverity.ERROR, "String",
                f"Tokenization error: {str(e)}",
                line=e.args[1][0] if len(e.args) > 1 else None
            )

    # =========================================================================
    # AST-BASED CHECKS
    # =========================================================================

    def _check_structure(self, code: str, tree: ast.AST) -> None:
        """Check code structure using AST analysis."""
        max_depth = self._get_max_depth(tree)
        if max_depth > 5:
            self._add_issue(
                ErrorSeverity.WARNING, "Structure",
                f"Code has deep nesting (depth: {max_depth})",
                suggestion="Consider refactoring to reduce nesting"
            )

        # Check for very long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    self._add_issue(
                        ErrorSeverity.INFO, "Structure",
                        f"Function '{node.name}' is {func_lines} lines long",
                        line=node.lineno,
                        suggestion="Consider breaking into smaller functions"
                    )

    def _check_naming_conventions(self, tree: ast.AST) -> None:
        """Check PEP 8 naming conventions."""
        for node in ast.walk(tree):
            # Class names should be PascalCase
            if isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    self._add_issue(
                        ErrorSeverity.STYLE, "Naming",
                        f"Class '{node.name}' should use PascalCase",
                        line=node.lineno,
                        suggestion=f"Rename to '{self._to_pascal_case(node.name)}'"
                    )

            # Function names should be snake_case
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        if node.name not in ('setUp', 'tearDown', 'setUpClass', 'tearDownClass'):
                            self._add_issue(
                                ErrorSeverity.STYLE, "Naming",
                                f"Function '{node.name}' should use snake_case",
                                line=node.lineno,
                                suggestion=f"Rename to '{self._to_snake_case(node.name)}'"
                            )

            # Constants (module-level all caps)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # Check if it shadows a builtin
                        if name in self.BUILTIN_NAMES:
                            self._add_issue(
                                ErrorSeverity.WARNING, "Naming",
                                f"Variable '{name}' shadows a builtin",
                                line=node.lineno,
                                suggestion=f"Use a different name like '{name}_'"
                            )
                        # Single letter names (except common ones)
                        if len(name) == 1 and name not in ('i', 'j', 'k', 'x', 'y', 'z', 'n', '_'):
                            self._add_issue(
                                ErrorSeverity.STYLE, "Naming",
                                f"Single-letter variable name '{name}'",
                                line=node.lineno,
                                suggestion="Use more descriptive names"
                            )

    def _check_imports(self, tree: ast.AST) -> None:
        """Check import statements."""
        imports = []
        import_from = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.append(node)
                for alias in node.names:
                    if alias.name == '*':
                        self._add_issue(
                            ErrorSeverity.WARNING, "Import",
                            "Wildcard import 'import *'",
                            line=node.lineno,
                            suggestion="Import specific names instead"
                        )
            elif isinstance(node, ast.ImportFrom):
                import_from.append(node)
                for alias in node.names:
                    if alias.name == '*':
                        self._add_issue(
                            ErrorSeverity.WARNING, "Import",
                            f"Wildcard import from '{node.module}'",
                            line=node.lineno,
                            suggestion="Import specific names instead"
                        )

        # Check for duplicate imports
        imported_names = defaultdict(list)
        for node in imports + import_from:
            for alias in node.names:
                name = alias.asname or alias.name
                imported_names[name].append(node.lineno)

        for name, lines in imported_names.items():
            if len(lines) > 1:
                self._add_issue(
                    ErrorSeverity.WARNING, "Import",
                    f"'{name}' imported multiple times",
                    line=lines[-1],
                    suggestion=f"Remove duplicate import (also on line {lines[0]})"
                )

        # Check import order (stdlib, third-party, local)
        if imports or import_from:
            all_imports = sorted(imports + import_from, key=lambda n: n.lineno)
            prev_line = 0
            for imp in all_imports:
                if imp.lineno > prev_line + 2 and prev_line > 0:
                    pass  # Gap is okay for grouping
                prev_line = imp.lineno

    def _check_variables(self, tree: ast.AST, code: str) -> None:
        """Check for variable-related issues."""
        # Collect all assigned and used names per scope
        class VariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.scopes = [{'assigned': set(), 'used': set(), 'name': 'module'}]
                self.issues = []

            def visit_FunctionDef(self, node):
                self.scopes.append({
                    'assigned': set(arg.arg for arg in node.args.args),
                    'used': set(),
                    'name': node.name
                })
                self.generic_visit(node)
                scope = self.scopes.pop()
                
                # Check for unused variables (except _ and self)
                unused = scope['assigned'] - scope['used'] - {'self', 'cls', '_'}
                for var in unused:
                    if not var.startswith('_'):
                        self.issues.append((
                            'unused', var, node.lineno, scope['name']
                        ))

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.scopes[-1]['assigned'].add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    self.scopes[-1]['used'].add(node.id)
                    # Also mark as used in parent scopes
                    for scope in self.scopes[:-1]:
                        scope['used'].add(node.id)
                self.generic_visit(node)

        visitor = VariableVisitor()
        visitor.visit(tree)

        for issue_type, var, line, scope_name in visitor.issues:
            if issue_type == 'unused':
                self._add_issue(
                    ErrorSeverity.INFO, "Variable",
                    f"Variable '{var}' is assigned but never used in '{scope_name}'",
                    line=line,
                    suggestion=f"Remove or use the variable, or prefix with '_' if intentional"
                )

    def _check_functions(self, tree: ast.AST) -> None:
        """Check function definitions for issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for mutable default arguments
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is None:
                        continue
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        self._add_issue(
                            ErrorSeverity.WARNING, "Function",
                            f"Mutable default argument in '{node.name}'",
                            line=node.lineno,
                            suggestion="Use None as default and create mutable object inside function"
                        )

                # Check for too many arguments
                total_args = (
                    len(node.args.args) +
                    len(node.args.posonlyargs) +
                    len(node.args.kwonlyargs) +
                    (1 if node.args.vararg else 0) +
                    (1 if node.args.kwarg else 0)
                )
                if total_args > 7:
                    self._add_issue(
                        ErrorSeverity.INFO, "Function",
                        f"Function '{node.name}' has {total_args} parameters",
                        line=node.lineno,
                        suggestion="Consider using a data class or breaking into smaller functions"
                    )

                # Check for duplicate function definitions in same scope
                # (handled elsewhere)

    def _check_classes(self, tree: ast.AST) -> None:
        """Check class definitions for issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                method_names = [m.name for m in methods]

                # Check for duplicate methods
                seen = set()
                for name in method_names:
                    if name in seen:
                        self._add_issue(
                            ErrorSeverity.WARNING, "Class",
                            f"Duplicate method '{name}' in class '{node.name}'",
                            line=node.lineno
                        )
                    seen.add(name)

                # Check for __init__ without super().__init__() in inherited class
                if node.bases and methods:
                    init_method = next((m for m in methods if m.name == '__init__'), None)
                    if init_method:
                        has_super_init = False
                        for child in ast.walk(init_method):
                            if isinstance(child, ast.Call):
                                if isinstance(child.func, ast.Attribute):
                                    if child.func.attr == '__init__':
                                        has_super_init = True
                                        break
                        if not has_super_init:
                            self._add_issue(
                                ErrorSeverity.INFO, "Class",
                                f"Class '{node.name}' inherits but __init__ may not call super().__init__()",
                                line=init_method.lineno,
                                suggestion="Consider calling super().__init__() in __init__"
                            )

    def _check_exception_handling(self, tree: ast.AST) -> None:
        """Check exception handling for issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Bare except
                if node.type is None:
                    self._add_issue(
                        ErrorSeverity.WARNING, "Exception",
                        "Bare 'except:' catches all exceptions including SystemExit and KeyboardInterrupt",
                        line=node.lineno,
                        suggestion="Use 'except Exception:' or catch specific exceptions"
                    )
                # Catching too broad exceptions
                elif isinstance(node.type, ast.Name):
                    if node.type.id == 'BaseException':
                        self._add_issue(
                            ErrorSeverity.WARNING, "Exception",
                            "Catching BaseException is too broad",
                            line=node.lineno,
                            suggestion="Use 'except Exception:' or catch specific exceptions"
                        )

                # Empty except block
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    self._add_issue(
                        ErrorSeverity.WARNING, "Exception",
                        "Empty except block (just 'pass')",
                        line=node.lineno,
                        suggestion="At minimum, log the exception"
                    )

            # Check for raise without exception in except block
            if isinstance(node, ast.Raise):
                if node.exc is None:
                    # This is okay inside an except block, but not outside
                    pass

    def _check_loops(self, tree: ast.AST) -> None:
        """Check loops for common issues."""
        for node in ast.walk(tree):
            # Check for range(len()) antipattern
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        if node.iter.args and isinstance(node.iter.args[0], ast.Call):
                            inner = node.iter.args[0]
                            if isinstance(inner.func, ast.Name) and inner.func.id == 'len':
                                self._add_issue(
                                    ErrorSeverity.STYLE, "Loop",
                                    "Using range(len(x)) antipattern",
                                    line=node.lineno,
                                    suggestion="Use 'for item in x:' or 'for i, item in enumerate(x):'"
                                )

            # Check for while True without break
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                    if not has_break:
                        self._add_issue(
                            ErrorSeverity.WARNING, "Loop",
                            "'while True' loop without 'break' statement",
                            line=node.lineno,
                            suggestion="Ensure there's a way to exit the loop"
                        )

    def _check_comparisons(self, tree: ast.AST) -> None:
        """Check comparisons for issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check for comparison to None using == or !=
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(comparator, ast.Constant) and comparator.value is None:
                        if isinstance(op, ast.Eq):
                            self._add_issue(
                                ErrorSeverity.STYLE, "Comparison",
                                "Comparison to None using '=='",
                                line=node.lineno,
                                suggestion="Use 'is None' instead"
                            )
                        elif isinstance(op, ast.NotEq):
                            self._add_issue(
                                ErrorSeverity.STYLE, "Comparison",
                                "Comparison to None using '!='",
                                line=node.lineno,
                                suggestion="Use 'is not None' instead"
                            )

                    # Check for comparison to True/False
                    if isinstance(comparator, ast.Constant) and comparator.value in (True, False):
                        if isinstance(op, (ast.Eq, ast.NotEq)):
                            self._add_issue(
                                ErrorSeverity.STYLE, "Comparison",
                                f"Comparison to {comparator.value} using '==' or '!='",
                                line=node.lineno,
                                suggestion="Use 'if x:' or 'if not x:' instead"
                            )

                # Check for type() comparison
                if isinstance(node.left, ast.Call):
                    if isinstance(node.left.func, ast.Name) and node.left.func.id == 'type':
                        self._add_issue(
                            ErrorSeverity.STYLE, "Comparison",
                            "Using type() for comparison",
                            line=node.lineno,
                            suggestion="Use isinstance() instead"
                        )

                # Self comparison (x == x)
                if len(node.comparators) == 1:
                    if ast.dump(node.left) == ast.dump(node.comparators[0]):
                        self._add_issue(
                            ErrorSeverity.WARNING, "Comparison",
                            "Comparing variable to itself",
                            line=node.lineno,
                            suggestion="This is always True/False - likely a bug"
                        )

    def _check_return_statements(self, tree: ast.AST) -> None:
        """Check return statements for consistency."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                returns = []

                class ReturnVisitor(ast.NodeVisitor):
                    def visit_Return(self, ret_node):
                        returns.append(ret_node)

                    def visit_FunctionDef(self, func_node):
                        pass  # Don't recurse into nested functions

                ReturnVisitor().visit(node)

                if returns:
                    has_value = [r.value is not None for r in returns]
                    if any(has_value) and not all(has_value):
                        self._add_issue(
                            ErrorSeverity.WARNING, "Return",
                            f"Inconsistent return statements in '{node.name}'",
                            line=node.lineno,
                            suggestion="All return statements should return a value or none should"
                        )

    def _check_docstrings(self, tree: ast.AST) -> None:
        """Check for missing docstrings."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                
                if isinstance(node, ast.Module):
                    if not docstring:
                        self._add_issue(
                            ErrorSeverity.INFO, "Docstring",
                            "Module is missing a docstring",
                            line=1
                        )
                elif isinstance(node, ast.ClassDef):
                    if not docstring:
                        self._add_issue(
                            ErrorSeverity.INFO, "Docstring",
                            f"Class '{node.name}' is missing a docstring",
                            line=node.lineno
                        )
                elif isinstance(node, ast.FunctionDef):
                    # Skip private methods and simple methods
                    if not node.name.startswith('_') and len(node.body) > 3:
                        if not docstring:
                            self._add_issue(
                                ErrorSeverity.INFO, "Docstring",
                                f"Function '{node.name}' is missing a docstring",
                                line=node.lineno
                            )

    def _check_dead_code(self, tree: ast.AST) -> None:
        """Check for unreachable code."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
                body = getattr(node, 'body', [])
                for i, stmt in enumerate(body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                        next_stmt = body[i + 1]
                        self._add_issue(
                            ErrorSeverity.WARNING, "DeadCode",
                            "Unreachable code after return/raise/break/continue",
                            line=next_stmt.lineno if hasattr(next_stmt, 'lineno') else None,
                            suggestion="Remove unreachable code"
                        )
                        break

    def _check_complexity(self, tree: ast.AST) -> None:
        """Check cyclomatic complexity of functions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    self._add_issue(
                        ErrorSeverity.WARNING, "Complexity",
                        f"Function '{node.name}' has high cyclomatic complexity ({complexity})",
                        line=node.lineno,
                        suggestion="Consider breaking into smaller functions"
                    )

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.Assert, ast.comprehension)):
                complexity += 1
        return complexity

    # =========================================================================
    # PATTERN-BASED CHECKS
    # =========================================================================

    def _check_common_issues(self, code: str) -> None:
        """Check for common coding issues using regex patterns."""
        lines = code.split('\n')

        patterns = [
            (r'print\s+[^(]', "Python 2 style print statement", ErrorSeverity.ERROR, "Add parentheses: print(...)"),
            (r'\bexec\s*\(', "Use of exec() can be dangerous", ErrorSeverity.WARNING, "Avoid exec() if possible"),
            (r'\beval\s*\(', "Use of eval() can be dangerous", ErrorSeverity.WARNING, "Consider safer alternatives like ast.literal_eval()"),
            (r'\bglobals\s*\(\s*\)\s*\[', "Modifying globals() dict", ErrorSeverity.WARNING, "Avoid modifying globals directly"),
            (r'assert\s+False', "assert False will always fail", ErrorSeverity.WARNING, "Use raise AssertionError or appropriate exception"),
            (r'#\s*TODO', "TODO comment found", ErrorSeverity.INFO, None),
            (r'#\s*FIXME', "FIXME comment found", ErrorSeverity.INFO, None),
            (r'#\s*HACK', "HACK comment found", ErrorSeverity.WARNING, None),
            (r'#\s*XXX', "XXX comment found", ErrorSeverity.WARNING, None),
            (r'\.has_key\s*\(', "dict.has_key() is Python 2", ErrorSeverity.ERROR, "Use 'key in dict' instead"),
            (r'\bprint\s*>>', "Python 2 print redirection", ErrorSeverity.ERROR, "Use print(..., file=...) instead"),
            (r'except\s+\w+\s*,\s*\w+:', "Python 2 except syntax", ErrorSeverity.ERROR, "Use 'except Error as e:' instead"),
        ]

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                # Still check TODO/FIXME in comments
                for pattern, message, severity, suggestion in patterns:
                    if 'TODO' in pattern or 'FIXME' in pattern or 'HACK' in pattern or 'XXX' in pattern:
                        if re.search(pattern, line, re.IGNORECASE):
                            self._add_issue(severity, "Comment", message, line=line_num)
                continue

            for pattern, message, severity, suggestion in patterns:
                if re.search(pattern, line):
                    self._add_issue(
                        severity, "Pattern",
                        message,
                        line=line_num,
                        code_snippet=stripped[:60],
                        suggestion=suggestion
                    )

    def _check_security_issues(self, code: str) -> None:
        """Check for potential security issues."""
        lines = code.split('\n')

        security_patterns = [
            (r'pickle\.loads?\s*\(', "Pickle can execute arbitrary code", "Avoid unpickling untrusted data"),
            (r'subprocess.*shell\s*=\s*True', "shell=True can be a security risk", "Use shell=False with a list of arguments"),
            (r'os\.system\s*\(', "os.system() is vulnerable to shell injection", "Use subprocess with shell=False"),
            (r'yaml\.load\s*\([^)]*\)', "yaml.load() without Loader is unsafe", "Use yaml.safe_load() or specify Loader"),
            (r'__import__\s*\(', "Dynamic import can be dangerous", "Use static imports when possible"),
            (r'input\s*\(\s*\)', "input() returns string in Python 3", None),
            (r'(password|secret|api_key|apikey|token)\s*=\s*["\'][^"\']+["\']', 
             "Possible hardcoded secret", "Use environment variables or config files"),
            (r'md5\s*\(|hashlib\.md5', "MD5 is cryptographically weak", "Use SHA-256 or stronger for security"),
            (r'random\.(random|randint|choice|shuffle)', "random module not cryptographically secure", 
             "Use secrets module for security-sensitive operations"),
            (r'chmod\s*\(\s*[^,]+,\s*0?777\s*\)', "Setting 777 permissions is insecure", "Use more restrictive permissions"),
            (r'(SELECT|INSERT|UPDATE|DELETE).*%s', "Possible SQL injection with string formatting", 
             "Use parameterized queries"),
            (r'\.format\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE)', "Possible SQL injection with .format()", 
             "Use parameterized queries"),
            (r'f["\'].*{.*}.*(?:SELECT|INSERT|UPDATE|DELETE)', "Possible SQL injection with f-string", 
             "Use parameterized queries"),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, message, suggestion in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self._add_issue(
                        ErrorSeverity.WARNING, "Security",
                        message,
                        line=line_num,
                        suggestion=suggestion
                    )

    def _check_string_formatting(self, code: str) -> None:
        """Check string formatting for issues."""
        lines = code.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Empty f-string
            if re.search(r'f["\']["\']', line):
                self._add_issue(
                    ErrorSeverity.WARNING, "String",
                    "Empty f-string",
                    line=line_num,
                    suggestion="Remove the 'f' prefix or add placeholders"
                )

            # f-string without placeholders
            if re.search(r'f["\'][^{}]+["\']', line):
                # Make sure it's not inside a larger f-string with placeholders
                f_strings = re.findall(r'f(["\'])([^"\']*)\1', line)
                for _, content in f_strings:
                    if '{' not in content and '}' not in content:
                        self._add_issue(
                            ErrorSeverity.INFO, "String",
                            "f-string without placeholders",
                            line=line_num,
                            suggestion="Remove the 'f' prefix if no formatting needed"
                        )

            # Mixed formatting styles
            has_percent = re.search(r'%[sd]', line) and '%' in line
            has_format = '.format(' in line
            has_fstring = re.search(r'f["\']', line)

            styles = sum([bool(has_percent), bool(has_format), bool(has_fstring)])
            if styles > 1:
                self._add_issue(
                    ErrorSeverity.STYLE, "String",
                    "Mixed string formatting styles",
                    line=line_num,
                    suggestion="Use f-strings consistently (Python 3.6+)"
                )

    def _check_type_hints(self, code: str) -> None:
        """Check type hints usage."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        functions_with_hints = 0
        functions_without_hints = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_hints = (
                    node.returns is not None or
                    any(arg.annotation is not None for arg in node.args.args)
                )
                if has_hints:
                    functions_with_hints += 1
                else:
                    functions_without_hints += 1

        # Warn about inconsistent type hint usage
        if functions_with_hints > 0 and functions_without_hints > 0:
            self._add_issue(
                ErrorSeverity.STYLE, "TypeHints",
                f"Inconsistent type hint usage: {functions_with_hints} functions with hints, {functions_without_hints} without",
                suggestion="Consider adding type hints to all functions for consistency"
            )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of the AST."""
        max_depth = current_depth
        depth_increasing = (ast.If, ast.For, ast.While, ast.With, ast.Try,
                           ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

        for child in ast.iter_child_nodes(node):
            if isinstance(child, depth_increasing):
                child_depth = self._get_max_depth(child, current_depth + 1)
            else:
                child_depth = self._get_max_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)

        return max_depth

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_pascal_case(self, name: str) -> str:
        """Convert name to PascalCase."""
        return ''.join(word.capitalize() for word in name.replace('_', ' ').split())

    def _collect_stats(self, tree: ast.AST, code: str) -> None:
        """Collect code statistics."""
        lines = code.split('\n')
        
        self.stats['total_lines'] = len(lines)
        self.stats['blank_lines'] = sum(1 for line in lines if not line.strip())
        self.stats['comment_lines'] = sum(1 for line in lines if line.strip().startswith('#'))
        self.stats['code_lines'] = self.stats['total_lines'] - self.stats['blank_lines'] - self.stats['comment_lines']

        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

        self.stats['classes'] = len(classes)
        self.stats['functions'] = len(functions)
        self.stats['imports'] = len(imports)


def validate_code(code: str, config: Optional[Dict] = None) -> ValidationResult:
    """Convenience function to validate Python code."""
    validator = CodeValidator(config)
    return validator.validate(code)


def validate_file(filepath: str, config: Optional[Dict] = None) -> ValidationResult:
    """Validate Python code from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    return validate_code(code, config)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Example code with various issues for testing
    test_code = '''
import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

# Physics constants
MASS_ELECTRON = 0.000511  # GeV
MASS_MUON = 0.10566       # GeV

def invariant_mass_2body(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
    """Calculate invariant mass from PT, Eta, Phi."""
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + mass1**2)
    
    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + mass2**2)
    
    E_tot = E1 + E2
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2
    
    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
    return np.sqrt(np.maximum(m2, 0))

def main():
    # ==========================================
    # LOAD DATA
    # ==========================================
    file = uproot.open("signal.root")
    tree = file["Delphes"]
    
    electrons = tree.arrays([
        "Electron/Electron.PT",
        "Electron/Electron.Eta",
        "Electron/Electron.Phi",
        "Electron/Electron.Charge"
    ])
    
    muons = tree.arrays([
        "Muon/Muon.PT",
        "Muon/Muon.Eta",
        "Muon/Muon.Phi",
        "Muon/Muon.Charge"
    ])
    
    # ==========================================
    # SELECTION CUTS
    # ==========================================
    total_events = len(electrons["Electron/Electron.PT"])
    
    # Object-level cuts
    obj_mask = (electrons["Electron/Electron.PT"] > 20) & (np.abs(electrons["Electron/Electron.Eta"]) < 2.4)
    sel_electrons = electrons[obj_mask]
    
    # Event-level cuts
    n_electrons = ak.num(sel_electrons)
    event_mask_2e = n_electrons == 2
    leading_electron_mask = sel_electrons["Electron/Electron.PT"] > 25
    subleading_electron_mask = ak.num(sel_electrons) > 1
    eta_mask = (electrons["Electron/Electron.Eta"]) > -2.5 & (electrons["Electron/Electron.Eta"]) < 2.5
    
    pt_2e = sel_electrons["Electron/Electron.PT"][event_mask_2e & leading_electron_mask]
    eta_2e = sel_electrons["Electron/Electron.Eta"][event_mask_2e & leading_electron_mask]
    phi_2e = sel_electrons["Electron/Electron.Phi"][event_mask_2e & leading_electron_mask]
    charge_2e = sel_electrons["Electron/Electron.Charge"][event_mask_2e & leading_electron_mask]
    
    pt_os = pt_2e[subleading_electron_mask]
    eta_os = eta_2e[subleading_electron_mask]
    phi_os = phi_2e[subleading_electron_mask]
    
    events_with_2_electrons = ak.sum(event_mask_2e)
    events_opposite_sign = ak.sum(charge_2e[:, 0] * charge_2e[:, 1] < 0)
    
    # ==========================================
    # CALCULATE OBSERVABLES
    # ==========================================
    # Sort by PT to get leading/subleading
    sorted_indices = ak.argsort(pt_os, ascending=False)
    pt_sorted = pt_os[sorted_indices]
    eta_sorted = eta_os[sorted_indices]
    phi_sorted = phi_os[sorted_indices]
    
    pt1 = pt_sorted[:, 0]
    eta1 = eta_sorted[:, 0]
    phi1 = phi_sorted[:, 0]
    
    pt2 = pt_sorted[:, 1]
    eta2 = eta_sorted[:, 1]
    phi2 = phi_sorted[:, 1]
    
    # Calculate invariant mass
    mee = invariant_mass_2body(pt1, eta1, phi1, MASS_ELECTRON,
                               pt2, eta2, phi2, MASS_ELECTRON)
    
    # ==========================================
    # PLOTS FOR VALIDATION
    # ==========================================
    # Plot 1: Dielectron invariant mass
    plt.figure(figsize=(10, 6))
    mee_numpy = ak.to_numpy(mee)
    plt.hist(mee_numpy, bins=60, range=(60, 120), 
             histtype='step', linewidth=2, color='blue')
    plt.xlabel('$m_{ee}$ [GeV]', fontsize=12)
    plt.ylabel('Events', fontsize=12)
    plt.title('Dielectron Invariant Mass')
    plt.savefig('dielectron_invariant_mass.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Leading electron PT
    plt.figure(figsize=(10, 6))
    pt1_numpy = ak.to_numpy(pt1)
    plt.hist(pt1_numpy, bins=50, range=(0, 150),
             histtype='step', linewidth=2, color='blue')
    plt.xlabel('Leading Electron $p_T$ [GeV]', fontsize=12)
    plt.ylabel('Events', fontsize=12)
    plt.title('Leading Electron $p_T$ Distribution')
    plt.savefig('leading_electron_pt.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Subleading electron PT
    plt.figure(figsize=(10, 6))
    pt_sorted_subleading = pt_sorted[:, 1]
    pt_subleading_numpy = ak.to_numpy(pt_sorted_subleading)
    plt.hist(pt_subleading_numpy, bins=50, range=(0, 100),
             histtype='step', linewidth=2, color='red')
    plt.xlabel('Subleading Electron $p_T$ [GeV]', fontsize=12)
    plt.ylabel('Events', fontsize=12)
    plt.title('Subleading Electron $p_T$ Distribution')
    plt.savefig('subleading_electron_pt.png', dpi=150, bbox_inches='tight')
    
    # Plot 4: Electron Eta distribution
    plt.figure(figsize=(10, 6))
    eta_flat = ak.flatten(electrons["Electron/Electron.Eta"][obj_mask])
    eta_numpy = ak.to_numpy(eta_flat)
    plt.hist(eta_numpy, bins=50, range=(-2.5, 2.5),
             histtype='step', linewidth=2, color='green')
    plt.xlabel('Electron $\\eta$', fontsize=12)
    plt.ylabel('Electrons', fontsize=12)
    plt.title('Electron $\\eta$ Distribution')
    plt.savefig('electron_eta.png', dpi=150, bbox_inches='tight')
    
    # ==========================================
    # OUTPUT STRUCTURE
    # ==========================================
    # Cut flow
    print("=" * 50)
    print("CUT FLOW")
    print("=" * 50)
    print(f"Total events:                    {total_events}")
    print(f"Events with selected electrons:  {events_with_selected_electrons}")
    print(f"Events with exactly 2 electrons: {events_with_2_electrons}")
    print(f"Events with opposite-sign pair:  {events_opposite_sign}")
    print("=" * 50)
    
    # Invariant mass statistics
    print("\nINVARIANT MASS STATISTICS")
    print("=" * 50)
    print(f"Mean m_ee:  {np.mean(mee_numpy):.2f} GeV")
    print(f"RMS m_ee:   {np.std(mee_numpy):.2f} GeV")
    print("=" * 50)
    
    print("\nPlots saved:")
    print("  - dielectron_invariant_mass.png")
    print("  - leading_electron_pt.png")
    print("  - subleading_electron_pt.png")
    print("  - electron_eta.png")

if __name__ == "__main__":
    main()
    '''


    result = validate_code(test_code)
    print(result)

    
