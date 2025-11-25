"""
torch.compile Diagnostic Dashboard

This module provides diagnostic tools for analyzing torch.compile behavior,
including graph break detection, categorization, and visualization.

Public API:
    - CompileDiagnostics: Context manager for collecting diagnostics
    - CompileReport: Report data structure
    - GraphBreakCategory: Enumeration of graph break categories
"""

from torch._dynamo.diagnostics.collector import CompileDiagnostics
from torch._dynamo.diagnostics.report import CompileReport, GraphBreak
from torch._dynamo.diagnostics.categories import GraphBreakCategory

__all__ = [
    "CompileDiagnostics",
    "CompileReport",
    "GraphBreak",
    "GraphBreakCategory",
]

