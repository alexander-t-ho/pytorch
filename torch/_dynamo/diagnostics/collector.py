"""
Diagnostic event collector.

This module provides the CompileDiagnostics context manager
for collecting diagnostic information during torch.compile execution.
"""

import contextlib
import traceback
from typing import Optional, Dict, Any
from threading import local

from torch._dynamo.diagnostics.report import CompileReport, GraphBreak
from torch._dynamo.diagnostics.categories import GraphBreakCategory


# Thread-local storage for diagnostic context
_thread_local = local()


class CompileDiagnostics:
    """
    Context manager for collecting torch.compile diagnostics.
    
    Usage:
        with CompileDiagnostics() as diag:
            compiled_fn = torch.compile(my_function)
            result = compiled_fn(input)
        
        report = diag.get_report()
        report.to_html("diagnostics.html")
    """
    
    def __init__(self, function_name: Optional[str] = None, enable: bool = True):
        """
        Initialize diagnostic collector.
        
        Args:
            function_name: Optional name for the function being compiled
            enable: Whether to enable diagnostics (can be disabled for performance)
        """
        self.function_name = function_name or "unknown_function"
        self.enable = enable
        self._report: Optional[CompileReport] = None
        self._previous_context: Optional[CompileDiagnostics] = None
    
    def __enter__(self) -> "CompileDiagnostics":
        """Enter diagnostic collection context."""
        if not self.enable:
            return self
        
        # Store previous context if nested
        if hasattr(_thread_local, "current_diagnostics"):
            self._previous_context = _thread_local.current_diagnostics
        
        # Set this as the current context
        _thread_local.current_diagnostics = self
        
        # Create report
        self._report = CompileReport(function_name=self.function_name)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit diagnostic collection context."""
        if not self.enable:
            return False
        
        # Finalize report
        if self._report is not None:
            self._report.finalize()
        
        # Restore previous context if nested
        if self._previous_context is not None:
            _thread_local.current_diagnostics = self._previous_context
        elif hasattr(_thread_local, "current_diagnostics"):
            delattr(_thread_local, "current_diagnostics")
        
        return False  # Don't suppress exceptions
    
    def get_report(self) -> Optional[CompileReport]:
        """Get the diagnostic report."""
        return self._report
    
    def record_graph_break(
        self,
        reason: str,
        category: Optional[GraphBreakCategory] = None,
        location: Optional[str] = None,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a graph break event.
        
        Args:
            reason: Reason for the graph break
            category: Optional category (will be inferred if not provided)
            location: Optional file location
            stack_trace: Optional stack trace
            metadata: Optional additional metadata
        """
        if not self.enable or self._report is None:
            return
        
        self._report.add_graph_break(
            reason=reason,
            category=category,
            location=location,
            stack_trace=stack_trace,
            metadata=metadata,
        )
    
    def record_compilation_attempt(self, successful: bool = True) -> None:
        """
        Record a compilation attempt.
        
        Args:
            successful: Whether the compilation was successful
        """
        if not self.enable or self._report is None:
            return
        
        self._report.total_compilations += 1
        if successful:
            self._report.successful_compilations += 1
        else:
            self._report.failed_compilations += 1


def get_current_diagnostics() -> Optional[CompileDiagnostics]:
    """
    Get the current active CompileDiagnostics context.
    
    Returns:
        The current CompileDiagnostics instance, or None if not in a context
    """
    return getattr(_thread_local, "current_diagnostics", None)


def record_graph_break(
    reason: str,
    category: Optional[GraphBreakCategory] = None,
    location: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record a graph break in the current diagnostic context.
    
    This is a convenience function that can be called from anywhere
    in the codebase to record graph breaks.
    
    Args:
        reason: Reason for the graph break
        category: Optional category (will be inferred if not provided)
        location: Optional file location
        metadata: Optional additional metadata
    """
    diag = get_current_diagnostics()
    if diag is not None:
        # Capture stack trace
        stack_trace = "".join(traceback.format_stack()[:-1])
        diag.record_graph_break(
            reason=reason,
            category=category,
            location=location,
            stack_trace=stack_trace,
            metadata=metadata,
        )

