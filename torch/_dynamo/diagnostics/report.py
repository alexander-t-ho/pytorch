"""
Diagnostic report data structures.

This module defines the data structures used to represent
compilation diagnostics and graph breaks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from torch._dynamo.diagnostics.categories import GraphBreakCategory


@dataclass
class GraphBreak:
    """
    Represents a single graph break event.
    
    Attributes:
        reason: The reason for the graph break
        category: Categorized type of graph break
        location: File location where break occurred (file:line)
        stack_trace: Optional stack trace information
        timestamp: When the break occurred
        metadata: Additional metadata about the break
    """
    reason: str
    category: GraphBreakCategory
    location: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure category is set if not provided."""
        if self.category is None:
            self.category = GraphBreakCategory.categorize(self.reason)


@dataclass
class CompileReport:
    """
    Complete diagnostic report for a torch.compile session.
    
    Attributes:
        function_name: Name of the compiled function
        graph_breaks: List of all graph breaks encountered
        total_compilations: Total number of compilation attempts
        successful_compilations: Number of successful compilations
        failed_compilations: Number of failed compilations
        start_time: When compilation started
        end_time: When compilation ended
        metadata: Additional metadata about the compilation session
    """
    function_name: str
    graph_breaks: List[GraphBreak] = field(default_factory=list)
    total_compilations: int = 0
    successful_compilations: int = 0
    failed_compilations: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def graph_break_count(self) -> int:
        """Total number of graph breaks."""
        return len(self.graph_breaks)
    
    @property
    def break_count_by_category(self) -> Dict[GraphBreakCategory, int]:
        """Count of breaks grouped by category."""
        counts: Dict[GraphBreakCategory, int] = {}
        for gb in self.graph_breaks:
            counts[gb.category] = counts.get(gb.category, 0) + 1
        return counts
    
    @property
    def duration(self) -> Optional[float]:
        """Duration of compilation in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    def add_graph_break(
        self,
        reason: str,
        category: Optional[GraphBreakCategory] = None,
        location: Optional[str] = None,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphBreak:
        """
        Add a graph break to this report.
        
        Args:
            reason: Reason for the graph break
            category: Optional category (will be inferred if not provided)
            location: Optional file location
            stack_trace: Optional stack trace
            metadata: Optional additional metadata
            
        Returns:
            The created GraphBreak object
        """
        if category is None:
            category = GraphBreakCategory.categorize(reason)
        
        graph_break = GraphBreak(
            reason=reason,
            category=category,
            location=location,
            stack_trace=stack_trace,
            metadata=metadata or {},
        )
        self.graph_breaks.append(graph_break)
        return graph_break
    
    def finalize(self) -> None:
        """Mark the report as complete."""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "graph_break_count": self.graph_break_count,
            "total_compilations": self.total_compilations,
            "successful_compilations": self.successful_compilations,
            "failed_compilations": self.failed_compilations,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "break_count_by_category": {
                cat.value: count for cat, count in self.break_count_by_category.items()
            },
            "graph_breaks": [
                {
                    "reason": gb.reason,
                    "category": gb.category.value,
                    "location": gb.location,
                    "timestamp": gb.timestamp.isoformat(),
                    "metadata": gb.metadata,
                }
                for gb in self.graph_breaks
            ],
            "metadata": self.metadata,
        }
    
    def to_html(self, output_path: str) -> None:
        """
        Generate an HTML dashboard report.
        
        Args:
            output_path: Path where HTML file should be written
        """
        from torch._dynamo.diagnostics.html_report import generate_html_report
        generate_html_report(self, output_path)

