"""
Suggestion engine for graph breaks.

This module provides suggestions for how to fix or avoid graph breaks
based on their category and context.
"""

from typing import List, Dict, Any, Optional
from torch._dynamo.diagnostics.categories import GraphBreakCategory
from torch._dynamo.diagnostics.report import GraphBreak, CompileReport


class SuggestionEngine:
    """
    Engine for generating suggestions based on graph breaks.
    """
    
    # Category-based suggestions
    CATEGORY_SUGGESTIONS: Dict[GraphBreakCategory, List[str]] = {
        GraphBreakCategory.CONDITIONAL: [
            "Consider using torch.where() or torch.select() instead of if/else",
            "Use torch.jit.script() for control flow if needed",
            "Try to make conditions tensor-based rather than Python bools",
        ],
        GraphBreakCategory.LOOP: [
            "Consider using vectorized operations instead of loops",
            "Use torch operations like torch.cumsum(), torch.cumprod() for accumulations",
            "For fixed-size loops, unroll them manually or use torch.jit.script()",
        ],
        GraphBreakCategory.DATA_DEPENDENT: [
            "Try to make operations shape-invariant",
            "Use torch.jit.script() for data-dependent control flow",
            "Consider restructuring to avoid data-dependent branches",
        ],
        GraphBreakCategory.DYNAMIC_SHAPE: [
            "Try to use fixed shapes where possible",
            "Consider using torch.jit.script() for dynamic shapes",
            "Use torch.compile with dynamic=True if shapes vary",
        ],
        GraphBreakCategory.UNSUPPORTED_OPERATION: [
            "Check if there's a torch equivalent for this operation",
            "Consider using torch.jit.script() for unsupported operations",
            "Move unsupported operations outside the compiled function",
        ],
        GraphBreakCategory.EXTERNAL_CALL: [
            "Move external function calls outside the compiled function",
            "Consider wrapping external calls in torch.jit.script()",
            "Use torch operations instead of external libraries where possible",
        ],
        GraphBreakCategory.PYTHON_BUILTIN: [
            "Use torch equivalents: torch.sum() instead of sum(), torch.len() instead of len()",
            "Move Python builtin calls outside the compiled function",
            "Consider using torch.jit.script() for Python builtin usage",
        ],
    }
    
    @classmethod
    def get_suggestions(
        cls,
        graph_break: GraphBreak,
        report: Optional[CompileReport] = None,
    ) -> List[str]:
        """
        Get suggestions for a specific graph break.
        
        Args:
            graph_break: The graph break to get suggestions for
            report: Optional full report for context-aware suggestions
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Category-based suggestions
        category_suggestions = cls.CATEGORY_SUGGESTIONS.get(
            graph_break.category, []
        )
        suggestions.extend(category_suggestions)
        
        # Context-aware suggestions based on report
        if report is not None:
            # If many breaks of same category, suggest pattern fix
            category_count = report.break_count_by_category.get(
                graph_break.category, 0
            )
            if category_count > 3:
                suggestions.append(
                    f"Multiple {graph_break.category.value} breaks detected. "
                    "Consider refactoring to address the pattern."
                )
            
            # If many breaks overall, suggest general optimization
            if report.graph_break_count > 10:
                suggestions.append(
                    "Many graph breaks detected. Consider breaking the function "
                    "into smaller, more compilable pieces."
                )
        
        # Metadata-based suggestions
        if graph_break.metadata:
            if "operation" in graph_break.metadata:
                op_name = graph_break.metadata["operation"]
                suggestions.append(
                    f"Operation '{op_name}' may not be fully supported. "
                    "Check PyTorch documentation for alternatives."
                )
        
        return suggestions if suggestions else [
            "Review the graph break reason and consider refactoring the code."
        ]
    
    @classmethod
    def get_summary_suggestions(cls, report: CompileReport) -> List[str]:
        """
        Get high-level suggestions based on the entire report.
        
        Args:
            report: The complete compilation report
            
        Returns:
            List of summary suggestion strings
        """
        suggestions = []
        
        # Overall break count
        if report.graph_break_count == 0:
            suggestions.append("✅ No graph breaks detected! Compilation is optimal.")
        elif report.graph_break_count < 3:
            suggestions.append(
                f"⚠️ {report.graph_break_count} graph break(s) detected. "
                "Consider reviewing and optimizing."
            )
        else:
            suggestions.append(
                f"⚠️ {report.graph_break_count} graph breaks detected. "
                "Significant optimization opportunities exist."
            )
        
        # Category distribution
        break_counts = report.break_count_by_category
        if break_counts:
            most_common = max(break_counts.items(), key=lambda x: x[1])
            suggestions.append(
                f"Most common break type: {most_common[0].value} "
                f"({most_common[1]} occurrence(s))"
            )
        
        # Compilation success rate
        if report.total_compilations > 0:
            success_rate = report.successful_compilations / report.total_compilations
            if success_rate < 0.5:
                suggestions.append(
                    "⚠️ Low compilation success rate. "
                    "Many compilation attempts are failing."
                )
            elif success_rate == 1.0:
                suggestions.append("✅ All compilation attempts succeeded.")
        
        return suggestions

