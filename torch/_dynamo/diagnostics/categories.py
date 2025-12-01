"""
Graph break categorization system.

This module defines categories for different types of graph breaks
that can occur during torch.compile execution.
"""

from enum import Enum
from typing import Optional


class GraphBreakCategory(Enum):
    """
    Categories of graph breaks that can occur during compilation.
    
    These categories help users understand why their code is breaking
    and what they can do to fix it.
    """
    
    # Control flow breaks
    CONDITIONAL = "conditional"
    LOOP = "loop"
    EXCEPTION_HANDLING = "exception_handling"
    
    # Data-dependent operations
    DATA_DEPENDENT = "data_dependent"
    DYNAMIC_SHAPE = "dynamic_shape"
    
    # Unsupported operations
    UNSUPPORTED_OPERATION = "unsupported_operation"
    UNSUPPORTED_TENSOR_OPERATION = "unsupported_tensor_operation"
    
    # External dependencies
    EXTERNAL_CALL = "external_call"
    PYTHON_BUILTIN = "python_builtin"
    THIRD_PARTY_LIBRARY = "third_party_library"
    
    # Graph construction issues
    GRAPH_CONSTRUCTION_ERROR = "graph_construction_error"
    COMPILATION_ERROR = "compilation_error"
    
    # Unknown/uncategorized
    UNKNOWN = "unknown"
    
    def get_description(self) -> str:
        """Get a human-readable description of this category."""
        descriptions = {
            GraphBreakCategory.CONDITIONAL: "Conditional branch (if/else) detected",
            GraphBreakCategory.LOOP: "Loop detected (for/while)",
            GraphBreakCategory.EXCEPTION_HANDLING: "Exception handling (try/except)",
            GraphBreakCategory.DATA_DEPENDENT: "Data-dependent operation",
            GraphBreakCategory.DYNAMIC_SHAPE: "Dynamic shape operation",
            GraphBreakCategory.UNSUPPORTED_OPERATION: "Unsupported operation",
            GraphBreakCategory.UNSUPPORTED_TENSOR_OPERATION: "Unsupported tensor operation",
            GraphBreakCategory.EXTERNAL_CALL: "External function call",
            GraphBreakCategory.PYTHON_BUILTIN: "Python builtin function call",
            GraphBreakCategory.THIRD_PARTY_LIBRARY: "Third-party library call",
            GraphBreakCategory.GRAPH_CONSTRUCTION_ERROR: "Error during graph construction",
            GraphBreakCategory.COMPILATION_ERROR: "Error during compilation",
            GraphBreakCategory.UNKNOWN: "Unknown reason",
        }
        return descriptions.get(self, "Unknown category")
    
    @classmethod
    def categorize(cls, reason: str, frame_info: Optional[dict] = None) -> "GraphBreakCategory":
        """
        Categorize a graph break based on its reason and frame information.
        
        Args:
            reason: The reason string for the graph break
            frame_info: Optional frame information for more context
            
        Returns:
            The most appropriate category for this graph break
        """
        reason_lower = reason.lower()
        
        # Control flow
        if any(keyword in reason_lower for keyword in ["if", "conditional", "branch"]):
            return cls.CONDITIONAL
        if any(keyword in reason_lower for keyword in ["loop", "for", "while", "iterate"]):
            return cls.LOOP
        if any(keyword in reason_lower for keyword in ["try", "except", "exception"]):
            return cls.EXCEPTION_HANDLING
        
        # Data-dependent
        if any(keyword in reason_lower for keyword in ["data", "dependent", "value"]):
            return cls.DATA_DEPENDENT
        if any(keyword in reason_lower for keyword in ["shape", "dynamic", "size"]):
            return cls.DYNAMIC_SHAPE
        
        # Unsupported operations
        if "unsupported" in reason_lower:
            if "tensor" in reason_lower:
                return cls.UNSUPPORTED_TENSOR_OPERATION
            return cls.UNSUPPORTED_OPERATION
        
        # External calls
        if any(keyword in reason_lower for keyword in ["builtin", "python"]):
            return cls.PYTHON_BUILTIN
        if any(keyword in reason_lower for keyword in ["external", "third", "library"]):
            return cls.THIRD_PARTY_LIBRARY
        if "call" in reason_lower:
            return cls.EXTERNAL_CALL
        
        # Errors
        if "error" in reason_lower or "exception" in reason_lower:
            if "construction" in reason_lower or "graph" in reason_lower:
                return cls.GRAPH_CONSTRUCTION_ERROR
            return cls.COMPILATION_ERROR
        
        return cls.UNKNOWN

