"""
Tests for torch.compile diagnostic dashboard.

This test suite verifies the functionality of the diagnostic
collection and reporting system.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
import os

# These imports will work once integrated into PyTorch
try:
    from torch._dynamo.diagnostics import (
        CompileDiagnostics,
        CompileReport,
        GraphBreak,
        GraphBreakCategory,
    )
    from torch._dynamo.diagnostics.collector import record_graph_break
    from torch._dynamo.diagnostics.suggestions import SuggestionEngine
except ImportError:
    # For standalone testing
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    from torch._dynamo.diagnostics import (
        CompileDiagnostics,
        CompileReport,
        GraphBreak,
        GraphBreakCategory,
    )
    from torch._dynamo.diagnostics.collector import record_graph_break
    from torch._dynamo.diagnostics.suggestions import SuggestionEngine


class TestGraphBreakCategory(unittest.TestCase):
    """Test graph break categorization."""
    
    def test_category_descriptions(self):
        """Test that all categories have descriptions."""
        for category in GraphBreakCategory:
            desc = category.get_description()
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0)
    
    def test_categorize_conditional(self):
        """Test categorization of conditional breaks."""
        cat = GraphBreakCategory.categorize("if statement detected")
        self.assertEqual(cat, GraphBreakCategory.CONDITIONAL)
    
    def test_categorize_loop(self):
        """Test categorization of loop breaks."""
        cat = GraphBreakCategory.categorize("for loop detected")
        self.assertEqual(cat, GraphBreakCategory.LOOP)
    
    def test_categorize_unsupported(self):
        """Test categorization of unsupported operations."""
        cat = GraphBreakCategory.categorize("unsupported operation: some_op")
        self.assertEqual(cat, GraphBreakCategory.UNSUPPORTED_OPERATION)
    
    def test_categorize_unknown(self):
        """Test categorization of unknown breaks."""
        cat = GraphBreakCategory.categorize("mysterious reason")
        self.assertEqual(cat, GraphBreakCategory.UNKNOWN)


class TestGraphBreak(unittest.TestCase):
    """Test GraphBreak data structure."""
    
    def test_graph_break_creation(self):
        """Test creating a graph break."""
        gb = GraphBreak(
            reason="test reason",
            category=GraphBreakCategory.CONDITIONAL,
            location="test.py:42",
        )
        self.assertEqual(gb.reason, "test reason")
        self.assertEqual(gb.category, GraphBreakCategory.CONDITIONAL)
        self.assertEqual(gb.location, "test.py:42")
        self.assertIsInstance(gb.timestamp, datetime)
    
    def test_graph_break_auto_category(self):
        """Test automatic categorization."""
        gb = GraphBreak(reason="if statement", category=None)
        # Category should be set in __post_init__
        self.assertIsNotNone(gb.category)


class TestCompileReport(unittest.TestCase):
    """Test CompileReport data structure."""
    
    def test_report_creation(self):
        """Test creating a report."""
        report = CompileReport(function_name="test_function")
        self.assertEqual(report.function_name, "test_function")
        self.assertEqual(report.graph_break_count, 0)
        self.assertEqual(report.total_compilations, 0)
    
    def test_add_graph_break(self):
        """Test adding graph breaks to report."""
        report = CompileReport(function_name="test_function")
        gb = report.add_graph_break(
            reason="test break",
            category=GraphBreakCategory.CONDITIONAL,
        )
        self.assertEqual(report.graph_break_count, 1)
        self.assertIn(gb, report.graph_breaks)
    
    def test_break_count_by_category(self):
        """Test counting breaks by category."""
        report = CompileReport(function_name="test_function")
        report.add_graph_break("break 1", GraphBreakCategory.CONDITIONAL)
        report.add_graph_break("break 2", GraphBreakCategory.CONDITIONAL)
        report.add_graph_break("break 3", GraphBreakCategory.LOOP)
        
        counts = report.break_count_by_category
        self.assertEqual(counts[GraphBreakCategory.CONDITIONAL], 2)
        self.assertEqual(counts[GraphBreakCategory.LOOP], 1)
    
    def test_finalize(self):
        """Test finalizing a report."""
        report = CompileReport(function_name="test_function")
        self.assertIsNone(report.end_time)
        report.finalize()
        self.assertIsNotNone(report.end_time)
        self.assertIsNotNone(report.duration)
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        report = CompileReport(function_name="test_function")
        report.add_graph_break("test", GraphBreakCategory.CONDITIONAL)
        report.finalize()
        
        data = report.to_dict()
        self.assertEqual(data["function_name"], "test_function")
        self.assertEqual(data["graph_break_count"], 1)
        self.assertIn("graph_breaks", data)
        self.assertIn("start_time", data)
        self.assertIn("end_time", data)


class TestCompileDiagnostics(unittest.TestCase):
    """Test CompileDiagnostics context manager."""
    
    def test_context_manager(self):
        """Test using CompileDiagnostics as context manager."""
        with CompileDiagnostics(function_name="test_fn") as diag:
            self.assertIsNotNone(diag.get_report())
            diag.record_graph_break("test reason", GraphBreakCategory.CONDITIONAL)
        
        report = diag.get_report()
        self.assertIsNotNone(report)
        self.assertEqual(report.graph_break_count, 1)
        self.assertIsNotNone(report.end_time)
    
    def test_record_compilation_attempt(self):
        """Test recording compilation attempts."""
        with CompileDiagnostics() as diag:
            diag.record_compilation_attempt(successful=True)
            diag.record_compilation_attempt(successful=False)
        
        report = diag.get_report()
        self.assertEqual(report.total_compilations, 2)
        self.assertEqual(report.successful_compilations, 1)
        self.assertEqual(report.failed_compilations, 1)
    
    def test_disabled_diagnostics(self):
        """Test disabled diagnostics."""
        with CompileDiagnostics(enable=False) as diag:
            diag.record_graph_break("test")
        
        report = diag.get_report()
        self.assertIsNone(report)


class TestSuggestionEngine(unittest.TestCase):
    """Test suggestion engine."""
    
    def test_get_suggestions(self):
        """Test getting suggestions for a graph break."""
        gb = GraphBreak(
            reason="if statement",
            category=GraphBreakCategory.CONDITIONAL,
        )
        suggestions = SuggestionEngine.get_suggestions(gb)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
    
    def test_get_summary_suggestions(self):
        """Test getting summary suggestions."""
        report = CompileReport(function_name="test")
        report.add_graph_break("break 1", GraphBreakCategory.CONDITIONAL)
        report.finalize()
        
        suggestions = SuggestionEngine.get_summary_suggestions(report)
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)


class TestHTMLReport(unittest.TestCase):
    """Test HTML report generation."""
    
    def test_html_generation(self):
        """Test generating HTML report."""
        report = CompileReport(function_name="test_function")
        report.add_graph_break("test break", GraphBreakCategory.CONDITIONAL)
        report.finalize()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            report.to_html(output_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Check that HTML content is valid
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn("torch.compile Diagnostics", content)
                self.assertIn("test_function", content)
                self.assertIn("test break", content)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete diagnostic workflow."""
        with CompileDiagnostics(function_name="integration_test") as diag:
            # Simulate some graph breaks
            diag.record_graph_break("if x > 0", GraphBreakCategory.CONDITIONAL)
            diag.record_graph_break("for i in range(10)", GraphBreakCategory.LOOP)
            diag.record_compilation_attempt(successful=True)
        
        report = diag.get_report()
        self.assertEqual(report.graph_break_count, 2)
        self.assertEqual(report.total_compilations, 1)
        
        # Test HTML export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            report.to_html(output_path)
            self.assertTrue(os.path.exists(output_path))
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == "__main__":
    unittest.main()

