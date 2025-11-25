"""
HTML dashboard report generator.

This module generates interactive HTML dashboards for visualizing
torch.compile diagnostic information.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

from torch._dynamo.diagnostics.report import CompileReport


def generate_html_report(report: CompileReport, output_path: str) -> None:
    """
    Generate an HTML dashboard report.
    
    Args:
        report: The CompileReport to visualize
        output_path: Path where HTML file should be written
    """
    html_content = _generate_html_content(report)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding="utf-8")


def _generate_html_content(report: CompileReport) -> str:
    """Generate the HTML content for the dashboard."""
    
    # Calculate statistics
    break_counts = report.break_count_by_category
    total_breaks = report.graph_break_count
    duration_str = f"{report.duration:.2f}s" if report.duration else "N/A"
    
    # Generate category breakdown HTML
    category_chart = _generate_category_chart(break_counts)
    
    # Generate graph breaks table
    breaks_table = _generate_breaks_table(report.graph_breaks)
    
    # Generate suggestions
    from torch._dynamo.diagnostics.suggestions import SuggestionEngine
    summary_suggestions = SuggestionEngine.get_summary_suggestions(report)
    suggestions_html = _generate_suggestions_html(report, SuggestionEngine)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>torch.compile Diagnostics - {report.function_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2em;
        }}
        
        .subtitle {{
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-card.warning {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .stat-card.success {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .category-chart {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .category-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}
        
        .category-name {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .category-count {{
            color: #667eea;
            font-size: 1.2em;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        
        .badge-conditional {{ background: #ffeaa7; color: #d63031; }}
        .badge-loop {{ background: #fab1a0; color: #2d3436; }}
        .badge-data-dependent {{ background: #fd79a8; color: white; }}
        .badge-unsupported {{ background: #e17055; color: white; }}
        .badge-external {{ background: #74b9ff; color: white; }}
        .badge-unknown {{ background: #b2bec3; color: #2d3436; }}
        
        .suggestions {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }}
        
        .suggestions h3 {{
            color: #856404;
            margin-bottom: 10px;
        }}
        
        .suggestions ul {{
            margin-left: 20px;
        }}
        
        .suggestions li {{
            margin-bottom: 8px;
            color: #856404;
        }}
        
        .code {{
            font-family: 'Monaco', 'Courier New', monospace;
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 torch.compile Diagnostics</h1>
        <div class="subtitle">
            Function: <span class="code">{report.function_name}</span> | 
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
        
        <div class="stats-grid">
            <div class="stat-card {'warning' if total_breaks > 0 else 'success'}">
                <div class="stat-value">{total_breaks}</div>
                <div class="stat-label">Graph Breaks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.total_compilations}</div>
                <div class="stat-label">Compilation Attempts</div>
            </div>
            <div class="stat-card {'success' if report.successful_compilations == report.total_compilations else 'warning'}">
                <div class="stat-value">{report.successful_compilations}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{duration_str}</div>
                <div class="stat-label">Duration</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Summary</h2>
            <div class="suggestions">
                <h3>Key Insights</h3>
                <ul>
                    {''.join(f'<li>{s}</li>' for s in summary_suggestions)}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Breakdown by Category</h2>
            {category_chart}
        </div>
        
        <div class="section">
            <h2 class="section-title">Graph Breaks Details</h2>
            {breaks_table}
        </div>
        
        <div class="section">
            <h2 class="section-title">Suggestions</h2>
            {suggestions_html}
        </div>
    </div>
</body>
</html>"""
    
    return html


def _generate_category_chart(break_counts: dict) -> str:
    """Generate HTML for category breakdown chart."""
    if not break_counts:
        return "<p>No graph breaks detected.</p>"
    
    items = []
    for category, count in sorted(break_counts.items(), key=lambda x: x[1], reverse=True):
        items.append(f"""
            <div class="category-item">
                <div class="category-name">{category.value.replace('_', ' ').title()}</div>
                <div class="category-count">{count} occurrence(s)</div>
            </div>
        """)
    
    return f'<div class="category-chart">{"".join(items)}</div>'


def _generate_breaks_table(graph_breaks: list) -> str:
    """Generate HTML table for graph breaks."""
    if not graph_breaks:
        return "<p>No graph breaks to display.</p>"
    
    rows = []
    for i, gb in enumerate(graph_breaks, 1):
        badge_class = f"badge-{gb.category.value.split('_')[0]}"
        location = gb.location or "N/A"
        timestamp = gb.timestamp.strftime("%H:%M:%S.%f")[:-3]
        
        rows.append(f"""
            <tr>
                <td>{i}</td>
                <td><span class="badge {badge_class}">{gb.category.value}</span></td>
                <td><span class="code">{gb.reason[:80]}{'...' if len(gb.reason) > 80 else ''}</span></td>
                <td>{location}</td>
                <td class="timestamp">{timestamp}</td>
            </tr>
        """)
    
    return f"""
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Category</th>
                    <th>Reason</th>
                    <th>Location</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    """


def _generate_suggestions_html(report: CompileReport, suggestion_engine) -> str:
    """Generate HTML for suggestions section."""
    from torch._dynamo.diagnostics.suggestions import SuggestionEngine
    
    all_suggestions = []
    
    # Get suggestions for each graph break
    for i, gb in enumerate(report.graph_breaks[:10], 1):  # Limit to first 10
        suggestions = SuggestionEngine.get_suggestions(gb, report)
        if suggestions:
            all_suggestions.append(f"""
                <div style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 6px;">
                    <strong>Break #{i} ({gb.category.value}):</strong>
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        {''.join(f'<li>{s}</li>' for s in suggestions[:3])}
                    </ul>
                </div>
            """)
    
    if not all_suggestions:
        return "<p>No specific suggestions available.</p>"
    
    return "".join(all_suggestions)

