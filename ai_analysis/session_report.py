"""
Automatic session report generator for AI analysis data.
"""
import json
import csv
from pathlib import Path
from datetime import datetime

def generate_session_report(session_dir):
    """
    Generate a comprehensive report for a completed session.
    
    Args:
        session_dir: Path to session directory
    """
    try:
        session_path = Path(session_dir)
        csv_file = session_path / "ai_results.csv"
        json_file = session_path / "session_summary.json"
        report_file = session_path / "session_report.html"
        
        if not csv_file.exists():
            print(f"Warning: No data found in {session_dir}")
            return None
        
        # Load data
        with open(json_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        if not data:
            print(f"Warning: No data rows found")
            return None
        
        # Calculate duration
        timestamps = [datetime.fromisoformat(row['timestamp']) for row in data]
        duration = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0
        
        # FIX: Use the 'overall_score' from statistics instead of recalculating
        if 'overall_score' in summary['statistics']:
            quality_score = summary['statistics']['overall_score']['avg'] * 100
        else:
            # Fallback: Calculate from individual metrics
            quality_components = []
            if 'sharpness' in summary['statistics']:
                quality_components.append(summary['statistics']['sharpness']['avg'] * 0.4)
            if 'brightness' in summary['statistics']:
                quality_components.append(summary['statistics']['brightness']['avg'] * 0.3)
            if 'contrast' in summary['statistics']:
                quality_components.append(summary['statistics']['contrast']['avg'] * 0.3)
            
            quality_score = (sum(quality_components) / len(quality_components) * 100) if quality_components else 0
        
        # Build statistics table rows
        stats_rows = ""
        for metric, values in summary['statistics'].items():
            metric_range = values['max'] - values['min']
            stats_rows += f"""
            <tr>
                <td><strong>{metric.replace('_', ' ').title()}</strong></td>
                <td>{values['avg']:.2f}</td>
                <td>{values['min']:.2f}</td>
                <td>{values['max']:.2f}</td>
                <td>{metric_range:.2f}</td>
            </tr>
            """
        
        # Build key metrics
        key_metrics = ""
        for metric, values in summary['statistics'].items():
            key_metrics += f"""
            <div class="metric">
                <div class="metric-value">{values['avg']:.2f}</div>
                <div class="metric-label">{metric.replace('_', ' ').title()}</div>
            </div>
            """
        
        # Build recommendations
        recommendations = []
        if 'sharpness' in summary['statistics']:
            if summary['statistics']['sharpness']['avg'] > 0.5:
                recommendations.append('<li style="color: #27ae60;">[OK] Excellent image sharpness - good for detailed scans</li>')
            elif summary['statistics']['sharpness']['avg'] > 0.3:
                recommendations.append('<li style="color: #f39c12;">[~] Fair image sharpness - acceptable for most scanning</li>')
            else:
                recommendations.append('<li style="color: #e67e22;">[!] Low sharpness - consider better lighting or focus</li>')
        
        if 'brightness' in summary['statistics']:
            if summary['statistics']['brightness']['avg'] > 0.5:
                recommendations.append('<li style="color: #27ae60;">[OK] Good brightness levels - well-lit scene</li>')
            elif summary['statistics']['brightness']['avg'] > 0.3:
                recommendations.append('<li style="color: #f39c12;">[~] Moderate brightness - consider adding more light</li>')
            else:
                recommendations.append('<li style="color: #e67e22;">[!] Low brightness - increase lighting</li>')
        
        if 'fps' in summary['statistics']:
            fps_range = summary['statistics']['fps']['max'] - summary['statistics']['fps']['min']
            if fps_range < 3:
                recommendations.append('<li style="color: #27ae60;">[OK] Stable frame rate - smooth scanning</li>')
            else:
                recommendations.append('<li style="color: #e67e22;">[!] Unstable FPS - system may be overloaded</li>')
        
        if 'contrast' in summary['statistics']:
            if summary['statistics']['contrast']['avg'] > 0.5:
                recommendations.append('<li style="color: #27ae60;">[OK] Good contrast - clear detail separation</li>')
            elif summary['statistics']['contrast']['avg'] > 0.3:
                recommendations.append('<li style="color: #f39c12;">[~] Moderate contrast - acceptable image detail</li>')
            else:
                recommendations.append('<li style="color: #e67e22;">[!] Low contrast - flat image, adjust lighting</li>')
        
        recommendations_html = '\n'.join(recommendations) if recommendations else '<li>No specific recommendations</li>'
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Session Report - {summary['session_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ display: inline-block; margin: 15px 20px; padding: 15px 25px; background: #ecf0f1; border-radius: 8px; }}
                .metric-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
                .quality-excellent {{ color: #27ae60; }}
                .quality-good {{ color: #f39c12; }}
                .quality-fair {{ color: #e67e22; }}
                .quality-poor {{ color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #3498db; color: white; }}
                tr:hover {{ background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Analysis Session Report</h1>
                <p><strong>Session:</strong> {summary['session_name']}</p>
                <p><strong>Date:</strong> {datetime.fromisoformat(summary['session_start']).strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Duration:</strong> {duration:.1f} seconds ({duration/60:.1f} minutes)</p>
                
                <h2>Overall Quality Score</h2>
                <div class="metric">
                    <div class="metric-value {'quality-excellent' if quality_score >= 70 else 'quality-good' if quality_score >= 50 else 'quality-fair' if quality_score >= 30 else 'quality-poor'}">{quality_score:.1f}%</div>
                    <div class="metric-label">Quality Score</div>
                </div>
                
                <h2>Key Metrics</h2>
                <div style="margin: 20px 0;">
                    <div class="metric">
                        <div class="metric-value">{summary['total_frames']}</div>
                        <div class="metric-label">Total Frames</div>
                    </div>
                    {key_metrics}
                </div>
                
                <h2>Statistics Table</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Average</th>
                        <th>Minimum</th>
                        <th>Maximum</th>
                        <th>Range</th>
                    </tr>
                    {stats_rows}
                </table>
                
                <h2>Recommendations</h2>
                <ul>
                    {recommendations_html}
                </ul>
                
                <h2>Data Files</h2>
                <ul>
                    <li><strong>Raw Data:</strong> ai_results.csv</li>
                    <li><strong>Summary:</strong> session_summary.json</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Write with UTF-8 encoding
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[OK] Report saved: {report_file}")
        print(f"   Quality Score: {quality_score:.1f}%")
        print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
        
        return report_file
    
    except Exception as e:
        print(f"   ⚠️ Could not generate report: {e}")
        return None