"""
Manual analysis tool for comparing multiple AI sessions.
Run this script whenever you want to analyze your session history.

Usage:
    python analyze_sessions.py
"""
import json
from pathlib import Path
from datetime import datetime

RESULTS_ROOT = Path("D:/ai_analysis_results")

def list_sessions():
    """List all available sessions."""
    sessions = sorted(RESULTS_ROOT.glob("session_*"))
    
    if not sessions:
        print("\n⚠️  No sessions found!")
        print(f"   Run your 3D scanner first to generate data in: {RESULTS_ROOT}")
        return []
    
    print("\n" + "="*80)
    print("📊 Available AI Analysis Sessions")
    print("="*80)
    
    for i, session in enumerate(sessions, 1):
        json_file = session / "session_summary.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            start_time = datetime.fromisoformat(data['session_start'])
            
            print(f"\n{i}. {session.name}")
            print(f"   📅 Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   🎬 Frames: {data['total_frames']}")
            
            # Show available metrics
            if 'statistics' in data:
                metrics = ", ".join(data['statistics'].keys())
                print(f"   📈 Metrics: {metrics}")
    
    print("\n" + "="*80)
    return sessions

def show_session_details(session_dir):
    """Show detailed information for a session."""
    json_file = session_dir / "session_summary.json"
    
    if not json_file.exists():
        print(f"⚠️  Summary not found for {session_dir.name}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print(f"📊 Session Details: {data['session_name']}")
    print("="*80)
    print(f"Started: {datetime.fromisoformat(data['session_start']).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Frames: {data['total_frames']}")
    
    print("\n📈 Statistics:")
    for metric, values in data['statistics'].items():
        print(f"\n  {metric.replace('_', ' ').title()}:")
        print(f"    Average: {values['avg']:.3f}")
        print(f"    Min: {values['min']:.3f}")
        print(f"    Max: {values['max']:.3f}")
        print(f"    Range: {values['max'] - values['min']:.3f}")

def compare_sessions(session_dirs):
    """Compare multiple sessions."""
    print("\n" + "="*80)
    print("📊 Session Comparison")
    print("="*80)
    
    # Load all session data
    sessions_data = []
    for session_dir in session_dirs:
        json_file = session_dir / "session_summary.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                sessions_data.append(data)
    
    if not sessions_data:
        print("⚠️  No valid sessions to compare")
        return
    
    # Find common metrics
    all_metrics = set()
    for data in sessions_data:
        all_metrics.update(data['statistics'].keys())
    
    # Compare each metric
    for metric in sorted(all_metrics):
        print(f"\n📈 {metric.replace('_', ' ').title()}:")
        print(f"   {'Session':<30} {'Avg':>10} {'Min':>10} {'Max':>10}")
        print(f"   {'-'*60}")
        
        for data in sessions_data:
            if metric in data['statistics']:
                values = data['statistics'][metric]
                session_name = data['session_name'][:28]
                print(f"   {session_name:<30} {values['avg']:>10.3f} {values['min']:>10.3f} {values['max']:>10.3f}")

if __name__ == "__main__":
    sessions = list_sessions()
    
    if not sessions:
        exit()
    
    print("\n🔍 Options:")
    print("  1. View session details (enter session number)")
    print("  2. Compare sessions (enter numbers separated by commas)")
    print("  3. Exit")
    
    choice = input("\n> ").strip()
    
    if choice == "3":
        print("Goodbye!")
        exit()
    
    # Parse input
    if ',' in choice:
        # Compare mode
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected = [sessions[i] for i in indices if 0 <= i < len(sessions)]
            if selected:
                compare_sessions(selected)
            else:
                print("❌ Invalid selection!")
        except ValueError:
            print("❌ Invalid input! Use comma-separated numbers.")
    else:
        # Details mode
        try:
            index = int(choice) - 1
            if 0 <= index < len(sessions):
                show_session_details(sessions[index])
            else:
                print("❌ Invalid session number!")
        except ValueError:
            print("❌ Invalid input! Enter a number.")