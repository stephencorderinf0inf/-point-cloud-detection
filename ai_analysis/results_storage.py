"""
Results storage and session management.
"""
import csv
import json
from datetime import datetime
from pathlib import Path

# Import without relative import
import session_report

# Storage location for AI results
RESULTS_ROOT = Path("D:/ai_analysis_results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

class AIResultsStorage:
    """Store AI analysis results for later review."""
    
    def __init__(self, session_name=None):
        """Initialize storage for a session."""
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.session_dir = RESULTS_ROOT / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CSV file for results
        self.csv_file = self.session_dir / "ai_results.csv"
        self.json_file = self.session_dir / "session_summary.json"
        
        # Session metadata
        self.session_start = datetime.now()
        self.frame_count = 0
        self.results_buffer = []
        self.all_keys = set()  # Track all keys for flexible CSV
        self.csv_initialized = False
    
    def _init_csv(self, headers):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        self.csv_initialized = True
    
    def save_result(self, ai_result, frame_number):
        """
        Save a single AI analysis result.
        
        Args:
            ai_result: Dictionary of AI analysis results
            frame_number: Current frame number
        """
        self.frame_count += 1
        
        # Prepare row data with timestamp and frame number
        row = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": frame_number,
            **ai_result  # Unpack all AI results
        }
        
        # Track all keys for flexible CSV
        self.all_keys.update(row.keys())
        
        # Initialize CSV on first write
        if not self.csv_initialized:
            headers = ["timestamp", "frame_number"] + sorted([k for k in self.all_keys if k not in ["timestamp", "frame_number"]])
            self._init_csv(headers)
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            # Use all tracked keys as fieldnames
            headers = ["timestamp", "frame_number"] + sorted([k for k in self.all_keys if k not in ["timestamp", "frame_number"]])
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writerow(row)
        
        # Add to buffer for summary
        self.results_buffer.append(row)
        
        # Save summary every 100 frames
        if len(self.results_buffer) >= 100:
            self._save_summary()
            self.results_buffer = []
    
    def _save_summary(self):
        """Save session summary statistics."""
        if not self.results_buffer:
            return
        
        # Dynamically calculate statistics for all numeric fields
        statistics = {}
        numeric_keys = [k for k in self.all_keys if k not in ["timestamp", "frame_number", "resolution", "status", "focal_length"]]
        
        for key in numeric_keys:
            try:
                values = [float(r.get(key, 0)) for r in self.results_buffer if key in r]
                if values:
                    statistics[key] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
            except (ValueError, TypeError):
                # Skip non-numeric fields
                pass
        
        summary = {
            "session_name": self.session_name,
            "session_start": self.session_start.isoformat(),
            "last_update": datetime.now().isoformat(),
            "total_frames": self.frame_count,
            "statistics": statistics
        }
        
        # Save summary
        with open(self.json_file, 'w') as f:
            json.dump(summary, f, indent=4)
    
    def finalize(self):
        """Finalize session and save final summary."""
        self._save_summary()
        print(f"✅ Session saved: {self.session_dir}")
        print(f"   Total frames analyzed: {self.frame_count}")
        print(f"   Results: {self.csv_file}")
        print(f"   Summary: {self.json_file}")
        
        # Auto-generate report
        try:
            report_file = session_report.generate_session_report(self.session_dir)
            print(f"   Report: {report_file}")
        except Exception as e:
            print(f"   ⚠️ Could not generate report: {e}")

# Global storage instance (created per session)
_current_storage = None

def get_storage(session_name=None):
    """Get or create storage instance."""
    global _current_storage
    if _current_storage is None:
        _current_storage = AIResultsStorage(session_name)
    return _current_storage

def save_analysis_result(ai_result, frame_number):
    """Save AI analysis result (convenience function)."""
    storage = get_storage()
    storage.save_result(ai_result, frame_number)

def finalize_session():
    """Finalize current session."""
    global _current_storage
    if _current_storage:
        _current_storage.finalize()
        _current_storage = None