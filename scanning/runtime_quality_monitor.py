"""
Runtime Quality Monitor for Advanced 3D Scanner
================================================
AI-powered monitoring system that tracks function execution, quality metrics,
and implementation issues in laser_3d_scanner_advanced.py and associated scripts.

Features:
- Function execution tracking with timing
- Error detection and logging
- Quality metrics monitoring (frame processing, detection accuracy)
- Resource usage tracking (memory, CPU)
- Automated issue reporting
- Real-time alerts for anomalies

Usage:
    # Import in your main script
    from runtime_quality_monitor import QualityMonitor
    
    # Initialize monitor
    monitor = QualityMonitor(
        log_file="scanner_quality_log.json",
        enable_alerts=True,
        performance_threshold_ms=100
    )
    
    # Wrap functions for monitoring
    @monitor.track_function
    def your_function():
        pass
    
    # Or manually track
    with monitor.track("operation_name"):
        # Your code here
        pass
    
    # Generate report
    monitor.generate_report()
"""

import functools
import time
import traceback
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable


class QualityMonitor:
    """
    Comprehensive quality monitoring system for 3D scanner functions.
    Tracks execution, detects issues, and provides analytics.
    """
    
    def __init__(
        self,
        log_file: str = "scanner_quality_monitor.json",
        enable_alerts: bool = True,
        performance_threshold_ms: float = 100,
        memory_threshold_mb: float = 500,
        auto_save_interval: int = 50
    ):
        """
        Initialize Quality Monitor.
        
        Args:
            log_file: Path to save monitoring data
            enable_alerts: Show real-time alerts for issues
            performance_threshold_ms: Alert if function exceeds this time (ms)
            memory_threshold_mb: Alert if memory exceeds this threshold (MB)
            auto_save_interval: Auto-save logs every N operations
        """
        self.log_file = Path(log_file)
        self.enable_alerts = enable_alerts
        self.performance_threshold = performance_threshold_ms / 1000  # Convert to seconds
        self.memory_threshold = memory_threshold_mb * 1024 * 1024  # Convert to bytes
        self.auto_save_interval = auto_save_interval
        
        # Monitoring data
        self.session_start = datetime.now()
        self.function_stats = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0,
            'avg_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'errors': [],
            'warnings': []
        })
        
        self.execution_log = []
        self.error_log = []
        self.warning_log = []
        self.performance_issues = []
        self.memory_snapshots = []
        
        # Operation counter for auto-save
        self.operation_count = 0
        
        # System info
        self.process = psutil.Process()
        
        print(f"✓ Quality Monitor initialized")
        print(f"  Log file: {self.log_file}")
        print(f"  Performance threshold: {performance_threshold_ms}ms")
        print(f"  Memory threshold: {memory_threshold_mb}MB")
    
    def track_function(self, func: Callable) -> Callable:
        """
        Decorator to track function execution.
        
        Usage:
            @monitor.track_function
            def my_function(arg1, arg2):
                # function code
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Record memory before
            mem_before = self.process.memory_info().rss
            
            # Track execution
            start_time = time.perf_counter()
            error_occurred = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            
            except Exception as e:
                error_occurred = e
                self._log_error(func_name, e, traceback.format_exc())
                raise  # Re-raise the error
            
            finally:
                # Calculate metrics
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                mem_after = self.process.memory_info().rss
                mem_delta = mem_after - mem_before
                
                # Update statistics
                self._update_stats(func_name, elapsed, error_occurred)
                
                # Log execution
                self._log_execution(
                    func_name=func_name,
                    elapsed_time=elapsed,
                    memory_delta=mem_delta,
                    success=error_occurred is None,
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                
                # Check for performance issues
                if elapsed > self.performance_threshold:
                    self._alert_slow_function(func_name, elapsed)
                
                # Check memory usage
                if mem_after > self.memory_threshold:
                    self._alert_high_memory(func_name, mem_after)
                
                # Auto-save periodically
                self.operation_count += 1
                if self.operation_count % self.auto_save_interval == 0:
                    self.save_logs()
        
        return wrapper
    
    def track(self, operation_name: str):
        """
        Context manager for tracking code blocks.
        
        Usage:
            with monitor.track("image_processing"):
                # process image
        """
        return _TrackingContext(self, operation_name)
    
    def _update_stats(self, func_name: str, elapsed: float, error: Optional[Exception]):
        """Update function statistics."""
        stats = self.function_stats[func_name]
        stats['call_count'] += 1
        stats['total_time'] += elapsed
        stats['avg_time'] = stats['total_time'] / stats['call_count']
        stats['min_time'] = min(stats['min_time'], elapsed)
        stats['max_time'] = max(stats['max_time'], elapsed)
        
        if error:
            stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'message': str(error)
            })
    
    def _log_execution(self, func_name: str, elapsed_time: float, 
                       memory_delta: int, success: bool, 
                       args_count: int, kwargs_count: int):
        """Log function execution details."""
        self.execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'function': func_name,
            'elapsed_ms': elapsed_time * 1000,
            'memory_delta_mb': memory_delta / (1024 * 1024),
            'success': success,
            'args_count': args_count,
            'kwargs_count': kwargs_count
        })
    
    def _log_error(self, func_name: str, error: Exception, trace: str):
        """Log error details."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'function': func_name,
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': trace
        }
        self.error_log.append(error_entry)
        
        if self.enable_alerts:
            print(f"\n⚠️  ERROR in {func_name}")
            print(f"   {type(error).__name__}: {error}")
    
    def log_warning(self, func_name: str, message: str):
        """Manually log a warning."""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'function': func_name,
            'message': message
        }
        self.warning_log.append(warning_entry)
        self.function_stats[func_name]['warnings'].append(warning_entry)
        
        if self.enable_alerts:
            print(f"\n⚠️  WARNING in {func_name}: {message}")
    
    def _alert_slow_function(self, func_name: str, elapsed: float):
        """Alert for slow function execution."""
        issue = {
            'timestamp': datetime.now().isoformat(),
            'function': func_name,
            'elapsed_ms': elapsed * 1000,
            'threshold_ms': self.performance_threshold * 1000
        }
        self.performance_issues.append(issue)
        
        if self.enable_alerts:
            print(f"\n🐌 SLOW: {func_name} took {elapsed*1000:.1f}ms (threshold: {self.performance_threshold*1000:.1f}ms)")
    
    def _alert_high_memory(self, func_name: str, memory_bytes: int):
        """Alert for high memory usage."""
        memory_mb = memory_bytes / (1024 * 1024)
        threshold_mb = self.memory_threshold / (1024 * 1024)
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'function': func_name,
            'memory_mb': memory_mb,
            'threshold_mb': threshold_mb
        }
        self.memory_snapshots.append(snapshot)
        
        if self.enable_alerts:
            print(f"\n💾 HIGH MEMORY: {func_name} using {memory_mb:.1f}MB (threshold: {threshold_mb:.1f}MB)")
    
    def record_metric(self, metric_name: str, value: float, context: str = ""):
        """
        Record a custom quality metric.
        
        Args:
            metric_name: Name of the metric (e.g., "detection_accuracy")
            value: Metric value
            context: Optional context information
        """
        if not hasattr(self, 'custom_metrics'):
            self.custom_metrics = defaultdict(list)
        
        self.custom_metrics[metric_name].append({
            'timestamp': datetime.now().isoformat(),
            'value': value,
            'context': context
        })
    
    def save_logs(self, filename: Optional[str] = None):
        """Save monitoring data to JSON file."""
        save_path = Path(filename) if filename else self.log_file
        
        # Calculate session duration
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        # Prepare data
        data = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'duration_seconds': session_duration,
                'total_operations': self.operation_count
            },
            'function_statistics': dict(self.function_stats),
            'execution_log': self.execution_log[-100:],  # Last 100 executions
            'error_log': self.error_log,
            'warning_log': self.warning_log,
            'performance_issues': self.performance_issues,
            'memory_snapshots': self.memory_snapshots[-50:],  # Last 50 snapshots
        }
        
        # Add custom metrics if any
        if hasattr(self, 'custom_metrics'):
            data['custom_metrics'] = dict(self.custom_metrics)
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Quality logs saved to {save_path}")
    
    def generate_report(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            verbose: Print detailed report to console
        
        Returns:
            Dictionary containing report data
        """
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        # Calculate summary statistics
        total_calls = sum(stat['call_count'] for stat in self.function_stats.values())
        total_errors = len(self.error_log)
        total_warnings = len(self.warning_log)
        total_perf_issues = len(self.performance_issues)
        
        # Find slowest functions
        slowest_funcs = sorted(
            [(name, stats['max_time']) for name, stats in self.function_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Find most called functions
        most_called = sorted(
            [(name, stats['call_count']) for name, stats in self.function_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        report = {
            'session_duration_seconds': session_duration,
            'total_function_calls': total_calls,
            'unique_functions_called': len(self.function_stats),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'performance_issues': total_perf_issues,
            'slowest_functions': slowest_funcs,
            'most_called_functions': most_called
        }
        
        if verbose:
            self._print_report(report)
        
        return report
    
    def _print_report(self, report: Dict[str, Any]):
        """Print formatted quality report."""
        print("\n" + "="*80)
        print("📊 QUALITY MONITOR REPORT")
        print("="*80)
        
        print(f"\n⏱️  Session Duration: {report['session_duration_seconds']:.1f}s")
        print(f"📞 Total Function Calls: {report['total_function_calls']}")
        print(f"🔧 Unique Functions: {report['unique_functions_called']}")
        
        print(f"\n❌ Errors: {report['total_errors']}")
        print(f"⚠️  Warnings: {report['total_warnings']}")
        print(f"🐌 Performance Issues: {report['performance_issues']}")
        
        if report['slowest_functions']:
            print(f"\n🐌 Slowest Functions:")
            for i, (func, time_ms) in enumerate(report['slowest_functions'], 1):
                print(f"   {i}. {func}: {time_ms*1000:.2f}ms")
        
        if report['most_called_functions']:
            print(f"\n📞 Most Called Functions:")
            for i, (func, count) in enumerate(report['most_called_functions'], 1):
                print(f"   {i}. {func}: {count} calls")
        
        if self.error_log:
            print(f"\n❌ Recent Errors:")
            for error in self.error_log[-5:]:
                print(f"   [{error['timestamp']}] {error['function']}")
                print(f"      {error['error_type']}: {error['message']}")
        
        print("\n" + "="*80)
    
    def get_function_health(self, func_name: str) -> Dict[str, Any]:
        """
        Get health metrics for a specific function.
        
        Returns:
            Dictionary with function health information
        """
        if func_name not in self.function_stats:
            return {'status': 'not_monitored'}
        
        stats = self.function_stats[func_name]
        
        # Calculate health score (0-100)
        health_score = 100
        if stats['errors']:
            health_score -= len(stats['errors']) * 10
        if stats['warnings']:
            health_score -= len(stats['warnings']) * 5
        if stats['max_time'] > self.performance_threshold:
            health_score -= 20
        
        health_score = max(0, health_score)
        
        return {
            'function': func_name,
            'health_score': health_score,
            'call_count': stats['call_count'],
            'avg_time_ms': stats['avg_time'] * 1000,
            'error_count': len(stats['errors']),
            'warning_count': len(stats['warnings']),
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'critical'
        }


class _TrackingContext:
    """Context manager for tracking code blocks."""
    
    def __init__(self, monitor: QualityMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.mem_before = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.mem_before = self.monitor.process.memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        mem_after = self.monitor.process.memory_info().rss
        mem_delta = mem_after - self.mem_before
        
        # Update stats
        self.monitor._update_stats(self.operation_name, elapsed, exc_val)
        
        # Log execution
        self.monitor._log_execution(
            func_name=self.operation_name,
            elapsed_time=elapsed,
            memory_delta=mem_delta,
            success=exc_val is None,
            args_count=0,
            kwargs_count=0
        )
        
        # Check thresholds
        if elapsed > self.monitor.performance_threshold:
            self.monitor._alert_slow_function(self.operation_name, elapsed)
        
        if mem_after > self.monitor.memory_threshold:
            self.monitor._alert_high_memory(self.operation_name, mem_after)
        
        return False  # Don't suppress exceptions


def create_scanner_monitor(log_dir: str = "scanning/analysis") -> QualityMonitor:
    """
    Create a pre-configured monitor for the 3D scanner.
    
    Args:
        log_dir: Directory to store logs
    
    Returns:
        Configured QualityMonitor instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"scanner_monitor_{timestamp}.json"
    
    monitor = QualityMonitor(
        log_file=str(log_file),
        enable_alerts=True,
        performance_threshold_ms=100,  # Alert if function > 100ms
        memory_threshold_mb=500,       # Alert if process > 500MB
        auto_save_interval=25          # Save every 25 operations
    )
    
    return monitor


# Example usage and integration guide
if __name__ == "__main__":
    print(__doc__)
    
    # Example 1: Basic usage
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Function Tracking")
    print("="*80)
    
    monitor = QualityMonitor()
    
    @monitor.track_function
    def example_function(x, y):
        """Example function with tracking."""
        time.sleep(0.05)  # Simulate work
        return x + y
    
    # Call function multiple times
    for i in range(5):
        result = example_function(i, i*2)
    
    # Generate report
    monitor.generate_report()
    
    # Example 2: Context manager
    print("\n" + "="*80)
    print("EXAMPLE 2: Context Manager Tracking")
    print("="*80)
    
    with monitor.track("image_processing"):
        time.sleep(0.03)
        # Process image here
    
    # Example 3: Custom metrics
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Metrics")
    print("="*80)
    
    monitor.record_metric("detection_accuracy", 0.95, "laser_dot_detection")
    monitor.record_metric("detection_accuracy", 0.92, "laser_dot_detection")
    
    # Save logs
    monitor.save_logs()
    
    print("\n✓ Examples completed! Check scanner_quality_monitor.json")
