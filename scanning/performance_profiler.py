"""
Performance profiling for 3D scanner.
"""
import time
from collections import defaultdict
import statistics

class PerformanceProfiler:
    """Profile execution time of different scanner components."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
    
    def start(self, operation):
        """Start timing an operation."""
        self.current_timers[operation] = time.perf_counter()
    
    def end(self, operation):
        """End timing and record."""
        if operation in self.current_timers:
            elapsed = (time.perf_counter() - self.current_timers[operation]) * 1000  # ms
            self.timings[operation].append(elapsed)
            del self.current_timers[operation]
            return elapsed
        return 0
    
    def get_stats(self):
        """Get statistics for all operations."""
        stats = {}
        for operation, times in self.timings.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': statistics.mean(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'median_ms': statistics.median(times)
                }
        return stats
    
    def print_report(self):
        """Print performance report."""
        stats = self.get_stats()
        if not stats:
            print("No profiling data collected")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING REPORT")
        print("="*80)
        print(f"{'Operation':<30} {'Count':<8} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-"*80)
        
        # Sort by total time descending
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        
        for operation, data in sorted_stats:
            print(f"{operation:<30} {data['count']:<8} {data['avg_ms']:<12.2f} {data['min_ms']:<12.2f} {data['max_ms']:<12.2f}")
        
        print("-"*80)
        
        # Calculate total time and percentages
        total_time = sum(s['total_ms'] for s in stats.values())
        print(f"\nTotal processing time: {total_time:.2f} ms")
        print("\nTime distribution:")
        for operation, data in sorted_stats[:5]:  # Top 5 operations
            percentage = (data['total_ms'] / total_time) * 100
            print(f"  {operation:<30} {percentage:>6.2f}%")
        print("="*80 + "\n")
    
    def clear(self):
        """Clear all timing data."""
        self.timings.clear()
        self.current_timers.clear()

# Global profiler instance
profiler = PerformanceProfiler()