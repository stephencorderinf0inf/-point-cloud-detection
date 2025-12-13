"""
Auto-Discovery System for Runtime Quality Monitor
==================================================
Automatically detects and monitors new functions added to the scanner.
No manual configuration needed - it adapts to code changes!

Features:
- Auto-discovers all functions in scanner module
- Intelligent filtering (skips internal/helper functions)
- Pattern-based function categorization
- Dynamic patching when new functions are added
- Comparison reports showing newly added functions
"""

import inspect
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Set, Callable, Optional
import re


class AutoDiscoveryMonitor:
    """
    Automatically discovers and monitors functions in target modules.
    Adapts to code changes without manual configuration.
    """
    
    def __init__(
        self,
        module_name: str = "laser_3d_scanner_advanced",
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None
    ):
        """
        Initialize auto-discovery monitor.
        
        Args:
            module_name: Name of module to monitor
            exclude_patterns: Regex patterns for functions to skip
            include_patterns: Regex patterns for functions to monitor (overrides excludes)
        """
        self.module_name = module_name
        
        # Default exclusions (internal Python functions)
        self.exclude_patterns = exclude_patterns or [
            r'^_.*',           # Private functions (_function)
            r'^__.*__$',       # Magic methods (__init__)
            r'^test_.*',       # Test functions
        ]
        
        # Default inclusions (all public functions)
        self.include_patterns = include_patterns or [
            r'^[a-z].*',       # Public functions starting with lowercase
            r'^[A-Z].*',       # Class/constructor functions
        ]
        
        self.module = None
        self.discovered_functions = {}
        self.categorized_functions = {
            'startup': [],
            'detection': [],
            'analysis': [],
            'capture': [],
            'processing': [],
            'utility': []
        }
    
    def load_module(self):
        """Load or reload the target module."""
        if self.module_name in sys.modules:
            # Reload if already imported
            self.module = importlib.reload(sys.modules[self.module_name])
            print(f"♻️  Reloaded module: {self.module_name}")
        else:
            # Import fresh
            self.module = importlib.import_module(self.module_name)
            print(f"✓ Loaded module: {self.module_name}")
        
        return self.module
    
    def should_monitor_function(self, func_name: str) -> bool:
        """
        Determine if a function should be monitored.
        
        Args:
            func_name: Function name to check
        
        Returns:
            True if function should be monitored
        """
        # Check include patterns first (whitelist)
        for pattern in self.include_patterns:
            if re.match(pattern, func_name):
                # Now check if it's excluded (blacklist)
                for exclude in self.exclude_patterns:
                    if re.match(exclude, func_name):
                        return False
                return True
        
        return False
    
    def categorize_function(self, func_name: str) -> str:
        """
        Categorize function based on name and purpose.
        
        Args:
            func_name: Function name
        
        Returns:
            Category name
        """
        func_lower = func_name.lower()
        
        # Startup/System category
        if any(keyword in func_lower for keyword in [
            'main', 'check', 'init', 'setup', 'start', 
            'calibration', 'load', 'config'
        ]):
            return 'startup'
        
        # Detection category
        elif any(keyword in func_lower for keyword in [
            'detect', 'find', 'locate', 'identify', 'recognize',
            'laser', 'curve', 'corner', 'ellipse', 'cylinder'
        ]):
            return 'detection'
        
        # Analysis category
        elif any(keyword in func_lower for keyword in [
            'analyze', 'analysis', 'estimate', 'calculate', 'measure',
            'ai', 'quality', 'suggest'
        ]):
            return 'analysis'
        
        # Capture category
        elif any(keyword in func_lower for keyword in [
            'capture', 'grab', 'acquire', 'record', 'snapshot',
            'auto_capture', 'mouse', 'click', 'overlay'
        ]):
            return 'capture'
        
        # Processing category
        elif any(keyword in func_lower for keyword in [
            'process', 'scan', 'save', 'export', 'convert',
            'transform', 'cloud', 'mesh'
        ]):
            return 'processing'
        
        # Default to utility
        else:
            return 'utility'
    
    def discover_functions(self) -> Dict[str, Callable]:
        """
        Automatically discover all monitorable functions in module.
        
        Returns:
            Dictionary of {function_name: function_object}
        """
        if not self.module:
            self.load_module()
        
        discovered = {}
        
        # Get all members of the module
        all_members = inspect.getmembers(self.module)
        
        for name, obj in all_members:
            # Check if it's a function (not a class or variable)
            if inspect.isfunction(obj):
                # Check if we should monitor it
                if self.should_monitor_function(name):
                    discovered[name] = obj
                    
                    # Categorize it
                    category = self.categorize_function(name)
                    if name not in self.categorized_functions[category]:
                        self.categorized_functions[category].append(name)
        
        self.discovered_functions = discovered
        return discovered
    
    def get_newly_added_functions(self, previous_functions: Set[str]) -> Set[str]:
        """
        Compare current functions with previous set to find new additions.
        
        Args:
            previous_functions: Set of previously known function names
        
        Returns:
            Set of newly added function names
        """
        current_functions = set(self.discovered_functions.keys())
        return current_functions - previous_functions
    
    def print_discovery_report(self, highlight_new: Optional[Set[str]] = None):
        """
        Print formatted report of discovered functions.
        
        Args:
            highlight_new: Set of function names to highlight as new
        """
        print("\n" + "="*80)
        print("🔍 AUTO-DISCOVERY REPORT")
        print("="*80)
        
        total_discovered = len(self.discovered_functions)
        print(f"\n✓ Discovered {total_discovered} monitorable functions")
        
        # Print by category
        for category, functions in self.categorized_functions.items():
            if functions:
                print(f"\n📁 {category.upper()} ({len(functions)} functions):")
                for func_name in sorted(functions):
                    # Highlight new functions
                    if highlight_new and func_name in highlight_new:
                        print(f"   ⭐ NEW: {func_name}")
                    else:
                        print(f"   ✓ {func_name}")
        
        print("\n" + "="*80)
    
    def patch_all_discovered(self, monitor):
        """
        Patch all discovered functions with the monitor.
        
        Args:
            monitor: QualityMonitor instance to use for tracking
        
        Returns:
            Number of functions patched
        """
        patched_count = 0
        
        for func_name, func_obj in self.discovered_functions.items():
            try:
                # Wrap with monitor
                monitored_func = monitor.track_function(func_obj)
                
                # Set back on module
                setattr(self.module, func_name, monitored_func)
                
                patched_count += 1
            except Exception as e:
                print(f"⚠️  Failed to patch {func_name}: {e}")
        
        return patched_count
    
    def save_function_snapshot(self, filepath: str = "scanning/analysis/function_snapshot.txt"):
        """
        Save current function list for future comparison.
        
        Args:
            filepath: Path to save snapshot
        """
        snapshot_path = Path(filepath)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(snapshot_path, 'w') as f:
            f.write("# Function Snapshot\n")
            f.write(f"# Module: {self.module_name}\n")
            f.write(f"# Total Functions: {len(self.discovered_functions)}\n\n")
            
            for category, functions in self.categorized_functions.items():
                if functions:
                    f.write(f"\n[{category.upper()}]\n")
                    for func_name in sorted(functions):
                        f.write(f"{func_name}\n")
        
        print(f"✓ Function snapshot saved to {snapshot_path}")
    
    def load_previous_snapshot(self, filepath: str = "scanning/analysis/function_snapshot.txt") -> Set[str]:
        """
        Load previous function snapshot for comparison.
        
        Args:
            filepath: Path to snapshot file
        
        Returns:
            Set of function names from previous snapshot
        """
        snapshot_path = Path(filepath)
        
        if not snapshot_path.exists():
            return set()
        
        previous_functions = set()
        
        with open(snapshot_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and section headers
                if line and not line.startswith('#') and not line.startswith('['):
                    previous_functions.add(line)
        
        return previous_functions


def create_auto_discovery_monitor(
    module_name: str = "laser_3d_scanner_advanced",
    show_report: bool = True
):
    """
    Create an auto-discovery monitor that finds all functions automatically.
    
    Args:
        module_name: Module to monitor
        show_report: Print discovery report
    
    Returns:
        AutoDiscoveryMonitor instance
    """
    print("\n" + "="*80)
    print("🤖 INITIALIZING AUTO-DISCOVERY MONITOR")
    print("="*80)
    
    # Create discovery system
    auto_monitor = AutoDiscoveryMonitor(module_name)
    
    # Load previous snapshot if exists
    previous_functions = auto_monitor.load_previous_snapshot()
    
    # Discover current functions
    discovered = auto_monitor.discover_functions()
    
    # Find newly added functions
    new_functions = auto_monitor.get_newly_added_functions(previous_functions)
    
    if new_functions:
        print(f"\n⭐ DETECTED {len(new_functions)} NEW FUNCTIONS!")
        for func in sorted(new_functions):
            print(f"   • {func}")
    
    # Show report
    if show_report:
        auto_monitor.print_discovery_report(highlight_new=new_functions)
    
    # Save current snapshot for next time
    auto_monitor.save_function_snapshot()
    
    return auto_monitor


# Example usage
if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*80)
    print("EXAMPLE: Auto-Discovery in Action")
    print("="*80)
    
    # Create auto-discovery monitor
    auto = create_auto_discovery_monitor(
        module_name="laser_3d_scanner_advanced",
        show_report=True
    )
    
    print(f"\n✓ Ready to monitor {len(auto.discovered_functions)} functions automatically!")
    print("\nNext time you run this:")
    print("  1. Any new functions will be automatically detected")
    print("  2. They'll be highlighted in the report")
    print("  3. They'll be automatically monitored")
    print("\nNo manual configuration needed! 🎉")
