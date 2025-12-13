"""
Example: Running Advanced 3D Scanner with Quality Monitoring
=============================================================
This script demonstrates how to integrate the Runtime Quality Monitor
with laser_3d_scanner_advanced.py for comprehensive quality tracking.

The monitor will:
- Track all function executions and timing
- Detect performance bottlenecks
- Log errors and warnings
- Monitor memory usage
- Generate detailed quality reports

Usage:
    python scanner_with_monitoring.py
    
    # Or import and use programmatically
    from scanner_with_monitoring import run_monitored_scan
    run_monitored_scan()
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from runtime_quality_monitor import create_scanner_monitor
from auto_discovery_monitor import create_auto_discovery_monitor
from enhanced_error_detection import setup_scanner_validation
import laser_3d_scanner_advanced as scanner_module


def patch_scanner_functions(monitor, use_auto_discovery=True):
    """
    Patch key functions in the scanner module with monitoring.
    
    This wraps important functions to track their performance and quality.
    ALL-INCLUSIVE: Monitors from startup (system checks) through completion.
    
    Args:
        monitor: QualityMonitor instance
        use_auto_discovery: If True, automatically discover and monitor ALL functions.
                           If False, use manual list (for legacy compatibility)
    
    Returns:
        Number of functions patched
    """
    if use_auto_discovery:
        print("\n🤖 AUTO-DISCOVERY MODE: Automatically detecting functions...")
        
        # Use auto-discovery to find all functions
        auto_monitor = create_auto_discovery_monitor(
            module_name="laser_3d_scanner_advanced",
            show_report=True
        )
        
        # Patch all discovered functions
        print("\n🔧 Patching all discovered functions...")
        patched_count = auto_monitor.patch_all_discovered(monitor)
        
        print(f"\n✓ Auto-patched {patched_count} functions")
        print("   New functions will be automatically detected next time!")
        
        return patched_count
    
    else:
        print("\n🔧 MANUAL MODE: Patching predefined functions...")
        
        # Manual list of functions to monitor (legacy mode)
        functions_to_monitor = [
            # === STARTUP FUNCTIONS ===
            'main',                              # Main entry point
            'check_system_requirements',         # System requirement checks
            'find_calibration_file',            # Calibration file detection
            
            # === DETECTION FUNCTIONS ===
            'detect_red_laser_dot',             # Core laser detection
            'detect_curves',                    # Curve detection
            'detect_corners',                   # Corner detection
            'detect_ellipses',                  # Ellipse detection
            'detect_cylinders',                 # Cylinder detection
            'detect_laser_with_spectrum',       # Spectrum-based detection
            
            # === ANALYSIS FUNCTIONS ===
            'suggest_roi_from_contrast',        # ROI suggestion
            'estimate_distance_linear',         # Distance estimation
            'run_ai_analysis',                  # AI analysis
            
            # === CAPTURE FUNCTIONS ===
            'auto_capture_3_points',            # Standard auto-capture
            'auto_capture_3_points_with_module',# Module-based auto-capture
            'mouse_callback',                   # Mouse interaction
            'show_capture_overlay',             # Capture UI overlay
            'apply_cartoon_settings',           # Camera settings
            
            # === DATA PROCESSING ===
            'save_point_cloud',                 # Point cloud saving
            'scan_3d_points',                   # Main scanning loop
        ]
        
        patched_count = 0
        for func_name in functions_to_monitor:
            if hasattr(scanner_module, func_name):
                original_func = getattr(scanner_module, func_name)
                monitored_func = monitor.track_function(original_func)
                setattr(scanner_module, func_name, monitored_func)
                patched_count += 1
                print(f"  ✓ Patched: {func_name}")
            else:
                print(f"  ⚠ Not found: {func_name}")
        
        print(f"\n✓ Patched {patched_count}/{len(functions_to_monitor)} functions")
        return patched_count


def run_monitored_scan(project_dir=None, generate_report=True, run_system_check=True, 
                       auto_discovery=True, validate_results=True):
    """
    Run 3D scanner with quality monitoring enabled.
    ALL-INCLUSIVE: Monitors from system checks through final save.
    AUTO-ADAPTIVE: Automatically detects new functions when code changes.
    ERROR-AWARE: Detects exceptions AND silent failures.
    
    Args:
        project_dir: Optional project directory for scanning
        generate_report: Generate quality report after scanning
        run_system_check: Run system requirement checks (monitored)
        auto_discovery: Automatically discover and monitor all functions
        validate_results: Enable enhanced error detection (catches silent failures)
    
    Returns:
        Tuple of (scan_result, quality_monitor)
    """
    print("\n" + "="*80)
    print("🚀 STARTING MONITORED 3D SCAN")
    print("   (ALL-INCLUSIVE + AUTO-ADAPTIVE + ERROR-AWARE)")
    print("="*80)
    
    # Create monitor BEFORE any operations
    monitor = create_scanner_monitor()
    
    # Setup enhanced error detection if requested
    validator = None
    if validate_results:
        validator = setup_scanner_validation(monitor)
    
    # Patch scanner functions EARLY (before system checks)
    # Use auto-discovery by default to automatically detect new functions
    patched = patch_scanner_functions(monitor, use_auto_discovery=auto_discovery)
    
    if patched == 0:
        print("\n⚠️  WARNING: No functions were patched!")
        print("   The scanner will run normally but without monitoring.")
    
    # Add custom metrics tracking for scanner-specific quality
    print("\n📊 Quality monitoring enabled:")
    print(f"  - Startup monitoring: ✓ (system checks tracked)")
    print(f"  - Auto-discovery: {'✓ (adapts to code changes)' if auto_discovery else '✗ (manual mode)'}")
    print(f"  - Enhanced error detection: {'✓ (catches silent failures)' if validate_results else '✗'}")
    print(f"  - Performance tracking: ✓")
    print(f"  - Memory monitoring: ✓")
    print(f"  - Error logging: ✓")
    print(f"  - Auto-save enabled: Every 25 operations")
    
    try:
        # === PHASE 1: SYSTEM CHECKS (MONITORED) ===
        if run_system_check:
            print("\n" + "="*80)
            print("Running system requirement checks...")
            print("="*80)
            
            # This is now monitored because we patched it above
            with monitor.track("system_requirements_check"):
                sys_check_passed = scanner_module.check_system_requirements()
            
            if not sys_check_passed:
                print("\n⚠️  System requirements not fully met!")
                monitor.log_warning("system_startup", "System requirements check failed")
                
                response = input("Continue anyway? (y/N): ")
                if response.strip().lower() != 'y':
                    print("Exiting...")
                    monitor.record_metric("scan_completion", 0.0, "aborted_system_check")
                    monitor.save_logs()
                    return None, monitor
                
                print("\n⚠️  Proceeding without all requirements - may crash!\n")
                monitor.log_warning("system_startup", "User chose to proceed despite failed checks")
        
        # === PHASE 2: MAIN SCAN (MONITORED) ===
        print("\n" + "="*80)
        print("Running 3D scanner...")
        print("="*80)
        
        # Track the entire scan operation
        with monitor.track("full_3d_scan"):
            result = scanner_module.scan_3d_points(project_dir)
        
        print("\n✓ Scan completed successfully!")
        
        # Record scan success metric
        monitor.record_metric("scan_completion", 1.0, "success")
        
    except Exception as e:
        print(f"\n❌ Scan failed: {e}")
        monitor.record_metric("scan_completion", 0.0, "failed")
        result = None
    
    finally:
        # Save final logs
        monitor.save_logs()
        
        # Generate quality report
        if generate_report:
            print("\n" + "="*80)
            print("QUALITY ANALYSIS")
            print("="*80)
            report = monitor.generate_report(verbose=True)
            
            # Additional analysis
            _analyze_scan_quality(monitor, report)
    
    return result, monitor


def _analyze_scan_quality(monitor, report):
    """Analyze scanner-specific quality metrics."""
    print("\n" + "="*80)
    print("📊 SCANNER-SPECIFIC QUALITY ANALYSIS")
    print("="*80)
    
    # Check function health
    critical_functions = [
        'detect_red_laser_dot',
        'auto_capture_3_points',
        'save_point_cloud'
    ]
    
    print("\n🏥 Critical Function Health:")
    for func_name in critical_functions:
        health = monitor.get_function_health(func_name)
        if health['status'] != 'not_monitored':
            status_icon = "✓" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'degraded' else "❌"
            print(f"\n  {status_icon} {func_name}")
            print(f"     Health Score: {health['health_score']}/100")
            print(f"     Calls: {health['call_count']}")
            print(f"     Avg Time: {health['avg_time_ms']:.2f}ms")
            print(f"     Errors: {health['error_count']}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    recommendations = []
    
    if report['total_errors'] > 0:
        recommendations.append(f"⚠️  {report['total_errors']} errors occurred - review error log")
    
    if report['performance_issues'] > 5:
        recommendations.append(f"🐌 {report['performance_issues']} performance issues - consider optimization")
    
    if report['slowest_functions']:
        slowest = report['slowest_functions'][0]
        if slowest[1] > 1.0:  # More than 1 second
            recommendations.append(f"⚡ Optimize {slowest[0]} (slowest at {slowest[1]*1000:.0f}ms)")
    
    if not recommendations:
        print("  ✓ All systems operating normally!")
    else:
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "="*80)


def compare_with_previous_scans(current_monitor, history_dir="scanning/analysis"):
    """
    Compare current scan quality with previous scans.
    
    Args:
        current_monitor: Current QualityMonitor instance
        history_dir: Directory containing previous scan logs
    """
    import json
    from pathlib import Path
    
    history_path = Path(history_dir)
    if not history_path.exists():
        print("⚠️  No previous scan history found")
        return
    
    # Find previous scan logs
    previous_logs = sorted(history_path.glob("scanner_monitor_*.json"))
    
    if len(previous_logs) < 2:
        print("ℹ️  Not enough history for comparison (need at least 2 scans)")
        return
    
    print("\n" + "="*80)
    print("📈 HISTORICAL COMPARISON")
    print("="*80)
    
    # Load most recent previous scan
    with open(previous_logs[-2], 'r') as f:
        previous_data = json.load(f)
    
    current_report = current_monitor.generate_report(verbose=False)
    
    # Compare key metrics
    prev_errors = len(previous_data.get('error_log', []))
    curr_errors = current_report['total_errors']
    
    prev_perf_issues = len(previous_data.get('performance_issues', []))
    curr_perf_issues = current_report['performance_issues']
    
    print(f"\n📊 Comparison with previous scan:")
    print(f"  Errors: {curr_errors} (previous: {prev_errors}) " + 
          ("📈 worse" if curr_errors > prev_errors else "📉 better" if curr_errors < prev_errors else "➡️ same"))
    print(f"  Performance issues: {curr_perf_issues} (previous: {prev_perf_issues}) " +
          ("📈 worse" if curr_perf_issues > prev_perf_issues else "📉 better" if curr_perf_issues < prev_perf_issues else "➡️ same"))
    
    # Calculate trend
    if curr_errors < prev_errors and curr_perf_issues <= prev_perf_issues:
        print("\n✅ TREND: Quality is improving!")
    elif curr_errors > prev_errors or curr_perf_issues > prev_perf_issues:
        print("\n⚠️  TREND: Quality may be degrading - review recent changes")
    else:
        print("\n➡️  TREND: Quality is stable")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run 3D scanner with quality monitoring"
    )
    parser.add_argument(
        '--project-dir',
        type=str,
        help='Project directory for scanning'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip quality report generation'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with previous scans'
    )
    parser.add_argument(
        '--skip-system-check',
        action='store_true',
        help='Skip system requirements check (not recommended)'
    )
    parser.add_argument(
        '--manual-mode',
        action='store_true',
        help='Use manual function list instead of auto-discovery'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable enhanced error detection (only catch exceptions)'
    )
    
    args = parser.parse_args()
    
    # Run monitored scan (ALL-INCLUSIVE + AUTO-ADAPTIVE + ERROR-AWARE)
    result, monitor = run_monitored_scan(
        project_dir=args.project_dir,
        generate_report=not args.no_report,
        run_system_check=not args.skip_system_check,
        auto_discovery=not args.manual_mode,
        validate_results=not args.no_validation
    )
    
    # Historical comparison if requested
    if args.compare:
        compare_with_previous_scans(monitor)
    
    print("\n✓ Monitoring session complete!")
    print(f"  Check logs at: scanning/analysis/")
