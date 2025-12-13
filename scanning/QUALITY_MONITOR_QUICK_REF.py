"""
Quick Reference: Runtime Quality Monitor
=========================================
ALL-INCLUSIVE monitoring from system startup through final save

INSTALLATION
------------
No installation needed! Just use the scripts directly.

Required dependency:
    pip install psutil

QUICK START
-----------

1. Run scanner with monitoring:
   
   python scanner_with_monitoring.py


2. Run with comparison:
   
   python scanner_with_monitoring.py --compare


3. Custom integration:
   
   from runtime_quality_monitor import create_scanner_monitor
   
   monitor = create_scanner_monitor()
   
   @monitor.track_function
   def my_function():
       pass
   
   monitor.generate_report()


WHAT GETS MONITORED
-------------------

✅ ALL-INCLUSIVE COVERAGE:

Phase 1 - Startup:
  ✓ main() - Entry point
  ✓ check_system_requirements() - System checks
  ✓ find_calibration_file() - Calibration loading

Phase 2 - Detection:
  ✓ detect_red_laser_dot() - Laser detection
  ✓ detect_curves/corners/ellipses/cylinders()
  ✓ detect_laser_with_spectrum()

Phase 3 - Analysis:
  ✓ run_ai_analysis()
  ✓ estimate_distance_linear()
  ✓ suggest_roi_from_contrast()

Phase 4 - Capture:
  ✓ auto_capture_3_points()
  ✓ mouse_callback()
  ✓ apply_cartoon_settings()

Phase 5 - Save:
  ✓ save_point_cloud()
  ✓ scan_3d_points()


COMMON COMMANDS
---------------

# Basic scan with monitoring
python scanner_with_monitoring.py

# Specify project directory
python scanner_with_monitoring.py --project-dir D:/my_project

# Skip final report
python scanner_with_monitoring.py --no-report

# Compare with previous scans
python scanner_with_monitoring.py --compare

# Skip system requirement checks (not recommended)
python scanner_with_monitoring.py --skip-system-check


MONITORING PHASES
----------------------

# High-performance threshold (alert only if > 200ms)
monitor = QualityMonitor(
    performance_threshold_ms=200
)

# Lower memory threshold (alert if > 300MB)
monitor = QualityMonitor(
    memory_threshold_mb=300
)

# Quiet mode (no console alerts)
monitor = QualityMonitor(
    enable_alerts=False
)

# Save less frequently (every 100 operations)
monitor = QualityMonitor(
    auto_save_interval=100
)


COMMON PATTERNS
---------------

# Pattern 1: Decorator
@monitor.track_function
def process_frame(frame):
    return processed

# Pattern 2: Context Manager
with monitor.track("preprocessing"):
    frame = preprocess(frame)

# Pattern 3: Manual Metrics
monitor.record_metric("accuracy", 0.95)

# Pattern 4: Warnings
if suspicious_condition:
    monitor.log_warning("function_name", "Issue detected")


READING REPORTS
---------------

Report shows:
- Session duration
- Total function calls
- Error count
- Warning count
- Slowest functions
- Most called functions

Example output:
    ⏱️  Session Duration: 145.3s
    📞 Total Function Calls: 1,247
    ❌ Errors: 0
    ⚠️  Warnings: 3
    🐌 Slowest Functions:
       1. auto_capture_3_points: 234.56ms


HEALTH SCORES
-------------

100-80: ✓ Healthy (green)
79-50:  ⚠️ Degraded (yellow)
49-0:   ❌ Critical (red)

health = monitor.get_function_health('my_function')
print(health['status'])  # 'healthy', 'degraded', or 'critical'


ALERTS EXPLAINED
----------------

🐌 SLOW: Function exceeded performance threshold
💾 HIGH MEMORY: Process exceeded memory threshold
⚠️  ERROR: Exception occurred in function
⚠️  WARNING: Custom warning logged


FILE LOCATIONS
--------------

Logs saved to:
    scanning/analysis/scanner_monitor_YYYYMMDD_HHMMSS.json

Contains:
- Session info
- Function statistics
- Execution log
- Error log
- Performance issues
- Custom metrics


TROUBLESHOOTING
---------------

Q: Functions not being tracked?
A: Make sure you're using the decorator or context manager

Q: Too many alerts?
A: Increase thresholds or disable alerts:
   monitor = QualityMonitor(enable_alerts=False)

Q: Monitor slowing things down?
A: Increase auto-save interval:
   monitor = QualityMonitor(auto_save_interval=100)

Q: Log files too large?
A: They auto-truncate, but save more frequently if needed


INTEGRATION CHECKLIST
---------------------

☐ Import monitor: from runtime_quality_monitor import create_scanner_monitor
☐ Create monitor: monitor = create_scanner_monitor()
☐ Decorate functions: @monitor.track_function
☐ Or use context: with monitor.track("name"):
☐ Generate report: monitor.generate_report()
☐ Save logs: monitor.save_logs()


BEST PRACTICES
--------------

✓ Always run production scans with monitoring
✓ Review reports after each scan session
✓ Compare quality across sessions regularly
✓ Set realistic thresholds for your hardware
✓ Track custom metrics for domain-specific quality
✓ Keep historical logs for trend analysis


EXAMPLE WORKFLOW
----------------

1. Development:
   - Write new function
   - Add @monitor.track_function decorator
   - Test and check performance

2. Testing:
   - Run with monitoring enabled
   - Review health scores
   - Fix any critical/degraded functions

3. Production:
   - Always use scanner_with_monitoring.py
   - Compare with --compare flag periodically
   - Review logs weekly


FUNCTION HEALTH CHECK
---------------------

Check before deployment:

health = monitor.get_function_health('critical_function')
if health['health_score'] < 80:
    print("⚠️  Function needs optimization!")
if health['error_count'] > 0:
    print("❌ Fix errors before deployment!")


CUSTOM METRICS GUIDE
--------------------

# Detection quality
monitor.record_metric("detection_accuracy", 0.95, "laser_dot")

# Point cloud quality
monitor.record_metric("point_density", 1250, "points/cm²")

# Calibration quality
monitor.record_metric("calibration_error", 0.23, "pixels_rms")

# Processing speed
monitor.record_metric("fps", 28.5, "camera_capture")


COMPARISON MODE
---------------

python scanner_with_monitoring.py --compare

Shows:
- Error count vs previous scan
- Performance issues vs previous scan
- Trend: improving/degrading/stable

Example output:
    📊 Comparison with previous scan:
      Errors: 0 (previous: 2) 📉 better
      Performance issues: 3 (previous: 5) 📉 better
    
    ✅ TREND: Quality is improving!


TERMINAL OUTPUT
---------------

Real-time monitoring shows:

🔧 Patching scanner functions...
  ✓ Patched: detect_red_laser_dot
  ✓ Patched: auto_capture_3_points
  [...]

During scan:
🐌 SLOW: detect_red_laser_dot took 152.3ms
💾 HIGH MEMORY: auto_capture_3_points using 612.5MB

After scan:
📊 QUALITY MONITOR REPORT
[Detailed statistics]


WHEN TO USE
-----------

USE for:
✓ Performance profiling
✓ Error tracking
✓ Quality assurance
✓ Regression testing
✓ Development debugging

DON'T USE for:
✗ Simple testing (adds overhead)
✗ When speed is absolutely critical
✗ If psutil not available


CONTACT & SUPPORT
-----------------

For issues or questions:
- Check QUALITY_MONITOR_README.md for detailed docs
- Review runtime_quality_monitor.py docstrings
- Check scanner_with_monitoring.py examples
"""

if __name__ == "__main__":
    print(__doc__)
