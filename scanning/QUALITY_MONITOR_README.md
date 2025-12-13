# Runtime Quality Monitor for Advanced 3D Scanner

**ALL-INCLUSIVE + AUTO-ADAPTIVE:** AI-powered quality monitoring system that:
- ✅ Tracks **everything** from startup system checks through final save
- ✅ **Automatically detects** when you add new functions (no manual configuration!)
- ✅ **Self-calibrates** to code changes
- ✅ Provides comprehensive analytics for `laser_3d_scanner_advanced.py` and associated scripts

## 📋 Overview

The Runtime Quality Monitor provides **complete coverage** from the moment the scanner starts, with intelligent **auto-discovery** that adapts to code changes:

- **🤖 Auto-Discovery** - Automatically finds and monitors new functions when code changes
- **🚀 Startup Monitoring** - Tracks system requirement checks, calibration loading, initialization
- **📊 Function Execution Tracking** - Monitors timing, call counts, and performance of ALL functions
- **❌ Error Detection & Logging** - Catches and logs all errors with full context
- **⚡ Performance Analysis** - Identifies bottlenecks and slow functions throughout entire execution
- **💾 Memory Monitoring** - Tracks memory usage from startup through completion
- **📈 Quality Metrics** - Custom metrics for scanner-specific quality tracking
- **🏥 Health Scoring** - Individual function health scores (0-100)
- **📊 Historical Comparison** - Compare quality across multiple scan sessions
- **📝 Automated Reporting** - Comprehensive quality reports after each scan

## ✅ What Gets Monitored (ALL-INCLUSIVE)

### Phase 1: Startup & Initialization
✓ `main()` - Main entry point
✓ `check_system_requirements()` - Python version, RAM, CPU, OpenCV, NumPy checks
✓ `find_calibration_file()` - Calibration file detection and loading

### Phase 2: Detection Functions  
✓ `detect_red_laser_dot()` - Core laser detection
✓ `detect_curves()` - Curve detection
✓ `detect_corners()` - Corner detection
✓ `detect_ellipses()` - Ellipse detection
✓ `detect_cylinders()` - Cylinder detection
✓ `detect_laser_with_spectrum()` - Spectrum-based detection

### Phase 3: Analysis & Processing
✓ `suggest_roi_from_contrast()` - ROI suggestion
✓ `estimate_distance_linear()` - Distance estimation  
✓ `run_ai_analysis()` - AI analysis

### Phase 4: Capture & Interaction
✓ `auto_capture_3_points()` - Standard auto-capture
✓ `auto_capture_3_points_with_module()` - Module-based auto-capture
✓ `mouse_callback()` - Mouse interaction
✓ `show_capture_overlay()` - Capture UI overlay
✓ `apply_cartoon_settings()` - Camera settings

### Phase 5: Data Processing & Save
✓ `save_point_cloud()` - Point cloud saving
✓ `scan_3d_points()` - Main scanning loop

**Result:** Complete visibility from system startup through final save!

## 📁 Files

### Main Scripts

1. **`runtime_quality_monitor.py`** - Core monitoring system
   - QualityMonitor class with all tracking functionality
   - Context managers and decorators for easy integration
   - Automated logging and reporting

2. **`auto_discovery_monitor.py`** - 🤖 **NEW!** Auto-discovery system
   - Automatically detects all functions in scanner module
   - Compares with previous runs to find new functions
   - Intelligent categorization and filtering
   - Self-calibrating to code changes

3. **`scanner_with_monitoring.py`** - Integration example
   - Shows how to run the advanced scanner with monitoring
   - **Uses auto-discovery by default** (automatic function detection)
   - Generates quality reports and comparisons

### Generated Files

- **`scanning/analysis/scanner_monitor_YYYYMMDD_HHMMSS.json`** - Monitoring logs
  - Session information
  - Function statistics
  - Execution logs
  - Error and warning logs
  - Performance issues
  - Custom metrics

## 🚀 Quick Start

### Method 1: Run with Monitoring Wrapper (ALL-INCLUSIVE + AUTO-ADAPTIVE)

```bash
# Run the scanner with COMPLETE monitoring from startup
# Uses AUTO-DISCOVERY to find all functions automatically
python scanner_with_monitoring.py

# With project directory
python scanner_with_monitoring.py --project-dir path/to/project

# Compare with previous scans
python scanner_with_monitoring.py --compare

# Use manual function list (legacy mode)
python scanner_with_monitoring.py --manual-mode

# Skip system requirement checks (not recommended)
python scanner_with_monitoring.py --skip-system-check
```

**Auto-Discovery Benefits:**
- ✅ Automatically detects when you add new functions
- ✅ No manual configuration needed
- ✅ Highlights new functions in reports
- ✅ Always monitors all public functions

This automatically monitors:
- ✓ System requirement checks
- ✓ Calibration loading
- ✓ All detection functions
- ✓ Capture and processing
- ✓ Final save operations
- ✓ **Any new functions you add!** 🎉

## 🤖 Auto-Discovery Feature

### Automatically Detects Code Changes!

The monitor now uses **intelligent auto-discovery** to find and monitor functions automatically:

**What it does:**
1. Scans your scanner module for all public functions
2. Categorizes them by purpose (startup, detection, analysis, etc.)
3. Compares with previous run to find new additions
4. Automatically monitors everything - no manual list needed!

**Example - Adding a new function:**

You add this to `laser_3d_scanner_advanced.py`:
```python
def detect_spheres(frame):
    """NEW: Detect spherical objects."""
    # Your code
    pass
```

Next time you run:
```bash
python scanner_with_monitoring.py
```

Output:
```
⭐ DETECTED 1 NEW FUNCTIONS!
   • detect_spheres

📁 DETECTION (8 functions):
   ⭐ NEW: detect_spheres
   ✓ detect_red_laser_dot
   [...]

✓ Auto-patched 19 functions
   New functions will be automatically detected next time!
```

**No manual configuration needed!** See [AUTO_DISCOVERY_GUIDE.md](AUTO_DISCOVERY_GUIDE.md) for complete details.

### Method 2: Integrate into Existing Code

```python
from runtime_quality_monitor import create_scanner_monitor

# Create monitor
monitor = create_scanner_monitor()

# Wrap functions with decorator
@monitor.track_function
def your_function():
    # Your code here
    pass

# Or use context manager
with monitor.track("operation_name"):
    # Your code here
    pass

# Generate report when done
monitor.generate_report()
monitor.save_logs()
```

### Method 3: Programmatic Integration

```python
from scanner_with_monitoring import run_monitored_scan

# Run scan with monitoring
result, monitor = run_monitored_scan(
    project_dir="path/to/project",
    generate_report=True
)

# Access monitoring data
report = monitor.generate_report()
health = monitor.get_function_health('detect_red_laser_dot')
```

## 📊 Monitoring Features

### 1. Function Performance Tracking

Tracks every function call:
- Execution time (min/max/avg)
- Call count
- Memory delta
- Success/failure rate

```python
@monitor.track_function
def detect_red_laser_dot(frame, ...):
    # Function automatically tracked
    pass
```

### 2. Real-time Alerts

Automatically alerts when:
- Function exceeds performance threshold (default: 100ms)
- Memory usage exceeds threshold (default: 500MB)
- Errors occur during execution

```
🐌 SLOW: detect_red_laser_dot took 152.3ms (threshold: 100.0ms)
💾 HIGH MEMORY: auto_capture_3_points using 612.5MB (threshold: 500.0MB)
⚠️  ERROR in save_point_cloud
   FileNotFoundError: Directory not found
```

### 3. Custom Quality Metrics

Track scanner-specific metrics:

```python
# Record detection accuracy
monitor.record_metric("laser_detection_accuracy", 0.95, "red_laser")

# Record point cloud quality
monitor.record_metric("point_cloud_density", 1250.5, "points_per_cm2")

# Record calibration quality
monitor.record_metric("calibration_error", 0.23, "pixels_rms")
```

### 4. Function Health Scoring

Get health score (0-100) for any function:

```python
health = monitor.get_function_health('detect_red_laser_dot')
# Returns:
# {
#     'function': 'detect_red_laser_dot',
#     'health_score': 85,
#     'call_count': 450,
#     'avg_time_ms': 45.2,
#     'error_count': 0,
#     'warning_count': 2,
#     'status': 'healthy'  # or 'degraded' or 'critical'
# }
```

### 5. Comprehensive Reports

Generates detailed quality reports:

```
================================================================================
📊 QUALITY MONITOR REPORT
================================================================================

⏱️  Session Duration: 145.3s
📞 Total Function Calls: 1,247
🔧 Unique Functions: 12

❌ Errors: 0
⚠️  Warnings: 3
🐌 Performance Issues: 5

🐌 Slowest Functions:
   1. auto_capture_3_points: 234.56ms
   2. save_point_cloud: 156.78ms
   3. detect_red_laser_dot: 89.12ms

📞 Most Called Functions:
   1. detect_red_laser_dot: 450 calls
   2. estimate_distance_linear: 450 calls
   3. detect_curves: 125 calls
```

## 🔧 Configuration

### Monitor Settings

```python
monitor = QualityMonitor(
    log_file="custom_log.json",           # Log file path
    enable_alerts=True,                    # Show real-time alerts
    performance_threshold_ms=100,          # Alert if function > 100ms
    memory_threshold_mb=500,               # Alert if memory > 500MB
    auto_save_interval=50                  # Auto-save every 50 operations
)
```

### Functions Monitored by Default (ALL-INCLUSIVE)

When using `scanner_with_monitoring.py`, **ALL critical functions** are automatically tracked from startup to completion:

**Startup & System:**
- `main` - Main entry point
- `check_system_requirements` - System requirement validation  
- `find_calibration_file` - Calibration detection

**Detection:**
- `detect_red_laser_dot` - Laser dot detection
- `detect_curves` - Curve detection
- `detect_corners` - Corner detection
- `detect_ellipses` - Ellipse detection
- `detect_cylinders` - Cylinder detection
- `detect_laser_with_spectrum` - Spectrum-based detection

**Analysis:**
- `suggest_roi_from_contrast` - ROI suggestion
- `estimate_distance_linear` - Distance estimation
- `run_ai_analysis` - AI analysis

**Capture:**
- `auto_capture_3_points` - Auto-capture mode
- `auto_capture_3_points_with_module` - Module-based capture
- `mouse_callback` - Mouse interaction
- `show_capture_overlay` - Capture overlay UI
- `apply_cartoon_settings` - Camera settings

**Processing:**
- `save_point_cloud` - Point cloud saving
- `scan_3d_points` - Main scanning function

## 📈 Use Cases

### 1. Debugging Performance Issues

```bash
# Run with monitoring to identify slow functions
python scanner_with_monitoring.py

# Check report for slowest functions
# Optimize the top bottlenecks
```

### 2. Quality Assurance

```python
# Run regular monitored scans
result, monitor = run_monitored_scan()

# Check function health
health = monitor.get_function_health('detect_red_laser_dot')
if health['status'] == 'critical':
    print("⚠️  Detection function needs attention!")
```

### 3. Regression Testing

```bash
# Compare current scan with previous
python scanner_with_monitoring.py --compare

# Check if quality is improving or degrading
```

### 4. Development Workflow

```python
# During development, track new feature quality
monitor = create_scanner_monitor()

@monitor.track_function
def new_experimental_feature():
    # Test new feature
    pass

# Check if it meets performance requirements
health = monitor.get_function_health('new_experimental_feature')
```

## 📝 Log File Format

The JSON log file contains:

```json
{
  "session_info": {
    "start_time": "2025-12-11T10:30:00",
    "duration_seconds": 145.3,
    "total_operations": 1247
  },
  "function_statistics": {
    "detect_red_laser_dot": {
      "call_count": 450,
      "total_time": 20.34,
      "avg_time": 0.045,
      "min_time": 0.023,
      "max_time": 0.089,
      "errors": [],
      "warnings": []
    }
  },
  "execution_log": [...],
  "error_log": [...],
  "warning_log": [...],
  "performance_issues": [...],
  "memory_snapshots": [...],
  "custom_metrics": {...}
}
```

## 🛠️ Advanced Usage

### Monitoring Code Blocks

```python
with monitor.track("image_preprocessing"):
    # Preprocess image
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

# Block timing and memory tracked automatically
```

### Manual Warning Logging

```python
if detection_confidence < 0.5:
    monitor.log_warning(
        "detect_red_laser_dot",
        f"Low confidence: {detection_confidence:.2f}"
    )
```

### Custom Metric Analysis

```python
# Record multiple measurements
for frame in frames:
    accuracy = calculate_accuracy(frame)
    monitor.record_metric("detection_accuracy", accuracy)

# Later, analyze metrics
metrics = monitor.custom_metrics['detection_accuracy']
avg_accuracy = sum(m['value'] for m in metrics) / len(metrics)
print(f"Average accuracy: {avg_accuracy:.2%}")
```

## 🔍 Troubleshooting

### Monitor Not Tracking Functions

**Problem**: Functions aren't being tracked

**Solution**: Ensure functions are properly decorated or patched:

```python
# Option 1: Decorator
@monitor.track_function
def my_function():
    pass

# Option 2: Manual patching
original_func = module.my_function
module.my_function = monitor.track_function(original_func)
```

### High Performance Overhead

**Problem**: Monitoring slows down execution

**Solution**: Adjust monitoring intervals:

```python
monitor = QualityMonitor(
    auto_save_interval=100,  # Save less frequently
    enable_alerts=False       # Disable console alerts
)
```

### Log Files Too Large

**Problem**: JSON logs become very large

**Solution**: Logs are automatically truncated to last N entries:
- Execution log: Last 100 entries
- Memory snapshots: Last 50 entries

You can manually save more frequently with smaller chunks.

## 🤝 Integration with Existing Analysis

The Runtime Quality Monitor complements existing analysis tools:

- **`analysis/sphere_analyzer.py`** - Sphere matrix accuracy
- **`analysis/calibration_profiler.py`** - Calibration quality
- **`ai_analysis/optimized_analyzer.py`** - AI frame analysis
- **`ai_analysis/analyze_sessions.py`** - Session comparison

Use Runtime Quality Monitor for:
- Real-time execution monitoring
- Performance profiling
- Error tracking
- Function-level health

Use existing tools for:
- Domain-specific analysis (sphere accuracy, calibration)
- AI-powered frame analysis
- Post-session review

## 📚 API Reference

See docstrings in `runtime_quality_monitor.py` for complete API documentation.

Key classes:
- `QualityMonitor` - Main monitoring class
- `_TrackingContext` - Context manager for code blocks

Key functions:
- `create_scanner_monitor()` - Create pre-configured monitor
- `patch_scanner_functions()` - Patch scanner module
- `run_monitored_scan()` - Run scanner with monitoring

## 📄 License

This monitoring system is part of the Advanced 3D Scanner project.

## 🎯 Summary

The Runtime Quality Monitor provides comprehensive, real-time quality tracking for the Advanced 3D Scanner. Use it to:

✅ Identify performance bottlenecks
✅ Track function health over time
✅ Catch and log errors automatically
✅ Monitor resource usage
✅ Generate quality reports
✅ Compare scan quality across sessions

**Recommended Usage**: Run all production scans with monitoring enabled to maintain quality awareness and catch issues early.
