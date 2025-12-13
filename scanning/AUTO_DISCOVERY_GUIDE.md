# Auto-Discovery & Auto-Calibration Guide

## ✅ YES - It Automatically Senses Changes!

The quality monitor now has **Auto-Discovery** that automatically detects when you add, remove, or modify functions in your scanner code.

---

## 🤖 How Auto-Discovery Works

### 1. **First Run** - Creates Baseline
```bash
python scanner_with_monitoring.py
```

Output:
```
🤖 INITIALIZING AUTO-DISCOVERY MONITOR
✓ Loaded module: laser_3d_scanner_advanced

🔍 AUTO-DISCOVERY REPORT
✓ Discovered 18 monitorable functions

📁 STARTUP (3 functions):
   ✓ check_system_requirements
   ✓ find_calibration_file
   ✓ main

📁 DETECTION (7 functions):
   ✓ detect_corners
   ✓ detect_curves
   ✓ detect_cylinders
   ✓ detect_ellipses
   ✓ detect_laser_with_spectrum
   ✓ detect_red_laser_dot

[... all categories ...]

✓ Function snapshot saved to scanning/analysis/function_snapshot.txt
```

### 2. **You Add a New Function**

In `laser_3d_scanner_advanced.py`:
```python
def detect_spheres(frame):
    """NEW: Detect spherical objects."""
    # Your new detection code
    pass
```

### 3. **Next Run** - Automatically Detects Changes
```bash
python scanner_with_monitoring.py
```

Output:
```
🤖 INITIALIZING AUTO-DISCOVERY MONITOR
♻️  Reloaded module: laser_3d_scanner_advanced

⭐ DETECTED 1 NEW FUNCTIONS!
   • detect_spheres

🔍 AUTO-DISCOVERY REPORT
✓ Discovered 19 monitorable functions

📁 DETECTION (8 functions):
   ⭐ NEW: detect_spheres    ← AUTOMATICALLY FOUND!
   ✓ detect_corners
   ✓ detect_curves
   [...]

🔧 Patching all discovered functions...
✓ Auto-patched 19 functions
   New functions will be automatically detected next time!
```

**The new function is automatically monitored!** No manual configuration needed.

---

## 🎯 What Gets Auto-Detected

### ✅ Automatically Monitored:
- All public functions (starting with lowercase)
- Main entry points
- Detection functions
- Analysis functions
- Capture functions
- Processing functions

### ❌ Automatically Excluded:
- Private functions (`_function_name`)
- Magic methods (`__init__`, `__str__`)
- Test functions (`test_something`)

---

## 📊 Auto-Categorization

Functions are automatically categorized based on their names:

| Category | Keywords Detected | Example Functions |
|----------|------------------|-------------------|
| **Startup** | main, check, init, setup, calibration, load | `check_system_requirements`, `find_calibration_file` |
| **Detection** | detect, find, locate, laser, curve, corner | `detect_red_laser_dot`, `detect_spheres` |
| **Analysis** | analyze, estimate, calculate, ai, quality | `run_ai_analysis`, `estimate_distance_linear` |
| **Capture** | capture, grab, record, auto_capture, mouse | `auto_capture_3_points`, `mouse_callback` |
| **Processing** | process, scan, save, export, cloud, mesh | `save_point_cloud`, `scan_3d_points` |
| **Utility** | Everything else | Helper functions |

---

## 🔄 Change Detection Examples

### Example 1: Adding a Function
**Before:**
```python
# Only has detect_red_laser_dot()
```

**You add:**
```python
def detect_green_laser_dot(frame):
    """Detect green laser (532nm)."""
    pass
```

**Next run:**
```
⭐ DETECTED 1 NEW FUNCTIONS!
   • detect_green_laser_dot

📁 DETECTION:
   ⭐ NEW: detect_green_laser_dot
```

### Example 2: Renaming a Function
**Before:**
```python
def old_function_name():
    pass
```

**You change to:**
```python
def new_improved_function():
    pass
```

**Next run:**
```
⭐ DETECTED 1 NEW FUNCTIONS!
   • new_improved_function

⚠️  Note: old_function_name no longer detected
```

### Example 3: Adding Multiple Functions
**You add:**
```python
def detect_spheres(frame):
    pass

def detect_triangles(frame):
    pass

def calculate_volume(points):
    pass
```

**Next run:**
```
⭐ DETECTED 3 NEW FUNCTIONS!
   • calculate_volume
   • detect_spheres
   • detect_triangles

📁 DETECTION:
   ⭐ NEW: detect_spheres
   ⭐ NEW: detect_triangles

📁 ANALYSIS:
   ⭐ NEW: calculate_volume
```

---

## 🛠️ Usage Modes

### Mode 1: Auto-Discovery (Default) ✅
```bash
# Automatically finds and monitors ALL functions
python scanner_with_monitoring.py
```

**Best for:**
- Active development
- Frequently adding new features
- Want zero manual configuration

### Mode 2: Manual Mode (Legacy)
```bash
# Uses predefined function list
python scanner_with_monitoring.py --manual-mode
```

**Best for:**
- Production environments
- Stable codebases
- Need explicit control

---

## 📁 Generated Files

### Function Snapshot File
Location: `scanning/analysis/function_snapshot.txt`

Contains:
```
# Function Snapshot
# Module: laser_3d_scanner_advanced
# Total Functions: 18

[STARTUP]
check_system_requirements
find_calibration_file
main

[DETECTION]
detect_corners
detect_curves
detect_cylinders
...
```

This file is used to detect changes between runs.

---

## 🔧 Customization

### Custom Exclude Patterns

Edit `auto_discovery_monitor.py`:
```python
exclude_patterns = [
    r'^_.*',           # Private functions
    r'^__.*__$',       # Magic methods
    r'^test_.*',       # Test functions
    r'^temp_.*',       # Add: Skip temporary functions
    r'^debug_.*',      # Add: Skip debug functions
]
```

### Custom Include Patterns

```python
include_patterns = [
    r'^[a-z].*',       # Public functions
    r'^[A-Z].*',       # Classes
    r'^MY_.*',         # Add: Your custom prefix
]
```

---

## 📈 Benefits of Auto-Discovery

### ✅ Advantages

1. **Zero Configuration**
   - No need to manually add function names
   - No need to update monitoring lists

2. **Always Up-to-Date**
   - New functions automatically monitored
   - Removed functions automatically unmonitored

3. **Change Awareness**
   - See exactly what changed since last run
   - Track when new functions were added

4. **Comprehensive Coverage**
   - Won't miss any new functions
   - All public functions automatically included

5. **Smart Categorization**
   - Functions automatically organized by purpose
   - Easy to see what type of functions you have

### ⚠️ Considerations

1. **First-Run Setup**
   - First run creates baseline snapshot
   - Subsequent runs compare against it

2. **Private Functions**
   - Functions starting with `_` are excluded
   - Mark internal helpers as private to skip them

3. **Performance**
   - Module inspection adds ~0.1s startup time
   - Negligible impact on scan performance

---

## 🎯 Workflow Examples

### Scenario 1: Adding a New Feature

1. **Write new detection function:**
```python
def detect_qr_codes(frame):
    """Detect QR codes in frame."""
    # Your code
    pass
```

2. **Run scanner:**
```bash
python scanner_with_monitoring.py
```

3. **Automatically monitored:**
```
⭐ DETECTED 1 NEW FUNCTIONS!
   • detect_qr_codes
   
✓ Auto-patched 19 functions
```

4. **Check quality report:**
```
📊 QUALITY MONITOR REPORT

🔧 Unique Functions: 19

📁 DETECTION:
   ⭐ NEW: detect_qr_codes: 23 calls, 45.2ms avg
```

**No manual steps required!**

### Scenario 2: Refactoring Code

1. **Split large function into smaller ones:**
```python
# Before: One large function
def process_everything(data):
    # 500 lines of code
    pass

# After: Multiple focused functions
def validate_input(data):
    pass

def transform_data(data):
    pass

def analyze_results(data):
    pass
```

2. **Run scanner:**
```bash
python scanner_with_monitoring.py
```

3. **All new functions automatically tracked:**
```
⭐ DETECTED 3 NEW FUNCTIONS!
   • analyze_results
   • transform_data
   • validate_input

Each function now has individual monitoring!
```

---

## 🚀 Quick Reference

| Command | What It Does |
|---------|-------------|
| `python scanner_with_monitoring.py` | Auto-discovery ON (default) |
| `python scanner_with_monitoring.py --manual-mode` | Auto-discovery OFF |
| Check `function_snapshot.txt` | See current function list |
| Look for ⭐ NEW in output | See newly detected functions |

---

## 💡 Pro Tips

1. **Regular Snapshots**
   - Snapshot is updated each run
   - Safe to delete to reset baseline

2. **Review New Functions**
   - Check report after adding features
   - Verify new functions are categorized correctly

3. **Performance Tracking**
   - New functions automatically get performance metrics
   - Compare with similar existing functions

4. **Version Control**
   - Commit `function_snapshot.txt` with your code
   - Track function evolution over time

---

## ❓ FAQ

**Q: Does it monitor changes within functions?**
A: No, it monitors function existence, not internal changes. Use the quality metrics to track performance changes.

**Q: What if I rename a function?**
A: It appears as one removed, one added. The monitor tracks the new name.

**Q: Can I exclude specific functions?**
A: Yes, either make them private (`_function`) or customize exclude patterns.

**Q: Does it work with class methods?**
A: Yes, class methods are detected and monitored automatically.

**Q: How do I reset the baseline?**
A: Delete `scanning/analysis/function_snapshot.txt` and run again.

---

## 🎉 Summary

### Question: Will the quality monitor automatically sense changes and calibrate for additions?

### Answer: **YES!**

✅ Automatically detects new functions when you add them
✅ Automatically monitors them (no configuration needed)
✅ Automatically categorizes them by purpose
✅ Automatically alerts you to what changed
✅ Automatically updates the function snapshot
✅ Automatically generates metrics for new functions

**It's fully automatic and self-calibrating!** 🤖
