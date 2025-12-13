# Broken Function Detection Guide

## ✅ YES - It Detects Broken Functions!

The quality monitor detects **ALL types of failures** - not just exceptions, but also silent failures, invalid returns, and suspicious behavior.

---

## 🎯 Types of Failures Detected

### 1. **Exceptions** ✅ (Already Built-In)

**What it catches:**
- Runtime errors (IndexError, ValueError, etc.)
- Type errors
- Import errors
- Any uncaught exception

**Example:**
```python
def detect_red_laser_dot(frame):
    # This will crash
    return frame[1000000, 1000000]  # Index out of bounds!
```

**Monitor Output:**
```
⚠️  ERROR in detect_red_laser_dot
   IndexError: index 1000000 is out of bounds for axis 0 with size 720

📊 Function Health:
   detect_red_laser_dot: 0/100 (critical)
   Error count: 1
```

---

### 2. **Silent Failures** ✅ (Enhanced Detection)

**What it catches:**
- Functions returning None when they shouldn't
- Wrong return types
- Invalid values
- Out-of-range results

**Example 1: Returns None unexpectedly**
```python
def save_point_cloud(points):
    # Forgot to return True
    # Just saves and returns None
    np.save("points.npy", points)
    # Missing: return True
```

**Monitor Output:**
```
⚠️  WARNING in save_point_cloud
   Silent failure detected: Returned None (expected non-null)
```

**Example 2: Wrong return type**
```python
def check_system_requirements():
    # Should return bool, but returns string!
    return "System OK"
```

**Monitor Output:**
```
⚠️  WARNING in check_system_requirements
   Silent failure detected: Wrong type: got str, expected ['bool']
```

**Example 3: Invalid value**
```python
def estimate_distance_linear(dot_y):
    # Returns negative distance (impossible!)
    return -50.0
```

**Monitor Output:**
```
⚠️  WARNING in estimate_distance_linear
   Silent failure detected: Value -50.0 below minimum 0
```

---

### 3. **Performance Anomalies** ✅ (Enhanced Detection)

**What it catches:**
- Functions running 3x slower than normal
- May indicate internal errors or degradation

**Example:**
```python
def detect_red_laser_dot(frame):
    # Normal: ~45ms
    # Now suddenly: ~450ms (10x slower!)
    # Might indicate frame corruption or algorithm issue
```

**Monitor Output:**
```
⚠️  WARNING in detect_red_laser_dot
   Performance anomaly: 450.5ms (avg: 45.2ms, 3x slower!)
```

---

### 4. **Consecutive Failures** ✅ (Enhanced Detection)

**What it catches:**
- Pattern of repeated failures
- Critical alert after 5 consecutive failures

**Example:**
```python
# Function fails 5 times in a row
```

**Monitor Output:**
```
⚠️  CRITICAL: 5 consecutive validation failures in detect_red_laser_dot!
```

---

## 📊 Complete Failure Detection Matrix

| Failure Type | Detection Method | Example | Severity |
|--------------|-----------------|---------|----------|
| **Exception** | Try-catch wrapper | IndexError, ValueError | 🔴 Critical |
| **Returns None** | Type validation | `return None` when shouldn't | 🟡 Warning |
| **Wrong Type** | Type checking | Returns str instead of bool | 🟡 Warning |
| **Out of Range** | Value validation | Negative distance | 🟡 Warning |
| **Too Slow** | Performance baseline | 3x slower than average | 🟠 Warning |
| **Repeated Fails** | Pattern detection | 5+ consecutive failures | 🔴 Critical |

---

## 🛠️ How It Works

### Step 1: Monitor Wraps Functions

```python
@monitor.track_function
def detect_red_laser_dot(frame):
    # Your code
    result = (x, y, radius)
    return result
```

### Step 2: Function Called

```python
result = detect_red_laser_dot(frame)
```

### Step 3: Monitor Checks

```
1. Did it throw an exception? ❌ No
2. Did it return None when shouldn't? ❌ No  
3. Is the return type correct? ✅ Yes (tuple)
4. Is the value in valid range? ✅ Yes
5. Did it take too long? ❌ No (45ms, normal)

✅ Function is healthy!
```

### Step 4: If Failure Detected

```python
# Exception thrown
⚠️  ERROR logged with full stack trace
Health score drops to 0
Status: critical

# Silent failure (wrong return)
⚠️  WARNING logged
Health score reduced by 5 points per failure
Status: degraded (if score 50-79) or critical (if <50)
```

---

## 🚀 Usage Examples

### Example 1: Basic Monitoring (Exceptions Only)

```bash
python scanner_with_monitoring.py
```

**Detects:**
- ✅ Exceptions
- ✅ Crashes
- ❌ Silent failures (not enabled)

### Example 2: Enhanced Monitoring (Recommended)

```bash
python scanner_with_monitoring.py
# Enhanced validation enabled by default
```

**Detects:**
- ✅ Exceptions
- ✅ Silent failures
- ✅ Wrong types
- ✅ Invalid values
- ✅ Performance anomalies

### Example 3: Disable Enhanced Detection

```bash
python scanner_with_monitoring.py --no-validation
```

**Detects:**
- ✅ Exceptions only
- ❌ Silent failures (disabled)

---

## 📋 Function-Specific Validation Rules

The monitor has built-in rules for scanner functions:

| Function | Expected Return | Range | Non-Null |
|----------|----------------|-------|----------|
| `detect_red_laser_dot` | tuple or None | N/A | No |
| `estimate_distance_linear` | float/int | 0-10000 cm | No |
| `save_point_cloud` | bool | N/A | Yes |
| `check_system_requirements` | bool | N/A | Yes |
| `auto_capture_3_points` | dict or None | N/A | No |

**Example:**
```python
# This will trigger validation warning:
def save_point_cloud(points):
    return None  # ⚠️  Should return True!

# This will trigger range warning:
def estimate_distance_linear(y):
    return -50  # ⚠️  Negative distance invalid!
```

---

## 🔍 Real-World Detection Examples

### Scenario 1: Broken Laser Detection

**Code:**
```python
def detect_red_laser_dot(frame):
    try:
        # Broken: wrong channel indexing
        red_channel = frame[:,:,5]  # Only 0-2 exist!
    except:
        return None  # Silently fails
```

**Detection:**
```
⚠️  ERROR in detect_red_laser_dot
   IndexError: index 5 is out of bounds for axis 2 with size 3

📊 Health: 0/100 (critical)
```

### Scenario 2: Corrupted Distance Calculation

**Code:**
```python
def estimate_distance_linear(dot_y):
    distance = (dot_y - 360) * -1.5  # Formula error
    return distance  # Returns negative for top half
```

**Detection:**
```
⚠️  WARNING in estimate_distance_linear
   Silent failure detected: Value -120.0 below minimum 0

After 5 calls:
⚠️  CRITICAL: 5 consecutive validation failures!
```

### Scenario 3: Failed System Check

**Code:**
```python
def check_system_requirements():
    # Should return bool, but returns string
    if all_ok:
        return "All checks passed"  # Wrong type!
```

**Detection:**
```
⚠️  WARNING in check_system_requirements
   Silent failure detected: Wrong type: got str, expected ['bool']
```

### Scenario 4: Performance Degradation

**Code:**
```python
def detect_red_laser_dot(frame):
    # Memory leak causes slowdown over time
    # Normal: 45ms
    # After 100 calls: 450ms (degraded)
```

**Detection:**
```
⚠️  WARNING in detect_red_laser_dot
   Performance anomaly: 450.5ms (avg: 45.2ms, 10x slower!)

Recommendation: Check for memory leaks or resource exhaustion
```

---

## 📈 Health Scoring with Failures

### Perfect Health (100/100)
```python
✅ No exceptions
✅ No validation failures
✅ Performance normal
✅ Returns expected types
```

### Degraded (50-79/100)
```python
⚠️  1-3 validation warnings
⚠️  OR performance anomaly detected
⚠️  No exceptions
```

### Critical (0-49/100)
```python
❌ Exceptions occurred
❌ OR 5+ consecutive validation failures
❌ OR multiple errors
```

---

## 🎯 Monitoring Report Examples

### Clean Run (No Issues)

```
📊 QUALITY MONITOR REPORT

✅ Errors: 0
⚠️  Warnings: 0

🏥 Critical Function Health:
   ✓ detect_red_laser_dot: 100/100 (healthy)
   ✓ auto_capture_3_points: 98/100 (healthy)
   ✓ save_point_cloud: 100/100 (healthy)

💡 Recommendations:
   ✓ All systems operating normally!
```

### Run with Issues

```
📊 QUALITY MONITOR REPORT

❌ Errors: 2
⚠️  Warnings: 7

🏥 Critical Function Health:
   ❌ detect_red_laser_dot: 30/100 (critical)
      - 2 exceptions
      - 5 validation failures
   ⚠️  estimate_distance_linear: 65/100 (degraded)
      - 7 out-of-range values
   ✓ save_point_cloud: 95/100 (healthy)

❌ Recent Errors:
   [10:30:45] detect_red_laser_dot
      IndexError: index out of bounds

⚠️  Recent Warnings:
   [10:31:12] estimate_distance_linear
      Silent failure: Value -50.0 below minimum 0
   [10:31:15] estimate_distance_linear
      Performance anomaly: 450ms (avg: 45ms)

💡 Recommendations:
   ❌ Fix critical issues in detect_red_laser_dot
   ⚠️  Review estimate_distance_linear range validation
```

---

## 🛠️ Customizing Validation Rules

### Add Custom Rules

```python
from enhanced_error_detection import ErrorValidator

validator = ErrorValidator(monitor)

# Custom function validation
validator.expect_return_type('my_custom_function', dict)
validator.expect_non_null('my_custom_function')
validator.expect_value_range('my_metric', min_val=0, max_val=100)
```

### Validate Manually

```python
result = my_custom_function()
is_valid = validator.validate_result('my_custom_function', result)

if not is_valid:
    print("⚠️  Function returned suspicious result!")
```

---

## 💡 Best Practices

### 1. **Always Use Enhanced Detection**
```bash
# Recommended
python scanner_with_monitoring.py

# Not recommended
python scanner_with_monitoring.py --no-validation
```

### 2. **Review Warnings Regularly**
- Check console output during runs
- Review JSON logs after sessions
- Fix validation warnings before they become errors

### 3. **Set Realistic Ranges**
- Distance: 0-10000 cm (reasonable for scanner)
- Confidence: 0.0-1.0 (probability)
- Coordinates: Based on frame size

### 4. **Monitor Health Scores**
```python
health = monitor.get_function_health('critical_function')
if health['health_score'] < 80:
    print("⚠️  Function needs attention!")
```

---

## 📚 Summary

### Question: Does it recognize if functions are called but broken?

### Answer: **YES - Multiple Ways!**

✅ **Exceptions**: Catches and logs all runtime errors  
✅ **Silent Failures**: Detects None returns, wrong types, invalid values  
✅ **Performance Issues**: Alerts when functions run abnormally slow  
✅ **Pattern Detection**: Warns about consecutive failures  
✅ **Health Scoring**: Tracks overall function reliability  
✅ **Detailed Logging**: Full stack traces and context  

**The monitor doesn't just detect crashes - it detects ANY malfunction!** 🔍🛡️

---

## 📖 Related Documentation

- [QUALITY_MONITOR_README.md](QUALITY_MONITOR_README.md) - Main documentation
- [AUTO_DISCOVERY_GUIDE.md](AUTO_DISCOVERY_GUIDE.md) - Auto-discovery feature
- [enhanced_error_detection.py](enhanced_error_detection.py) - Implementation details
