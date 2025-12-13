"""
Enhanced Error Detection for Runtime Quality Monitor
=====================================================
Detects not just exceptions, but also:
- Silent failures (returns None when shouldn't)
- Invalid return values
- Performance degradation (might indicate issues)
- Functions that succeed but produce suspicious results

Usage:
    from enhanced_error_detection import ErrorValidator
    
    monitor = create_scanner_monitor()
    validator = ErrorValidator(monitor)
    
    # Set expected return types
    validator.expect_return_type('detect_red_laser_dot', tuple)
    validator.expect_non_null('detect_red_laser_dot')
    
    # Validate during execution
    result = detect_red_laser_dot(frame)
    validator.validate_result('detect_red_laser_dot', result)
"""

import inspect
from typing import Any, Optional, Callable, Type, Union, List
import warnings


class ErrorValidator:
    """
    Enhanced error detection beyond exceptions.
    Detects silent failures, invalid returns, and suspicious behavior.
    """
    
    def __init__(self, quality_monitor):
        """
        Initialize error validator.
        
        Args:
            quality_monitor: QualityMonitor instance to log issues to
        """
        self.monitor = quality_monitor
        
        # Expected return types per function
        self.expected_types = {}
        
        # Functions that should never return None
        self.non_null_functions = set()
        
        # Expected value ranges
        self.value_ranges = {}
        
        # Performance baselines (auto-learned)
        self.performance_baselines = {}
        
        # Suspicious pattern tracking
        self.consecutive_failures = {}
    
    def expect_return_type(self, func_name: str, expected_type: Union[Type, List[Type]]):
        """
        Set expected return type for a function.
        
        Args:
            func_name: Function name
            expected_type: Expected type or list of acceptable types
        """
        if not isinstance(expected_type, list):
            expected_type = [expected_type]
        
        self.expected_types[func_name] = expected_type
        print(f"✓ Expecting {func_name} to return: {[t.__name__ for t in expected_type]}")
    
    def expect_non_null(self, func_name: str):
        """
        Mark function as should never return None.
        
        Args:
            func_name: Function name
        """
        self.non_null_functions.add(func_name)
        print(f"✓ {func_name} should never return None")
    
    def expect_value_range(self, func_name: str, min_val: float = None, max_val: float = None):
        """
        Set expected value range for numeric returns.
        
        Args:
            func_name: Function name
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
        """
        self.value_ranges[func_name] = {'min': min_val, 'max': max_val}
        print(f"✓ {func_name} values should be in range [{min_val}, {max_val}]")
    
    def validate_result(self, func_name: str, result: Any) -> bool:
        """
        Validate function result for silent failures.
        
        Args:
            func_name: Function that returned the result
            result: The return value to validate
        
        Returns:
            True if valid, False if suspicious
        """
        issues_found = []
        
        # Check 1: Non-null validation
        if func_name in self.non_null_functions:
            if result is None:
                issues_found.append("Returned None (expected non-null)")
        
        # Check 2: Type validation
        if func_name in self.expected_types:
            expected_types = self.expected_types[func_name]
            if not any(isinstance(result, t) for t in expected_types):
                actual_type = type(result).__name__
                expected_names = [t.__name__ for t in expected_types]
                issues_found.append(
                    f"Wrong type: got {actual_type}, expected {expected_names}"
                )
        
        # Check 3: Value range validation (for numeric types)
        if func_name in self.value_ranges:
            if isinstance(result, (int, float)):
                range_spec = self.value_ranges[func_name]
                if range_spec['min'] is not None and result < range_spec['min']:
                    issues_found.append(
                        f"Value {result} below minimum {range_spec['min']}"
                    )
                if range_spec['max'] is not None and result > range_spec['max']:
                    issues_found.append(
                        f"Value {result} above maximum {range_spec['max']}"
                    )
        
        # Log issues if found
        if issues_found:
            for issue in issues_found:
                self.monitor.log_warning(
                    func_name,
                    f"Silent failure detected: {issue}"
                )
            
            # Track consecutive failures
            if func_name not in self.consecutive_failures:
                self.consecutive_failures[func_name] = 0
            self.consecutive_failures[func_name] += 1
            
            # Alert if many consecutive failures
            if self.consecutive_failures[func_name] >= 5:
                self.monitor.log_warning(
                    func_name,
                    f"⚠️  CRITICAL: {self.consecutive_failures[func_name]} consecutive validation failures!"
                )
            
            return False
        
        else:
            # Reset consecutive failure counter on success
            self.consecutive_failures[func_name] = 0
            return True
    
    def create_validated_wrapper(self, func: Callable, func_name: str) -> Callable:
        """
        Create a wrapper that validates results automatically.
        
        Args:
            func: Function to wrap
            func_name: Function name for validation
        
        Returns:
            Wrapped function with validation
        """
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.validate_result(func_name, result)
            return result
        
        return wrapper
    
    def detect_performance_anomaly(self, func_name: str, elapsed_time: float) -> bool:
        """
        Detect if function is running slower than baseline (may indicate issues).
        
        Args:
            func_name: Function name
            elapsed_time: Time taken in seconds
        
        Returns:
            True if anomaly detected
        """
        # Build baseline from function stats
        if func_name in self.monitor.function_stats:
            stats = self.monitor.function_stats[func_name]
            
            if stats['call_count'] >= 10:  # Need enough data
                avg_time = stats['avg_time']
                
                # If current time is 3x average, that's suspicious
                if elapsed_time > avg_time * 3:
                    self.monitor.log_warning(
                        func_name,
                        f"Performance anomaly: {elapsed_time*1000:.1f}ms (avg: {avg_time*1000:.1f}ms, 3x slower!)"
                    )
                    return True
        
        return False
    
    def check_function_health(self, func_name: str) -> dict:
        """
        Get comprehensive health check including silent failures.
        
        Args:
            func_name: Function to check
        
        Returns:
            Health report dict
        """
        health = self.monitor.get_function_health(func_name)
        
        # Add validation info
        health['validation'] = {
            'has_type_check': func_name in self.expected_types,
            'requires_non_null': func_name in self.non_null_functions,
            'has_range_check': func_name in self.value_ranges,
            'consecutive_failures': self.consecutive_failures.get(func_name, 0)
        }
        
        # Adjust health score for silent failures
        consecutive = self.consecutive_failures.get(func_name, 0)
        if consecutive > 0:
            health['health_score'] -= consecutive * 5
            health['health_score'] = max(0, health['health_score'])
            
            if consecutive >= 5:
                health['status'] = 'critical'
        
        return health


def setup_scanner_validation(monitor) -> ErrorValidator:
    """
    Set up validation rules for scanner functions.
    
    Args:
        monitor: QualityMonitor instance
    
    Returns:
        Configured ErrorValidator
    """
    print("\n" + "="*80)
    print("🔍 SETTING UP ENHANCED ERROR DETECTION")
    print("="*80)
    
    validator = ErrorValidator(monitor)
    
    # Detection functions should return tuple (x, y, radius) or None
    validator.expect_return_type('detect_red_laser_dot', [tuple, type(None)])
    
    # Distance estimation should return float
    validator.expect_return_type('estimate_distance_linear', [float, int])
    validator.expect_value_range('estimate_distance_linear', min_val=0, max_val=10000)  # cm
    
    # Save functions should return True or raise exception
    validator.expect_return_type('save_point_cloud', bool)
    validator.expect_non_null('save_point_cloud')
    
    # System check should return bool
    validator.expect_return_type('check_system_requirements', bool)
    validator.expect_non_null('check_system_requirements')
    
    # Auto-capture should return dict with results
    validator.expect_return_type('auto_capture_3_points', [dict, type(None)])
    
    print("\n✓ Validation rules configured")
    print("  Functions will be checked for:")
    print("    - Type correctness")
    print("    - Null returns (when shouldn't be)")
    print("    - Value ranges")
    print("    - Performance anomalies")
    
    return validator


# Example usage
if __name__ == "__main__":
    print(__doc__)
    
    from runtime_quality_monitor import QualityMonitor
    
    # Create monitor
    monitor = QualityMonitor()
    
    # Setup validation
    validator = setup_scanner_validation(monitor)
    
    # Example 1: Valid result
    print("\n" + "="*80)
    print("Example 1: Valid Result")
    print("="*80)
    
    result = (320, 240, 15)  # Valid laser dot detection
    is_valid = validator.validate_result('detect_red_laser_dot', result)
    print(f"Result {result}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Example 2: Invalid result (None when shouldn't be)
    print("\n" + "="*80)
    print("Example 2: Invalid Result - Silent Failure")
    print("="*80)
    
    result = None
    is_valid = validator.validate_result('save_point_cloud', result)
    print(f"Result {result}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Example 3: Wrong type
    print("\n" + "="*80)
    print("Example 3: Wrong Type")
    print("="*80)
    
    result = "error string"  # Should be bool!
    is_valid = validator.validate_result('check_system_requirements', result)
    print(f"Result {result}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Example 4: Out of range
    print("\n" + "="*80)
    print("Example 4: Out of Range Value")
    print("="*80)
    
    result = -50  # Negative distance doesn't make sense!
    is_valid = validator.validate_result('estimate_distance_linear', result)
    print(f"Result {result}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    print("\n✓ Enhanced error detection ready!")
    print("  Catches exceptions AND silent failures!")
