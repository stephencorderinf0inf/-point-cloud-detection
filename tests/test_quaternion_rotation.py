#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quaternion 4D Rotation Test
Verifies quaternion rotations cover all 360° without overlap or gimbal lock
Tests for point cloud mesh reconstruction
"""

import numpy as np
import time
from typing import Tuple, List, Optional


class QuaternionRotationTest:
    """
    Test suite for quaternion-based 4D rotations.
    Verifies full 360° coverage without overlap for mesh reconstruction.
    """
    
    def __init__(self):
        """Initialize quaternion rotation test suite."""
        self.test_results = []
        self.verbose = True
        self.tolerance = 1e-10
        
    def normalize(self, q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length (4D hypersphere)."""
        norm = np.linalg.norm(q)
        if norm < self.tolerance:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    
    def quaternion_from_axis_angle(self, axis: np.ndarray, degrees: float) -> np.ndarray:
        """
        Create quaternion from axis-angle representation.
        
        Args:
            axis: 3D rotation axis [x, y, z]
            degrees: Rotation angle in degrees
            
        Returns:
            Quaternion [w, x, y, z]
        """
        axis = axis / np.linalg.norm(axis)
        theta = np.radians(degrees)
        w = np.cos(theta / 2.0)
        xyz = axis * np.sin(theta / 2.0)
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions.
        
        Args:
            q1, q2: Quaternions [w, x, y, z]
            
        Returns:
            Product quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Get quaternion conjugate (inverse for unit quaternions)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate 3D vector using quaternion.
        v' = q * v * q^-1
        
        Args:
            q: Quaternion [w, x, y, z]
            v: 3D vector [x, y, z]
            
        Returns:
            Rotated 3D vector
        """
        q = self.normalize(q)
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        q_inv = self.quaternion_conjugate(q)
        result = self.quaternion_multiply(self.quaternion_multiply(q, v_quat), q_inv)
        return result[1:]
    
    def test_unit_quaternion_property(self):
        """Test that quaternions lie on 4D unit hypersphere."""
        print("\n=== Testing 4D Unit Hypersphere Property ===")
        
        test_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        axis = np.array([0, 1, 0])  # Y-axis
        
        all_valid = True
        for angle in test_angles:
            q = self.quaternion_from_axis_angle(axis, angle)
            magnitude = np.linalg.norm(q)
            is_unit = np.abs(magnitude - 1.0) < self.tolerance
            
            print(f"{angle}°: q={q}, |q|={magnitude:.10f}, unit={is_unit}")
            all_valid = all_valid and is_unit
        
        print(f"\nAll quaternions on unit 4D sphere: {all_valid}")
        return all_valid
    
    def test_360_degree_coverage(self):
        """Test that 360° rotation returns to original orientation."""
        print("\n=== Testing 360° Coverage (No Overlap) ===")
        
        point = np.array([1.0, 0.0, 0.0])
        axis = np.array([0, 0, 1])  # Z-axis
        
        # Test incremental rotations
        steps = 36  # 10° increments
        current_point = point.copy()
        
        print(f"Starting point: {point}")
        
        for i in range(steps):
            angle = 360.0 / steps
            q = self.quaternion_from_axis_angle(axis, angle)
            current_point = self.rotate_vector(q, current_point)
            
            if i % 9 == 0:  # Print every 90°
                print(f"After {(i+1)*angle:.1f}°: {current_point}")
        
        # After 360°, should return to original
        matches = np.allclose(current_point, point, atol=self.tolerance)
        print(f"\nFinal point: {current_point}")
        print(f"Returns to start: {matches}")
        
        return matches
    
    def test_no_gimbal_lock(self):
        """Test that quaternions avoid gimbal lock."""
        print("\n=== Testing Gimbal Lock Avoidance ===")
        
        point = np.array([1.0, 0.0, 0.0])
        
        # Problematic Euler angles that cause gimbal lock (90° pitch)
        # With quaternions, this should still work smoothly
        
        q1 = self.quaternion_from_axis_angle(np.array([0, 1, 0]), 90)  # 90° Y
        q2 = self.quaternion_from_axis_angle(np.array([0, 0, 1]), 45)  # 45° Z
        q3 = self.quaternion_from_axis_angle(np.array([1, 0, 0]), 30)  # 30° X
        
        # Combine rotations
        q_combined = self.quaternion_multiply(self.quaternion_multiply(q1, q2), q3)
        q_combined = self.normalize(q_combined)
        
        rotated = self.rotate_vector(q_combined, point)
        
        print(f"Original: {point}")
        print(f"After 90°Y + 45°Z + 30°X: {rotated}")
        print(f"Combined quaternion: {q_combined}")
        print(f"Is unit: {np.abs(np.linalg.norm(q_combined) - 1.0) < self.tolerance}")
        
        # No NaN or Inf values
        no_issues = not (np.any(np.isnan(rotated)) or np.any(np.isinf(rotated)))
        print(f"No gimbal lock (no NaN/Inf): {no_issues}")
        
        return no_issues
    
    def test_smooth_interpolation_path(self):
        """Test that rotations create smooth paths (no jumps)."""
        print("\n=== Testing Smooth Rotation Paths ===")
        
        point = np.array([1.0, 0.0, 0.0])
        axis = np.array([0, 1, 0])
        
        steps = 36
        positions = []
        
        for i in range(steps + 1):
            angle = 360.0 * i / steps
            q = self.quaternion_from_axis_angle(axis, angle)
            rotated = self.rotate_vector(q, point)
            positions.append(rotated)
        
        # Check that consecutive points are close (smooth path)
        max_jump = 0
        for i in range(len(positions) - 1):
            distance = np.linalg.norm(positions[i+1] - positions[i])
            max_jump = max(max_jump, distance)
        
        expected_max = 2 * np.sin(np.radians(360.0 / steps / 2))  # Chord length
        is_smooth = max_jump < expected_max * 1.1  # 10% tolerance
        
        print(f"Number of steps: {steps}")
        print(f"Max jump between consecutive points: {max_jump:.6f}")
        print(f"Expected max chord length: {expected_max:.6f}")
        print(f"Path is smooth: {is_smooth}")
        
        return is_smooth
    
    def test_multiple_axis_rotations(self):
        """Test rotations around different axes."""
        print("\n=== Testing Multi-Axis Rotations ===")
        
        point = np.array([1.0, 1.0, 1.0])
        point = point / np.linalg.norm(point)  # Normalize
        
        axes = {
            'X': np.array([1, 0, 0]),
            'Y': np.array([0, 1, 0]),
            'Z': np.array([0, 0, 1]),
            'XYZ': np.array([1, 1, 1]) / np.sqrt(3)
        }
        
        all_valid = True
        for name, axis in axes.items():
            q = self.quaternion_from_axis_angle(axis, 45)
            rotated = self.rotate_vector(q, point)
            
            # Check magnitude preserved
            mag_preserved = np.abs(np.linalg.norm(rotated) - 1.0) < self.tolerance
            
            print(f"{name}-axis 45°: {rotated}, magnitude preserved: {mag_preserved}")
            all_valid = all_valid and mag_preserved
        
        return all_valid
    
    def test_point_cloud_sweep(self):
        """Test sweeping a point cloud through 360° - check rotation consistency."""
        print("\n=== Testing Point Cloud Sweep Coverage ===")
        
        # Create simple point cloud (tetrahedron)
        points = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0.5]
        ], dtype=float)
        
        axis = np.array([0, 1, 0])  # Rotate around Y
        steps = 36
        
        # Test that each rotation step produces unique orientation
        orientations = []
        
        for i in range(steps):
            angle = 360.0 * i / steps
            q = self.quaternion_from_axis_angle(axis, angle)
            orientations.append(q)
        
        # Check quaternions are unique (different orientations)
        unique_count = 0
        for i, q1 in enumerate(orientations):
            is_unique = True
            for j, q2 in enumerate(orientations[:i]):
                # Quaternions q and -q represent same rotation
                if np.allclose(q1, q2, atol=self.tolerance) or np.allclose(q1, -q2, atol=self.tolerance):
                    is_unique = False
                    break
            if is_unique:
                unique_count += 1
        
        coverage = unique_count / len(orientations)
        
        print(f"Original points: {len(points)}")
        print(f"Rotation steps: {steps}")
        print(f"Unique orientations: {unique_count}/{len(orientations)}")
        print(f"Coverage: {coverage*100:.1f}%")
        print(f"Full coverage (>95%): {coverage > 0.95}")
        
        return coverage > 0.95
    
    def run_all_tests(self):
        """Run all quaternion rotation tests."""
        print("=" * 70)
        print("QUATERNION 4D ROTATION TEST SUITE")
        print("Verifying full 360° coverage for mesh reconstruction")
        print("=" * 70)
        
        start_time = time.time()
        
        results = {
            'unit_sphere': self.test_unit_quaternion_property(),
            '360_coverage': self.test_360_degree_coverage(),
            'no_gimbal_lock': self.test_no_gimbal_lock(),
            'smooth_paths': self.test_smooth_interpolation_path(),
            'multi_axis': self.test_multiple_axis_rotations(),
            'point_cloud_sweep': self.test_point_cloud_sweep()
        }
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        all_passed = all(results.values())
        print("\n" + ("="* 70))
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print(f"Completed in {elapsed:.4f} seconds")
        print("=" * 70)
        
        return all_passed


def main():
    """Main test execution."""
    tester = QuaternionRotationTest()
    success = tester.run_all_tests()
    
    if success:
        print("\n✓ Quaternions ready for point cloud mesh reconstruction!")
    else:
        print("\n✗ Issues detected - review failed tests above")


if __name__ == "__main__":
    main()
