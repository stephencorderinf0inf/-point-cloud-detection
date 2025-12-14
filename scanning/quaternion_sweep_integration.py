#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quaternion Point Cloud Sweep Integration
Adds 360° rotation sweep capability to scanner for full mesh reconstruction
"""

import numpy as np
import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent / 'utils'
sys.path.insert(0, str(utils_path))

from quaternion_rotation import QuaternionRotation, sweep_point_cloud


class PointCloudSweepIntegration:
    """
    Integrates quaternion-based 360° sweeps with existing scanner.
    Prevents mesh self-overlap by covering all rotational degrees.
    """
    
    def __init__(self, steps: int = 36):
        """
        Initialize sweep integration.
        
        Args:
            steps: Number of rotation steps for 360° (36 = 10° increments)
        """
        self.qr = QuaternionRotation()
        self.steps = steps
        self.sweep_enabled = False
        self.sweep_axis = np.array([0, 1, 0])  # Default: Y-axis (vertical)
        
    def enable_sweep(self, axis: str = 'Y'):
        """
        Enable 360° sweep mode.
        
        Args:
            axis: Rotation axis ('X', 'Y', or 'Z')
        """
        self.sweep_enabled = True
        
        if axis.upper() == 'X':
            self.sweep_axis = np.array([1, 0, 0])
        elif axis.upper() == 'Y':
            self.sweep_axis = np.array([0, 1, 0])
        elif axis.upper() == 'Z':
            self.sweep_axis = np.array([0, 0, 1])
        else:
            raise ValueError(f"Invalid axis: {axis}. Use 'X', 'Y', or 'Z'")
        
        print(f"✅ 360° sweep enabled: {self.steps} steps around {axis}-axis")
    
    def disable_sweep(self):
        """Disable sweep mode (normal single-view scanning)."""
        self.sweep_enabled = False
        print("Sweep mode disabled")
    
    def process_point_cloud(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Process captured point cloud.
        If sweep enabled, rotates through 360° and merges.
        
        Args:
            points_3d: Nx3 array of captured 3D points
            
        Returns:
            Processed point cloud (swept if enabled, original otherwise)
        """
        if not self.sweep_enabled:
            return points_3d
        
        if len(points_3d) == 0:
            return points_3d
        
        print(f"🔄 Sweeping {len(points_3d)} points through 360°...")
        swept_points = self.qr.sweep_360(points_3d, self.sweep_axis, self.steps)
        print(f"✅ Sweep complete: {len(swept_points)} total points")
        
        return swept_points
    
    def get_incremental_clouds(self, points_3d: np.ndarray) -> list:
        """
        Get individual point clouds for each rotation step.
        Useful for progressive reconstruction or visualization.
        
        Args:
            points_3d: Nx3 array of original points
            
        Returns:
            List of Nx3 arrays for each rotation
        """
        if len(points_3d) == 0:
            return [points_3d]
        
        return self.qr.sweep_360_clouds(points_3d, self.sweep_axis, self.steps)
    
    def get_rotation_quaternion(self, step: int) -> np.ndarray:
        """
        Get quaternion for specific rotation step.
        
        Args:
            step: Rotation step index (0 to steps-1)
            
        Returns:
            Quaternion [w, x, y, z]
        """
        angle = 360.0 * step / self.steps
        return self.qr.from_axis_angle(self.sweep_axis, angle)
    
    def rotate_to_step(self, points_3d: np.ndarray, step: int) -> np.ndarray:
        """
        Rotate points to specific step angle.
        
        Args:
            points_3d: Nx3 array of points
            step: Rotation step index
            
        Returns:
            Rotated points
        """
        q = self.get_rotation_quaternion(step)
        return self.qr.rotate_point_cloud(q, points_3d)


# Integration hooks for laser_3d_scanner_advanced.py

def add_sweep_to_scanner_keys(sweep_integration):
    """
    Returns keyboard controls for sweep mode.
    Add these to the main scanner keyboard handling.
    """
    controls = {
        'r': ('Toggle 360° rotation sweep', lambda: toggle_sweep(sweep_integration)),
        'x': ('Set sweep axis to X', lambda: sweep_integration.enable_sweep('X')),
        'y': ('Set sweep axis to Y', lambda: sweep_integration.enable_sweep('Y')),
        'z': ('Set sweep axis to Z', lambda: sweep_integration.enable_sweep('Z')),
        '[': ('Decrease sweep steps', lambda: adjust_steps(sweep_integration, -6)),
        ']': ('Increase sweep steps', lambda: adjust_steps(sweep_integration, 6)),
    }
    return controls


def toggle_sweep(sweep_integration):
    """Toggle sweep mode on/off."""
    if sweep_integration.sweep_enabled:
        sweep_integration.disable_sweep()
    else:
        sweep_integration.enable_sweep()


def adjust_steps(sweep_integration, delta):
    """Adjust number of sweep steps."""
    new_steps = max(12, min(72, sweep_integration.steps + delta))
    sweep_integration.steps = new_steps
    print(f"Sweep steps: {new_steps} ({360/new_steps:.1f}° increments)")


def integrate_with_save_point_cloud(points_3d, sweep_integration, **kwargs):
    """
    Wrapper for save_point_cloud that includes sweep processing.
    
    Usage in laser_3d_scanner_advanced.py:
        # Before saving
        points_3d = sweep_integration.process_point_cloud(points_3d)
        save_point_cloud(points_3d, **kwargs)
    """
    processed_points = sweep_integration.process_point_cloud(points_3d)
    return processed_points


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("QUATERNION SWEEP INTEGRATION TEST")
    print("=" * 70)
    
    # Create test points (simple cube)
    test_points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=float)
    
    # Test sweep
    sweep = PointCloudSweepIntegration(steps=36)
    sweep.enable_sweep('Y')
    
    swept = sweep.process_point_cloud(test_points)
    
    print(f"\nOriginal points: {len(test_points)}")
    print(f"After 360° sweep: {len(swept)} points")
    print(f"Rotation steps: {sweep.steps}")
    print(f"Degrees per step: {360/sweep.steps:.1f}°")
    
    print("\n✅ Integration ready for laser_3d_scanner_advanced.py")
