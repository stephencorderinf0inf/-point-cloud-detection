#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quaternion 4D Rotation Module
Provides quaternion-based rotations for point cloud mesh reconstruction.
Ensures full 360° coverage without gimbal lock or overlap.
"""

import numpy as np
from typing import Tuple, List, Optional, Union


class QuaternionRotation:
    """
    Quaternion rotation utilities for 4D transformations.
    Quaternions live on 4D unit hypersphere and enable smooth rotations.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize quaternion rotation handler.
        
        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
    
    def normalize(self, q: np.ndarray) -> np.ndarray:
        """
        Normalize quaternion to unit length (4D hypersphere).
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Normalized quaternion
        """
        norm = np.linalg.norm(q)
        if norm < self.tolerance:
            # Return identity quaternion if near zero
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm
    
    def from_axis_angle(self, axis: np.ndarray, degrees: float) -> np.ndarray:
        """
        Create quaternion from axis-angle representation.
        
        Args:
            axis: 3D rotation axis [x, y, z] (will be normalized)
            degrees: Rotation angle in degrees
            
        Returns:
            Unit quaternion [w, x, y, z]
        """
        axis = np.array(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < self.tolerance:
            # No rotation for zero axis
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        axis = axis / axis_norm
        theta = np.radians(degrees)
        
        w = np.cos(theta / 2.0)
        xyz = axis * np.sin(theta / 2.0)
        
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    def from_euler(self, x_deg: float, y_deg: float, z_deg: float, 
                   order: str = 'XYZ') -> np.ndarray:
        """
        Create quaternion from Euler angles.
        
        Args:
            x_deg: Rotation around X-axis in degrees
            y_deg: Rotation around Y-axis in degrees
            z_deg: Rotation around Z-axis in degrees
            order: Order of rotations ('XYZ', 'ZYX', etc.)
            
        Returns:
            Combined quaternion
        """
        q_x = self.from_axis_angle([1, 0, 0], x_deg)
        q_y = self.from_axis_angle([0, 1, 0], y_deg)
        q_z = self.from_axis_angle([0, 0, 1], z_deg)
        
        if order == 'XYZ':
            return self.multiply(self.multiply(q_z, q_y), q_x)
        elif order == 'ZYX':
            return self.multiply(self.multiply(q_x, q_y), q_z)
        elif order == 'YXZ':
            return self.multiply(self.multiply(q_z, q_x), q_y)
        else:
            raise ValueError(f"Unsupported rotation order: {order}")
    
    def multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions (Hamilton product).
        Order matters: q1 * q2 != q2 * q1
        
        Args:
            q1, q2: Quaternions [w, x, y, z]
            
        Returns:
            Product quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])
    
    def conjugate(self, q: np.ndarray) -> np.ndarray:
        """
        Get quaternion conjugate.
        For unit quaternions, conjugate = inverse.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Conjugate quaternion [w, -x, -y, -z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def inverse(self, q: np.ndarray) -> np.ndarray:
        """
        Get quaternion inverse.
        For unit quaternions, this is the same as conjugate.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            Inverse quaternion
        """
        norm_sq = np.dot(q, q)
        if norm_sq < self.tolerance:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        conj = self.conjugate(q)
        return conj / norm_sq
    
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
        
        # Convert vector to pure quaternion [0, x, y, z]
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        
        # Perform rotation: q * v * q^-1
        q_inv = self.conjugate(q)
        result = self.multiply(self.multiply(q, v_quat), q_inv)
        
        # Extract vector part
        return result[1:]
    
    def rotate_point_cloud(self, q: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Rotate multiple 3D points using quaternion.
        
        Args:
            q: Quaternion [w, x, y, z]
            points: Nx3 array of 3D points
            
        Returns:
            Nx3 array of rotated points
        """
        rotated = np.array([self.rotate_vector(q, p) for p in points])
        return rotated
    
    def slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two quaternions.
        Provides smooth rotation interpolation.
        
        Args:
            q1, q2: Start and end quaternions
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated quaternion
        """
        q1 = self.normalize(q1)
        q2 = self.normalize(q2)
        
        # Compute dot product
        dot = np.dot(q1, q2)
        
        # If dot < 0, negate q2 to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return self.normalize(result)
        
        # Calculate interpolation coefficients
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        q3 = self.normalize(q2 - q1 * dot)
        
        return q1 * np.cos(theta) + q3 * np.sin(theta)
    
    def to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        q = self.normalize(q)
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def sweep_360(self, points: np.ndarray, axis: np.ndarray, 
                  steps: int = 36) -> np.ndarray:
        """
        Sweep point cloud through full 360° rotation.
        Returns all rotated positions (original points × steps).
        
        Args:
            points: Nx3 array of original points
            axis: Rotation axis [x, y, z]
            steps: Number of rotation increments (36 = 10° steps)
            
        Returns:
            (N×steps)×3 array of all rotated points
        """
        all_points = []
        
        for i in range(steps):
            angle = 360.0 * i / steps
            q = self.from_axis_angle(axis, angle)
            rotated = self.rotate_point_cloud(q, points)
            all_points.append(rotated)
        
        return np.vstack(all_points)
    
    def sweep_360_clouds(self, points: np.ndarray, axis: np.ndarray,
                         steps: int = 36) -> List[np.ndarray]:
        """
        Sweep point cloud through 360° rotation, returning each step separately.
        Useful for sequential processing or visualization.
        
        Args:
            points: Nx3 array of original points
            axis: Rotation axis [x, y, z]
            steps: Number of rotation increments
            
        Returns:
            List of Nx3 arrays, one for each rotation step
        """
        clouds = []
        
        for i in range(steps):
            angle = 360.0 * i / steps
            q = self.from_axis_angle(axis, angle)
            rotated = self.rotate_point_cloud(q, points)
            clouds.append(rotated)
        
        return clouds


# Convenience functions for direct use
def create_quaternion(axis: Union[List, np.ndarray], degrees: float) -> np.ndarray:
    """Create quaternion from axis and angle."""
    qr = QuaternionRotation()
    return qr.from_axis_angle(np.array(axis), degrees)


def rotate_points(points: np.ndarray, axis: Union[List, np.ndarray], 
                  degrees: float) -> np.ndarray:
    """Rotate points using quaternion rotation."""
    qr = QuaternionRotation()
    q = qr.from_axis_angle(np.array(axis), degrees)
    return qr.rotate_point_cloud(q, points)


def sweep_point_cloud(points: np.ndarray, axis: Union[List, np.ndarray],
                      steps: int = 36) -> np.ndarray:
    """Sweep point cloud through 360° rotation."""
    qr = QuaternionRotation()
    return qr.sweep_360(points, np.array(axis), steps)
