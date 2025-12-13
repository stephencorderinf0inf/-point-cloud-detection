#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Modal Capture Module
Combines curve and corner detection into unified point cloud capture
(No laser - that's handled separately in dedicated laser mode)
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class MultiModalCapture:
    """
    Captures 3D points using curve + corner detection simultaneously.
    Laser detection is intentionally excluded (use dedicated laser mode for that).
    """
    
    def __init__(self):
        """Initialize multi-modal capture module."""
        self.name = "Multi-Modal Fusion Capture"
        self.description = "Curves + Corners in one capture"
    
    def capture_all_methods(self, 
                           undistorted: np.ndarray,
                           new_camera_matrix: np.ndarray,
                           curve_sample_rate: int = 5,
                           corner_max_count: int = 100,
                           canny_threshold1: int = 50,
                           canny_threshold2: int = 150,
                           current_angle: float = 0.0,
                           current_session: int = 0) -> Dict:
        """
        Capture points using curve + corner detection.
        
        Args:
            undistorted: Undistorted camera frame
            new_camera_matrix: Camera intrinsic matrix
            curve_sample_rate: Sample every Nth point on curves (1=all, 5=every 5th)
            corner_max_count: Maximum corners to capture
            canny_threshold1: Canny edge lower threshold
            canny_threshold2: Canny edge upper threshold
            current_angle: Current rotation angle (for metadata)
            current_session: Current capture session (for metadata)
        
        Returns:
            Dict with:
                - 'points_3d': List of [x, y, z] coordinates
                - 'colors': List of [r, g, b] colors
                - 'angles': List of angles
                - 'sessions': List of session IDs
                - 'method_counts': Dict with point counts per method
                - 'total_added': Total points added
        """
        from curve_detection import detect_curves
        from corner_detection import detect_corners
        from distance_estimation import estimate_distance_linear
        
        h, w = undistorted.shape[:2]
        
        # Storage for captured points
        points_3d = []
        points_colors = []
        point_angles = []
        point_sessions = []
        
        method_counts = {
            'curves': 0,
            'corners': 0
        }
        
        # Camera intrinsics
        fx = new_camera_matrix[0, 0]
        fy = new_camera_matrix[1, 1]
        cx = new_camera_matrix[0, 2]
        cy = new_camera_matrix[1, 2]
        
        # === METHOD 1: CURVE TRACING (Edge Details) ===
        try:
            curves, _ = detect_curves(undistorted, canny_threshold1, canny_threshold2)
            
            for curve in curves:
                # Sample points from curve
                for point in curve[::curve_sample_rate]:
                    px, py = point[0]
                    py = max(0, min(int(py), h-1))
                    px = max(0, min(int(px), w-1))
                    
                    distance_cm = estimate_distance_linear(py)
                    z = distance_cm * 10  # mm
                    x = (px - cx) * z / fx
                    y = (py - cy) * z / fy
                    points_3d.append([x, y, z])
                    
                    # Capture color at curve point
                    color_bgr = undistorted[py, px]
                    color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
                    points_colors.append(color_rgb)
                    
                    point_angles.append(current_angle)
                    point_sessions.append(current_session)
                    
                    method_counts['curves'] += 1
        except Exception as e:
            print(f"⚠️  Curve detection failed: {e}")
        
        # === METHOD 2: CORNER DETECTION (Feature Points) ===
        try:
            corner_points, _ = detect_corners(undistorted)
            
            for (px, py) in corner_points[:corner_max_count]:
                py = max(0, min(int(py), h-1))
                px = max(0, min(int(px), w-1))
                
                distance_cm = estimate_distance_linear(py)
                z = distance_cm * 10  # mm
                x = (px - cx) * z / fx
                y = (py - cy) * z / fy
                points_3d.append([x, y, z])
                
                # Capture color at corner
                color_bgr = undistorted[py, px]
                color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
                points_colors.append(color_rgb)
                
                point_angles.append(current_angle)
                point_sessions.append(current_session)
                
                method_counts['corners'] += 1
        except Exception as e:
            print(f"⚠️  Corner detection failed: {e}")
        
        # Return results
        return {
            'points_3d': points_3d,
            'colors': points_colors,
            'angles': point_angles,
            'sessions': point_sessions,
            'method_counts': method_counts,
            'total_added': len(points_3d)
        }
    
    def visualize_all_methods(self, 
                             undistorted: np.ndarray,
                             curve_sample_rate: int = 5,
                             corner_max_count: int = 100,
                             canny_threshold1: int = 50,
                             canny_threshold2: int = 150) -> np.ndarray:
        """
        Draw curve + corner overlays on frame (for preview).
        
        Args:
            undistorted: Frame to draw on
            curve_sample_rate: Curve sampling rate
            corner_max_count: Max corners to show
            canny_threshold1: Canny lower threshold
            canny_threshold2: Canny upper threshold
        
        Returns:
            Frame with detection overlays drawn
        """
        from curve_detection import detect_curves
        from corner_detection import detect_corners
        
        display_frame = undistorted.copy()
        h, w = display_frame.shape[:2]
        
        detection_counts = {'curves': 0, 'corners': 0}
        
        # Draw curves (yellow contours)
        try:
            curves, _ = detect_curves(undistorted, canny_threshold1, canny_threshold2)
            cv2.drawContours(display_frame, curves, -1, (0, 255, 255), 2)
            detection_counts['curves'] = len(curves)
        except:
            pass
        
        # Draw corners (pink dots)
        try:
            corner_points, _ = detect_corners(undistorted)
            for (px, py) in corner_points[:corner_max_count]:
                cv2.circle(display_frame, (int(px), int(py)), 4, (255, 0, 255), -1)
            detection_counts['corners'] = len(corner_points)
        except:
            pass
        
        # Draw info overlay
        cv2.putText(display_frame, "MULTI-MODAL MODE (Curves + Corners)", 
                   (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_text = f"Curves:{detection_counts['curves']} | Corners:{detection_counts['corners']}"
        cv2.putText(display_frame, info_text, 
                   (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.putText(display_frame, "Press 'm' to capture ALL", 
                   (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return display_frame


# Example usage
if __name__ == "__main__":
    print("""
    Multi-Modal Capture Module (Curves + Corners Only)
    
    Import this in laser_3d_scanner_advanced.py:
    
        from multi_modal_capture import MultiModalCapture
        
        # Create instance
        multi_capture = MultiModalCapture()
        
        # Capture points (curves + corners, NO laser)
        result = multi_capture.capture_all_methods(
            undistorted, new_camera_matrix,
            curve_sample_rate, corner_max_count,
            canny_threshold1, canny_threshold2,
            current_angle, current_session
        )
        
        # Add to point cloud
        points_3d.extend(result['points_3d'])
        points_colors.extend(result['colors'])
        
    Output:
        ✓ Multi-modal capture:
           Curves: 127 points
           Corners: 43 points
           TOTAL: 170 points added
    """)