# Add this to laser_3d_scanner_advanced.py

import torch
import cv2
import numpy as np
from pathlib import Path

class DepthEstimator:
    """Estimate depth from a single RGB image using MiDaS."""
    
    def __init__(self):
        """Load MiDaS depth estimation model."""
        print("\n" + "=" * 70)
        print("📊 LOADING DEPTH ESTIMATION MODEL")
        print("=" * 70)
        
        # Load MiDaS model (fast but less accurate)
        self.model_type = "DPT_Large"  # or "MiDaS_small" for speed
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        print(f"✓ Model loaded: {self.model_type}")
        print(f"✓ Device: {self.device}")
        print("=" * 70)
    
    def estimate_depth(self, rgb_image):
        """
        Estimate depth map from RGB image.
        
        Args:
            rgb_image: BGR image from camera (H, W, 3)
        
        Returns:
            depth_map: Normalized depth (H, W) - closer = smaller values
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Transform for model input
        input_batch = self.transform(rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-1 (inverse depth - closer = smaller values)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map
    
    def depth_to_point_cloud(self, rgb_image, depth_map, camera_matrix, 
                            max_depth_m=2.0, min_depth_m=0.2):
        """
        Convert depth map + RGB to 3D point cloud.
        
        Args:
            rgb_image: BGR image (H, W, 3)
            depth_map: Normalized depth (H, W)
            camera_matrix: Camera intrinsics K (3, 3)
            max_depth_m: Maximum depth in meters
            min_depth_m: Minimum depth in meters
        
        Returns:
            points_3d: (N, 3) array of XYZ coordinates
            colors: (N, 3) array of RGB colors
        """
        h, w = depth_map.shape
        
        # Convert normalized depth to real-world depth (meters)
        depth_real = min_depth_m + (max_depth_m - min_depth_m) * (1.0 - depth_map)
        
        # Get camera parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert to 3D (pinhole camera model)
        z = depth_real * 1000  # Convert to mm
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack into point cloud
        points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Get colors (convert BGR to RGB)
        colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        
        # Filter out invalid points
        valid_mask = (z > min_depth_m * 1000) & (z < max_depth_m * 1000)
        valid_mask = valid_mask.reshape(-1)
        
        return points_3d[valid_mask], colors[valid_mask]


# ADD THIS TO YOUR SCANNER'S MAIN LOOP (around line 1400):

def scan_3d_points(project_dir=None):
    """Main 3D scanning function with depth estimation."""
    
    # ... existing initialization ...
    
    # 🎨 NEW: Initialize depth estimator
    use_depth_estimation = True  # Toggle this
    depth_estimator = None
    
    if use_depth_estimation:
        try:
            depth_estimator = DepthEstimator()
        except Exception as e:
            print(f"⚠️  Could not load depth estimator: {e}")
            print("   Falling back to laser-only mode")
            use_depth_estimation = False
    
    # ... existing camera loop ...
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ... existing processing ...
        
        # 🎨 NEW: Depth estimation mode
        if key == ord('4'):  # Press '4' for depth estimation
            print("\n[CAPTURE] Estimating depth for full frame...")
            
            if depth_estimator is None:
                print("❌ Depth estimator not loaded!")
                continue
            
            # Estimate depth
            depth_map = depth_estimator.estimate_depth(frame)
            
            # Convert to point cloud
            new_points, new_colors = depth_estimator.depth_to_point_cloud(
                frame, 
                depth_map, 
                new_camera_matrix,
                max_depth_m=2.0,  # Adjust based on your scene
                min_depth_m=0.2
            )
            
            # Add to point cloud
            count_before = len(points_3d)
            points_3d.extend(new_points.tolist())
            points_colors.extend(new_colors.tolist())
            
            # Store metadata
            for _ in range(len(new_points)):
                point_angles.append(current_angle)
                point_sessions.append(current_session)
            
            count_added = len(points_3d) - count_before
            print(f"✓ Added {count_added:,} points from depth estimation")
            
            # Show depth map
            depth_vis = (depth_map * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
            cv2.imshow("Depth Map", depth_colored)