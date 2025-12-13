"""
Camera information extraction module.
"""
import cv2
import time

# Global FPS tracker
_last_time = None
_fps_history = []

def get_camera_info(frame, camera_matrix, cap):
    """
    Extract camera information.
    
    Args:
        frame: Input video frame
        camera_matrix: Camera calibration matrix
        cap: VideoCapture object
    
    Returns:
        dict: Camera information (resolution, focal length, FPS, etc.)
    """
    global _last_time, _fps_history
    
    # Calculate FPS
    current_time = time.time()
    if _last_time is not None:
        fps = 1.0 / (current_time - _last_time)
        _fps_history.append(fps)
        
        # Keep only last 30 frames for smoothing
        if len(_fps_history) > 30:
            _fps_history.pop(0)
        
        # Use average of recent FPS values
        avg_fps = sum(_fps_history) / len(_fps_history)
    else:
        avg_fps = 0.0
    
    _last_time = current_time
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Extract focal length from camera matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Calculate average focal length in pixels
    focal_length_px = (fx + fy) / 2.0
    
    # Format focal length for display
    focal_length_str = f"{focal_length_px:.1f}px"
    
    return {
        "fps": avg_fps,
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "focal_length": focal_length_str,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "frame_count": -1  # This will be updated by the analyzer
    }