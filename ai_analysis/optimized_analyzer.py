"""
Optimized AI analyzer with frame skipping and caching.
"""
from camera_info import get_camera_info
from image_quality import analyze_image_quality
from results_storage import save_analysis_result

class OptimizedAIAnalyzer:
    """Optimized AI analysis with configurable frame skipping."""
    
    def __init__(self, analysis_interval=5, cache_camera_info=True):
        """
        Initialize optimizer.
        
        Args:
            analysis_interval: Analyze every N frames (default: 5)
            cache_camera_info: Cache camera info (doesn't change per frame)
        """
        self.analysis_interval = analysis_interval
        self.cache_camera_info = cache_camera_info
        self.frame_count = 0
        
        # Cached data
        self.cached_camera_info = None
        self.last_quality_result = None
    
    def should_analyze(self):
        """Check if current frame should be analyzed."""
        return self.frame_count % self.analysis_interval == 0
    
    def analyze_frame(self, frame, camera_matrix, dist_coeffs, cap):
        """
        Analyze frame with optimization.
        
        Args:
            frame: Video frame
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            cap: VideoCapture object
        
        Returns:
            dict: Analysis results (or cached if skipped)
        """
        self.frame_count += 1
        
        # Always get camera info (lightweight operation)
        if self.cache_camera_info and self.cached_camera_info is None:
            # Cache camera info on first frame
            self.cached_camera_info = get_camera_info(frame, camera_matrix, cap)
        
        # Only run heavy analysis every N frames
        if self.should_analyze():
            quality_result = analyze_image_quality(frame)
            self.last_quality_result = quality_result
            
            # Combine with cached camera info
            if self.cache_camera_info:
                ai_result = {**self.cached_camera_info, **quality_result}
            else:
                camera_info = get_camera_info(frame, camera_matrix, cap)
                ai_result = {**camera_info, **quality_result}
            
            # Save to storage
            save_analysis_result(ai_result, self.frame_count)
            
            return ai_result
        else:
            # Return last result for display (no heavy computation)
            if self.last_quality_result:
                if self.cache_camera_info:
                    return {**self.cached_camera_info, **self.last_quality_result}
                else:
                    return {**get_camera_info(frame, camera_matrix, cap), **self.last_quality_result}
            return {}
    
    def reset(self):
        """Reset analyzer state."""
        self.frame_count = 0
        self.cached_camera_info = None
        self.last_quality_result = None