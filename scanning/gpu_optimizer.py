"""
GPU acceleration utilities for 3D scanner.
"""
import cv2
import numpy as np

class GPUOptimizer:
    """GPU acceleration manager."""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_support()
        self.use_gpu = self.gpu_available
        
        # Cached GPU matrices
        self.gpu_camera_matrix = None
        self.gpu_dist_coeffs = None
        self.gpu_map1 = None
        self.gpu_map2 = None
    
    def _check_gpu_support(self):
        """Check if OpenCV was built with CUDA support."""
        try:
            # Check if CUDA is available
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                print(f"[GPU] Found {cuda_devices} CUDA device(s)")
                return True
            else:
                print("[GPU] No CUDA devices found, using CPU")
                return False
        except:
            print("[GPU] OpenCV not built with CUDA support, using CPU")
            return False
    
    def initialize_undistortion_maps(self, camera_matrix, dist_coeffs, image_shape):
        """
        Pre-compute undistortion maps (CPU or GPU).
        
        Args:
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            image_shape: (height, width) tuple
        """
        h, w = image_shape[:2]
        
        # Pre-compute remap matrices (done once, massive speedup!)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, 
            (w, h), cv2.CV_16SC2
        )
        
        print(f"[OPTIMIZE] Undistortion maps pre-computed for {w}x{h}")
        
        if self.use_gpu:
            try:
                # Upload to GPU
                self.gpu_map1 = cv2.cuda_GpuMat()
                self.gpu_map2 = cv2.cuda_GpuMat()
                self.gpu_map1.upload(self.map1)
                self.gpu_map2.upload(self.map2)
                print("[GPU] Undistortion maps uploaded to GPU")
            except Exception as e:
                print(f"[GPU] Failed to upload maps: {e}")
                self.use_gpu = False
    
    def undistort_frame(self, frame):
        """
        Undistort frame using pre-computed maps (CPU or GPU).
        
        Args:
            frame: Input frame
        
        Returns:
            Undistorted frame
        """
        if self.map1 is None or self.map2 is None:
            return frame  # Maps not initialized
        
        if self.use_gpu and self.gpu_map1 is not None:
            try:
                # GPU remap
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_result = cv2.cuda.remap(gpu_frame, self.gpu_map1, self.gpu_map2, cv2.INTER_LINEAR)
                return gpu_result.download()
            except Exception as e:
                print(f"[GPU] Remap failed, falling back to CPU: {e}")
                self.use_gpu = False
        
        # CPU remap (still much faster than cv2.undistort!)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
    
    def get_status(self):
        """Get GPU status string."""
        if self.use_gpu:
            return "GPU (CUDA)"
        elif self.gpu_available:
            return "GPU Available (Not Used)"
        else:
            return "CPU Only"