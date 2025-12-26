"""
Monocular Depth Estimation Module
==================================
AI-powered depth estimation using Depth Anything V2 for dense 3D reconstruction.
Compatible with laser_3d_scanner_advanced.py

Usage:
    from depth_estimator import DepthEstimator
    
    estimator = DepthEstimator("vits")
    depth_map = estimator.estimate_depth(rgb_image)
    points, colors = estimator.depth_to_point_cloud(rgb_image, depth_map, K)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import time
import sys
import os

# ========== MSIX-COMPATIBLE PATHS ==========
# Find bundled depth_anything_v2 module relative to this file
MODULE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MODULE_DIR.parent.parent.parent  # Up to scripts/graphene root

# Try multiple possible locations for Depth Anything V2
DEPTH_PATHS = [
    MODULE_DIR / "depth_anything_v2",  # Bundled with scanning tools
    PROJECT_ROOT / "depth_anything_v2",  # In graphene root
    Path("D:/Users/Planet UI/world_folder/projects/Depth-Anything-V2"),  # Development path
]

DEPTH_ANYTHING_PATH = None
for path in DEPTH_PATHS:
    if path.exists():
        DEPTH_ANYTHING_PATH = path
        sys.path.insert(0, str(path.parent))  # Add parent to path
        break

# Import Depth Anything V2
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_V2_AVAILABLE = True
except ImportError:
    DepthAnythingV2 = None
    DEPTH_V2_AVAILABLE = False


class DepthEstimator:
    """Estimate depth from a single RGB image using Depth Anything V2."""
    
    # Available models
    MODELS = {
        'small': 'vits',      # Fast, good accuracy
        'medium': 'vitb',     # Balanced
        'large': 'vitl'       # Best accuracy, slower
    }
    
    def __init__(self, model_type="vits", max_retries=3, timeout=60):
        """
        Load Depth Anything V2 depth estimation model.
        
        Args:
            model_type: "vits", "vitb", or "vitl"
                       Can also use shortcuts: "small", "medium", "large"
            max_retries: Unused (kept for compatibility)
            timeout: Unused (kept for compatibility)
        """
        # Handle shortcuts
        if model_type.lower() in self.MODELS:
            model_type = self.MODELS[model_type.lower()]
        
        print("\n" + "=" * 70)
        print("📊 LOADING DEPTH ESTIMATION MODEL - DEPTH ANYTHING V2")
        print("=" * 70)
        
        self.model_type = model_type
        self.model = None
        
        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        # Check if model is available
        if not DEPTH_V2_AVAILABLE or DEPTH_ANYTHING_PATH is None:
            print(f"❌ Depth Anything V2 module not found")
            print(f"   Searched: {', '.join(str(p) for p in DEPTH_PATHS)}")
            print("\n💡 To bundle for MSIX:")
            print("   1. Copy depth_anything_v2/ folder to camera_tools/scanning/")
            print("   2. Copy checkpoint .pth file to camera_tools/scanning/models/")
            raise FileNotFoundError(f"Depth Anything V2 not found in bundled locations")
        
        # Try multiple checkpoint locations (bundled, external, development)
        checkpoint_locations = [
            MODULE_DIR / 'models' / f'depth_anything_v2_{model_type}.pth',  # Bundled
            DEPTH_ANYTHING_PATH / 'checkpoints' / f'depth_anything_v2_{model_type}.pth',  # Module dir
            DEPTH_ANYTHING_PATH.parent / 'checkpoints' / f'depth_anything_v2_{model_type}.pth',  # External
        ]
        
        checkpoint_path = None
        for loc in checkpoint_locations:
            if loc.exists():
                checkpoint_path = loc
                break
        
        if checkpoint_path is None:
            print(f"❌ Model checkpoint not found: depth_anything_v2_{model_type}.pth")
            print(f"   Searched locations:")
            for loc in checkpoint_locations:
                print(f"   - {loc}")
            print(f"\n💡 Download from:")
            print(f"   https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_{model_type}.pth")
            print(f"\n   Save to: {checkpoint_locations[0]}")
            raise FileNotFoundError(f"Checkpoint not found in any location")
        
        print(f"\n✅ Loading model: {model_type}")
        print(f"   Checkpoint: {checkpoint_path.name}")
        
        # Detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {self.device.type.upper()}")
        
        # Load model
        try:
            self.model = DepthAnythingV2(**model_configs[model_type])
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device))
            self.model = self.model.to(self.device).eval()
            
            print(f"\n✅ Model loaded successfully!")
            print(f"✅ GPU available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self._print_fallback_instructions()
            raise
    
    def _print_fallback_instructions(self):
        """Print instructions for manual model download or alternative solutions."""
        print("\n💡 SOLUTIONS:")
        print("=" * 70)
        print("\n1. CLONE DEPTH ANYTHING V2:")
        print("   cd D:/Users/Planet UI/world_folder/projects/")
        print("   git clone https://github.com/DepthAnything/Depth-Anything-V2")
        print("\n2. DOWNLOAD CHECKPOINT:")
        print("   Download from HuggingFace:")
        print(f"   https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_{self.model_type}.pth")
        print(f"   Save to: {DEPTH_ANYTHING_PATH / 'checkpoints'}")
        print("\n3. INSTALL DEPENDENCIES:")
        print("   pip install torch torchvision opencv-python")
        print("\n4. USE LIGHTWEIGHT MODE:")
        print("   Scanner can run WITHOUT depth estimation")
        print("   Just skip MODE_DEPTH (press 1, 2, or 3 for other modes)")
        print("=" * 70)
    
    def estimate_depth(self, rgb_image):
        """
        Estimate depth map from RGB image.
        
        Args:
            rgb_image: BGR image from OpenCV (H, W, 3)
        
        Returns:
            depth_map: Normalized depth (H, W) where 0=far, 1=close
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Predict depth using Depth Anything V2
        with torch.no_grad():
            depth = self.model.infer_image(rgb)
        
        # Normalize to 0-1 (inverse depth - closer = smaller values)
        depth_min = depth.min()
        depth_max = depth.max()
        depth_map = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        return depth_map
    
    def depth_to_point_cloud(self, rgb_image, depth_map, camera_matrix, 
                            max_depth_m=2.0, min_depth_m=0.2, downsample=1,
                            distance_correction=1.0):
        """
        Convert depth map + RGB to 3D point cloud.
        
        Args:
            rgb_image: BGR image (H, W, 3)
            depth_map: Normalized depth (H, W) - output from estimate_depth()
            camera_matrix: Camera intrinsics K (3, 3)
            max_depth_m: Maximum depth in meters (default: 2.0)
            min_depth_m: Minimum depth in meters (default: 0.2)
            downsample: Downsample factor for performance (1=full, 2=half, 4=quarter)
            distance_correction: Correction factor to match real-world distances (default: 1.0)
        
        Returns:
            points_3d: (N, 3) array of XYZ coordinates in mm
            colors: (N, 3) array of RGB colors (0-255)
        """
        h, w = depth_map.shape
        
        # Downsample if requested
        if downsample > 1:
            depth_map = depth_map[::downsample, ::downsample]
            rgb_image = rgb_image[::downsample, ::downsample]
            h, w = depth_map.shape
        
        # Convert normalized depth to real-world depth (meters)
        # Inverse depth: 0 = far (max_depth_m), 1 = close (min_depth_m)
        depth_real = min_depth_m + (max_depth_m - min_depth_m) * (1.0 - depth_map)
        
        # Get camera parameters (adjust for downsampling)
        fx = camera_matrix[0, 0] / downsample
        fy = camera_matrix[1, 1] / downsample
        cx = camera_matrix[0, 2] / downsample
        cy = camera_matrix[1, 2] / downsample
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert to 3D using pinhole camera model
        z = depth_real * 1000 * distance_correction  # Convert to mm with correction
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack into point cloud (N, 3)
        points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Get colors (convert BGR to RGB)
        colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        
        # Filter out invalid points (outside depth range)
        valid_mask = (z.reshape(-1) > min_depth_m * 1000) & (z.reshape(-1) < max_depth_m * 1000)
        
        return points_3d[valid_mask], colors[valid_mask]
    
    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_PLASMA):
        """
        Convert depth map to colored visualization.
        
        Args:
            depth_map: Normalized depth (H, W)
            colormap: OpenCV colormap (PLASMA, TURBO, JET, etc.)
        
        Returns:
            colored_depth: BGR image for display
        """
        depth_vis = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_vis, colormap)
        return colored
    
    @staticmethod
    def show_model_guide():
        """Print model selection guide."""
        print("\n" + "=" * 70)
        print("🤖 DEPTH ESTIMATION MODEL GUIDE - DEPTH ANYTHING V2")
        print("=" * 70)
        print("\nAvailable Models:")
        print("  'small'  (vits)  - Fast, good accuracy (~25 FPS)")
        print("  'medium' (vitb)  - Balanced (~12 FPS)")
        print("  'large'  (vitl)  - Best accuracy (~5 FPS)")
        print("\nRecommended Settings:")
        print("  Real-time preview:  'small' with downsample=4")
        print("  Quality capture:    'medium' with downsample=2")
        print("  Best quality:       'large' with downsample=1")
        print("\nGPU Requirements:")
        print("  VRAM needed: small=1GB, medium=2GB, large=3GB")
        print("  CPU fallback available (slower)")
        print("\n✅ V2 Benefits: Better than MiDaS, no torch.hub, easy packaging!")
        print("=" * 70)
    
    def get_info(self):
        """Get model information string."""
        gpu_text = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
        return f"{self.model_type} on {self.device} ({gpu_text})"


# ========== SAVE POINT CLOUD FUNCTION ==========

def save_point_cloud(points_3d, colors, filename):
    """
    Save point cloud to PLY file.
    
    Args:
        points_3d: (N, 3) array of XYZ coordinates
        colors: (N, 3) array of RGB colors
        filename: Output path (.ply)
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"💾 SAVING POINT CLOUD")
    print(f"{'='*70}")
    print(f"Output file: {output_path}")
    print(f"Total points: {len(points_3d):,}")
    
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(len(points_3d)):
            x, y, z = points_3d[i]
            r, g, b = colors[i]
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")
    
    print(f"✓ Point cloud saved!")
    print(f"{'='*70}")


# ========== TEST FUNCTION (FIXED) ==========

def test_depth_estimator(image_path=None):
    """
    Test depth estimator on an image or webcam.
    
    Args:
        image_path: Path to test image, or None for webcam
    """
    import time
    from datetime import datetime
    
    print("\n" + "=" * 70)
    print("DEPTH ESTIMATOR TEST")
    print("=" * 70)
    
    # Show model guide
    DepthEstimator.show_model_guide()
    
    # Load camera calibration
    calib_path = Path(__file__).parent / "camera_calibration.npz"
    if calib_path.exists():
        data = np.load(calib_path)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print(f"\n✓ Loaded calibration: {calib_path}")
    else:
        print("\n⚠️  No calibration found, using default")
        camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
    
    # Initialize estimator
    estimator = DepthEstimator("small")  # Fast model for testing
    print(f"\n📊 {estimator.get_info()}")
    
    if image_path:
        # Test on static image - FIX PATH HANDLING
        print(f"\nLoading image: {image_path}")
        
        # Convert to Path object to handle Windows paths correctly
        image_file = Path(image_path)
        
        # Check if file exists
        if not image_file.exists():
            print(f"❌ File not found: {image_file}")
            print("\n💡 Make sure the file exists:")
            print(f"   {image_file.absolute()}")
            
            # Try to suggest nearby files
            if image_file.parent.exists():
                print("\n📁 Files in that directory:")
                for f in image_file.parent.glob("*.*"):
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        print(f"   - {f.name}")
            return
        
        # Load image with proper path conversion
        img = cv2.imread(str(image_file.absolute()))
        
        if img is None:
            print(f"❌ Could not load image: {image_file}")
            print("💡 Supported formats: JPG, PNG, BMP")
            print(f"   Your file: {image_file.suffix}")
            return
        
        print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]}")
        print("Estimating depth...")
        start = time.perf_counter()
        
        depth_map = estimator.estimate_depth(img)
        points_3d, colors = estimator.depth_to_point_cloud(
            img, depth_map, camera_matrix,
            max_depth_m=2.0, min_depth_m=0.2, downsample=2
        )
        
        elapsed = time.perf_counter() - start
        print(f"✓ Captured {len(points_3d):,} points in {elapsed:.2f}s")
        
        # Show results
        depth_colored = estimator.visualize_depth(depth_map)
        
        # Create side-by-side view
        h, w = img.shape[:2]
        depth_resized = cv2.resize(depth_colored, (w, h))
        combined = np.hstack([img, depth_resized])
        
        # Resize if too large for display
        max_width = 1920
        if combined.shape[1] > max_width:
            scale = max_width / combined.shape[1]
            new_w = int(combined.shape[1] * scale)
            new_h = int(combined.shape[0] * scale)
            combined = cv2.resize(combined, (new_w, new_h))
        
        cv2.imshow("Original (Left) | Depth Map (Right)", combined)
        
        # Save point cloud
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = image_file.parent / f"depth_{image_file.stem}_{timestamp}.ply"
        save_point_cloud(points_3d, colors, str(output_file))
        
        print(f"\n✅ Results:")
        print(f"   Image: {img.shape[1]}x{img.shape[0]}")
        print(f"   Points: {len(points_3d):,}")
        print(f"   Point cloud: {output_file}")
        print("\n💡 Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        # Test on webcam - WITH PROPER CAMERA DETECTION
        print("\n🔍 Searching for camera...")
        
        cap = None
        
        # Try different camera indices and backends
        for camera_id in [0, 1, 2]:
            print(f"\n  Testing camera {camera_id}...")
            
            # Try DSHOW first (best for Windows)
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                # Verify we can actually read frames
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"  ✓ Camera {camera_id} working with DSHOW!")
                    break
                else:
                    cap.release()
                    cap = None
            
            # Try default backend
            if cap is None:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"  ✓ Camera {camera_id} working!")
                        break
                    else:
                        cap.release()
                        cap = None
        
        if cap is None or not cap.isOpened():
            print("\n❌ NO CAMERA FOUND!")
            print("\n💡 Solutions:")
            print("   1. Plug in your camera")
            print("   2. Make sure no other app is using it")
            print("   3. Run with an image instead:")
            print('      python depth_estimator.py "C:\\Users\\steph\\Downloads\\image.jpg"')
            return
        
        # Configure camera
        print("\n⚙️  Configuring camera...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual settings
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  Resolution: {actual_w}x{actual_h}")
        print(f"  FPS: {actual_fps}")
        
        print("\n⏳ Warming up camera...")
        
        # Discard first 10 frames (camera warmup)
        for i in range(10):
            ret, _ = cap.read()
            if not ret:
                print(f"  ⚠️  Frame {i+1}/10 failed")
        
        print("✓ Camera ready!\n")
        
        print("=" * 70)
        print("CONTROLS")
        print("=" * 70)
        print("  SPACE - Capture depth (~0.5s with small model)")
        print("  s     - Save point cloud to PLY")
        print("  q     - Quit")
        print("=" * 70)
        
        depth_map = None
        points_3d = None
        colors = None
        frame_count = 0
        last_capture_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame read failed!")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Show depth overlay if available
            if depth_map is not None:
                depth_colored = estimator.visualize_depth(depth_map)
                depth_small = cv2.resize(depth_colored, (w//3, h//3))
                
                overlay_y = h - h//3 - 10
                overlay_x = w - w//3 - 10
                
                # Add semi-transparent overlay
                alpha = 0.9
                display[overlay_y:overlay_y+h//3, overlay_x:overlay_x+w//3] = \
                    cv2.addWeighted(
                        display[overlay_y:overlay_y+h//3, overlay_x:overlay_x+w//3],
                        1-alpha,
                        depth_small,
                        alpha,
                        0
                    )
                
                cv2.rectangle(display, (overlay_x, overlay_y),
                            (overlay_x+w//3, overlay_y+h//3), (0, 255, 255), 2)
                cv2.putText(display, "Depth", (overlay_x+5, overlay_y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Show point count
            if points_3d is not None:
                cv2.putText(display, f"Points: {len(points_3d):,}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show instructions
            cv2.rectangle(display, (10, h-90), (400, h-10), (0, 0, 0), -1)
            cv2.putText(display, "SPACE=Capture  S=Save  Q=Quit",
                       (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Frame: {frame_count}",
                       (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Depth Estimator Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            
            elif key == ord(' '):
                # Prevent rapid captures
                current_time = time.perf_counter()
                if current_time - last_capture_time < 1.0:
                    print("⚠️  Wait 1 second between captures")
                    continue
                
                print("\n" + "="*70)
                print("[CAPTURE] Estimating depth...")
                print("="*70)
                
                try:
                    start = time.perf_counter()
                    
                    depth_map = estimator.estimate_depth(frame)
                    points_3d, colors = estimator.depth_to_point_cloud(
                        frame, depth_map, camera_matrix,
                        max_depth_m=2.0, min_depth_m=0.2, downsample=2
                    )
                    
                    elapsed = time.perf_counter() - start
                    last_capture_time = current_time
                    
                    print(f"✓ Captured {len(points_3d):,} points in {elapsed:.2f}s")
                    print("="*70)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print("="*70)
            
            elif key == ord('s'):
                if points_3d is not None and len(points_3d) > 0:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"depth_test_{timestamp}.ply"
                        save_point_cloud(points_3d, colors, filename)
                    except Exception as e:
                        print(f"❌ Save error: {e}")
                else:
                    print("⚠️  No point cloud to save! Press SPACE first.")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test complete!")


# ========== MAIN ==========

if __name__ == "__main__":
    import sys
    
    # Check for image argument
    if len(sys.argv) > 1:
        test_depth_estimator(sys.argv[1])
    else:
        test_depth_estimator()  # Use webcam