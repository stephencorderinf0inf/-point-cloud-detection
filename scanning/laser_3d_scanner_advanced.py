"""
Advanced 3D Scanner optimized for 635nm RED LASER (Bosch GLM 42)
"""

# ========== SECTION 1: FAST IMPORTS (Lines 1-30) ==========
# Core imports only - defer heavy modules for faster startup
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import argparse

# ========== LAZY LOADING FLAGS ==========
# Heavy modules loaded on-demand
DEPTH_AVAILABLE = None  # Lazy check
OPEN3D_AVAILABLE = None  # Lazy check
_depth_estimator = None
_open3d_module = None
_spectrum_analyzer = None
_panel_display = None
_gpu_optimizer = None

def get_depth_estimator():
    """Lazy load depth estimator only when needed."""
    global _depth_estimator, DEPTH_AVAILABLE
    if DEPTH_AVAILABLE is None:
        # First check if PyTorch is available before trying to load depth_estimator
        try:
            import torch
            import torchvision
            import timm
        except ImportError as e:
            module_name = str(e).split("'")[1] if "'" in str(e) else "required module"
            print(f"⚠️  Depth estimation unavailable: {module_name} not installed")
            print(f"   Install with: pip install torch torchvision timm")
            DEPTH_AVAILABLE = False
            _depth_estimator = None
            return None
        
        # PyTorch available, now try to load depth_estimator module
        try:
            from depth_estimator import DepthEstimator
            DEPTH_AVAILABLE = True
            _depth_estimator = DepthEstimator
            print("✅ Depth estimation module loaded successfully")
        except Exception as e:
            print(f"⚠️  Depth estimator failed to load: {e}")
            DEPTH_AVAILABLE = False
            _depth_estimator = None
    return _depth_estimator

def get_open3d():
    """Lazy load Open3D only when generating meshes."""
    global _open3d_module, OPEN3D_AVAILABLE
    if OPEN3D_AVAILABLE is None:
        try:
            import open3d as o3d
            OPEN3D_AVAILABLE = True
            _open3d_module = o3d
        except ImportError:
            print("⚠️  Open3D unavailable - mesh generation disabled")
            OPEN3D_AVAILABLE = False
            _open3d_module = None
    return _open3d_module

# Wrap the entire script in try-except
try:
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Add ai_analysis to path BEFORE importing
    ai_analysis_path = Path(__file__).parent.parent / 'ai_analysis'
    sys.path.insert(0, str(ai_analysis_path))

    # Add scanning path for profiler/gpu_optimizer
    scanning_path = Path(__file__).parent
    sys.path.insert(0, str(scanning_path))

    # ✅ FIXED: Dynamic calibration path resolution
    import calibration.camera_distance_detector_calibrated
    
    # Build calibration directory path (relative to script)
    calibration_dir = Path(__file__).parent.parent / 'dual_checkerboard_3d' / 'calibration'
    
    # Function to detect camera and find matching calibration
    def find_calibration_file():
        """
        Find appropriate calibration file for current camera.
        Looks for:
        1. camera_calibration.npz (default)
        2. camera_calibration_<camera_name>.npz (camera-specific)
        3. Prompts user if multiple found
        """
        if not calibration_dir.exists():
            print(f"\n❌ Calibration directory not found: {calibration_dir}")
            print("   Please run calibration first:")
            print("   cd dual_checkerboard_3d")
            print("   python checkerboard.py")
            return None
        
        # Find all calibration files
        calibration_files = list(calibration_dir.glob("camera_calibration*.npz"))
        
        if len(calibration_files) == 0:
            print(f"\n❌ No calibration files found in: {calibration_dir}")
            print("   Please run calibration first:")
            print("   cd dual_checkerboard_3d")
            print("   python checkerboard.py")
            return None
        
        elif len(calibration_files) == 1:
            # Only one calibration file - use it
            cal_file = calibration_files[0]
            print(f"✓ Found calibration: {cal_file.name}")
            return cal_file
        
        else:
            # Multiple calibration files - let user choose
            print(f"\n📁 Found {len(calibration_files)} calibration files:")
            for i, cal_file in enumerate(calibration_files, 1):
                # Try to extract camera name from filename
                name = cal_file.stem.replace("camera_calibration", "").strip("_")
                name = name or "default"
                
                # Get file modification time
                import time
                mod_time = time.ctime(cal_file.stat().st_mtime)
                
                print(f"  {i}. {cal_file.name}")
                print(f"      Camera: {name}")
                print(f"      Modified: {mod_time}")
                print()
            
            # Auto-select most recent by default
            default_choice = "1"
            choice = input(f"👉 Select calibration file (1-{len(calibration_files)}) [{default_choice}]: ").strip()
            
            if not choice:
                choice = default_choice
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(calibration_files):
                    cal_file = calibration_files[idx]
                    print(f"✓ Using: {cal_file.name}")
                    return cal_file
                else:
                    print("⚠️  Invalid choice - using first file")
                    return calibration_files[0]
            except ValueError:
                print("⚠️  Invalid input - using first file")
                return calibration_files[0]
    
    # Find and set calibration file
    calibration_file = find_calibration_file()
    
    if calibration_file:
        calibration.camera_distance_detector_calibrated.CALIBRATION_FILE = str(calibration_file)
        print(f"✓ Calibration loaded: {calibration_file}")
    else:
        print("\n❌ Cannot continue without calibration!")
        print("   Run: python dual_checkerboard_3d/checkerboard.py")
        sys.exit(1)

    # Prompt for save file location
    def prompt_save_location():
        """Ask user where to save point cloud files."""
        print("\n" + "="*70)
        print("💾 SAVE FILE LOCATION")
        print("="*70)
        print("Where do you want to save point cloud files?")
        print()
        print("  1. Custom location (name file/folder)")
        print("  2. Legacy default (data/point_clouds/)")
        print("  3. List existing scan folders")
        print()
        
        choice = input("👉 Select option (1-3) [2]: ").strip()
        
        if not choice:
            choice = "2"
        
        if choice == "1":
            # Custom location
            print("\nEnter custom save location:")
            print("Examples:")
            print("  - scan_results/my_object")
            print("  - ../output/scan_20250112")
            print("  - C:/Users/MyName/Desktop/scans")
            print()
            custom_path = input("👉 Path: ").strip()
            
            if custom_path:
                save_dir = Path(custom_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                print(f"✓ Will save to: {save_dir.absolute()}")
                return save_dir
            else:
                print("⚠️  No path entered, using legacy default")
                return Path(__file__).parent / "data" / "point_clouds"
        
        elif choice == "3":
            # List existing folders
            data_dir = Path(__file__).parent / "data"
            if data_dir.exists():
                subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
                if subdirs:
                    print("\nExisting scan folders:")
                    for i, folder in enumerate(subdirs, 1):
                        file_count = len(list(folder.glob("*.ply"))) + len(list(folder.glob("*.npz")))
                        print(f"  {i}. {folder.name} ({file_count} files)")
                    print()
                    
                    folder_choice = input(f"👉 Select folder (1-{len(subdirs)}) or press Enter for new: ").strip()
                    
                    if folder_choice and folder_choice.isdigit():
                        idx = int(folder_choice) - 1
                        if 0 <= idx < len(subdirs):
                            selected_dir = subdirs[idx]
                            print(f"✓ Will save to: {selected_dir}")
                            return selected_dir
                else:
                    print("\n⚠️  No existing folders found")
            
            # Fallback to default
            print("Using legacy default location")
            return Path(__file__).parent / "data" / "point_clouds"
        
        else:
            # Option 2 or default - legacy location
            legacy_dir = Path(__file__).parent / "data" / "point_clouds"
            print(f"✓ Using legacy location: {legacy_dir}")
            return legacy_dir
    
    # Get save location from user
    SAVE_DIRECTORY = prompt_save_location()
    print("="*70 + "\n")

    # ========== LAZY IMPORT: Only load when first used ==========
    print("⚡ Fast startup mode - deferring heavy module loading...")
    
    # These will be imported on-demand during runtime
    _calibration_module = None
    _ai_modules_loaded = False
    
    print("✓ Core imports ready! (Heavy modules load on-demand)")

except Exception as e:
    print(f"\n❌ ERROR during imports:")
    print(f"   {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# ========== LAZY LOADER FUNCTIONS ==========
def load_calibration():
    """Load calibration module only when needed."""
    global _calibration_module
    if _calibration_module is None:
        from calibration.camera_distance_detector_calibrated import load_camera_calibration
        _calibration_module = load_camera_calibration
    return _calibration_module

def load_ai_modules():
    """Load AI analysis modules only when needed."""
    global _ai_modules_loaded
    if not _ai_modules_loaded:
        print("🔄 Loading AI modules...")
        global get_camera_info, analyze_image_quality, save_analysis_result
        global finalize_session, OptimizedAIAnalyzer
        
        from camera_info import get_camera_info
        from image_quality import analyze_image_quality
        from results_storage import save_analysis_result, finalize_session
        from optimized_analyzer import OptimizedAIAnalyzer
        
        _ai_modules_loaded = True
        print("✓ AI modules loaded")

def load_performance_modules():
    """Load performance monitoring modules only when needed."""
    from performance_profiler import profiler
    from gpu_optimizer import GPUOptimizer
    return profiler, GPUOptimizer

def load_spectrum_analyzer():
    """Load spectrum analyzer on-demand."""
    global _spectrum_analyzer
    if _spectrum_analyzer is None:
        from spectrum_config import SpectrumAnalyzer
        _spectrum_analyzer = SpectrumAnalyzer
    return _spectrum_analyzer

def load_panel_display():
    """Load panel display module on-demand."""
    global _panel_display
    if _panel_display is None:
        from panel_display_module import PanelDisplayModule
        _panel_display = PanelDisplayModule
    return _panel_display

# ========== SECTION 2: GLOBAL CONSTANTS (Lines 151-250) ==========
WEBCAM_INDEX = 0

# Detection modes
MODE_LASER = 0
MODE_CURVE = 1
MODE_CORNERS = 2
MODE_DEPTH = 3  # 🎨 NEW: AI depth estimation mode

# Capture modes
CAPTURE_MODE_PHOTO = 0  # Capture current video frame
CAPTURE_MODE_POINTCLOUD = 1  # Capture point cloud 3D view

# Red laser detection settings for 635nm (ORANGE-RED, not pure red!)
# 635nm appears as ORANGE-RED, so we expand the hue range
DEFAULT_RED_HUE_MIN = 0
DEFAULT_RED_HUE_MAX = 20  # Expanded to 20 to catch orange-red (635nm)
DEFAULT_SATURATION_MIN = 80  # Lower saturation for 635nm (not as pure as deep red)
DEFAULT_VALUE_MIN = 100
DEFAULT_BRIGHTNESS = 140  # Lower brightness for 635nm (dimmer than green on camera)
DEFAULT_MIN_AREA = 5
DEFAULT_MAX_AREA = 1000

red_hue_min = DEFAULT_RED_HUE_MIN
red_hue_max = DEFAULT_RED_HUE_MAX
saturation_min = DEFAULT_SATURATION_MIN
value_min = DEFAULT_VALUE_MIN

LASER_MIN_AREA = DEFAULT_MIN_AREA
LASER_MAX_AREA = DEFAULT_MAX_AREA

# Distance estimation
DISTANCE_MIN_CM = 20
DISTANCE_MAX_CM = 150
Y_AT_MIN_DISTANCE = 500
Y_AT_MAX_DISTANCE = 100

# ROI (Region of Interest) cropping
roi_enabled = False
roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, 0, 0
roi_selecting = False
roi_start_x, roi_start_y = 0, 0

# Canny edge detection thresholds (for curve detection)
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Add this near the top, after other global variables
info_box_visible = True
ai_panel_visible = True
cartoon_mode = False  # Toggle for cartoon-style camera settings
auto_capture_mode = True  # Toggle for automatic 3-capture mode
auto_capture_countdown = 0  # Countdown timer
auto_capture_count = 0  # How many captures done
auto_capture_target = 3  # How many captures to do automatically

# === NEW: Rotation tracking variables ===
capture_metadata = {
    'angles': [],        # Rotation angle for each point
    'sessions': [],      # Session ID for each point
    'timestamps': [],    # Capture time for each point
}
current_session = 0
current_angle = 0.0
rotation_step = 30.0  # Default: 30° per rotation (user can change)

# Add after line ~1228 (initialization section):
curve_sample_rate = 5      # Default: every 5th point
corner_max_count = 100     # Default: 100 corners
canny_threshold1 = 50      # Edge detection lower threshold
canny_threshold2 = 150     # Edge detection upper threshold

# ========== SECTION 3: HELPER FUNCTIONS (Lines 251-1100) ==========
def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for ROI selection"""
    global roi_selecting, roi_start_x, roi_start_y, roi_x1, roi_y1, roi_x2, roi_y2, roi_enabled
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start selecting ROI
        roi_selecting = True
        roi_start_x, roi_start_y = x, y
        roi_x1, roi_y1 = x, y
        roi_x2, roi_y2 = x, y
        print("[SCISSORS] Drawing ROI - drag and release...")
        
    elif event == cv2.EVENT_MOUSEMOVE and roi_selecting:
        # Update ROI rectangle while dragging
        roi_x2, roi_y2 = x, y
        
    elif event == cv2.EVENT_LBUTTONUP:
        # Finish selecting ROI
        roi_selecting = False
        roi_x2, roi_y2 = x, y
        
        # Ensure coordinates are in correct order
        roi_x1, roi_x2 = min(roi_start_x, x), max(roi_start_x, x)
        roi_y1, roi_y2 = min(roi_start_y, y), max(roi_start_y, y)
        
        # Only enable if area is reasonable
        width = roi_x2 - roi_x1
        height = roi_y2 - roi_y1
        
        if width > 50 and height > 50:
            roi_enabled = True
            print(f"[CHECK] ROI set: {width}x{height} at ({roi_x1}, {roi_y1})")
            print(f"        Detection will focus on this region only")
        else:
            print("[X] ROI too small (min 50x50 pixels), disabled")
            roi_enabled = False

def detect_red_laser_dot(frame, brightness_threshold, min_area, max_area, 
                         hue_min, hue_max, sat_min, val_min):
    """Detect 635nm RED-ORANGE laser using HSV color detection."""
    
    # Store original frame dimensions for coordinate adjustment
    frame_offset_x, frame_offset_y = 0, 0
    
    # Apply ROI cropping if enabled
    if roi_enabled:
        h, w = frame.shape[:2]
        x1 = max(0, min(roi_x1, w-1))
        y1 = max(0, min(roi_y1, h-1))
        x2 = max(x1+1, min(roi_x2, w))
        y2 = max(y1+1, min(roi_y2, h))
        
        if x2 > x1 and y2 > y1:
            frame = frame[y1:y2, x1:x2].copy()
            frame_offset_x = x1
            frame_offset_y = y1
    
    # Check if frame is valid
    if frame.size == 0:
        return None, None, None, [], np.zeros((10,10), dtype=np.uint8), np.zeros((10,10), dtype=np.uint8), np.zeros((10,10), dtype=np.uint8)
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # RED-ORANGE color (635nm) has hue range 0-20
    lower_red = np.array([hue_min, sat_min, val_min])
    upper_red = np.array([hue_max, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Also check upper red range (170-180) for pure red
    lower_red2 = np.array([170, sat_min, val_min])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine both ranges
    red_mask = cv2.bitwise_or(red_mask, mask2)
    
    # Also check brightness
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # COMBINED: Must be BOTH red AND bright (red laser!)
    combined_mask = cv2.bitwise_and(red_mask, bright_mask)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_dots = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Adjust coordinates back to full frame
                cx_full = cx + frame_offset_x
                cy_full = cy + frame_offset_y
                
                brightness = gray[cy, cx]
                
                # Get HSV values at center
                h, s, v = hsv[cy, cx]
                
                valid_dots.append({
                    'x': cx_full,
                    'y': cy_full,
                    'area': area,
                    'brightness': int(brightness),
                    'hue': int(h),
                    'saturation': int(s),
                    'value': int(v)
                })
    
    if len(valid_dots) == 0:
        return None, None, None, [], combined_mask, red_mask, bright_mask
    
    # Return the brightest RED dot (fix overflow by using float)
    best = max(valid_dots, key=lambda d: float(d['saturation']) * float(d['brightness']))
    return best['x'], best['y'], best['area'], valid_dots, combined_mask, red_mask, bright_mask

def detect_curves(frame):
    """Detect curves/edges in the scene."""
    
    # Store original frame dimensions for coordinate adjustment
    frame_offset_x, frame_offset_y = 0, 0
    
    # Apply ROI cropping if enabled
    if roi_enabled:
        # Validate ROI bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(roi_x1, w-1))
        y1 = max(0, min(roi_y1, h-1))
        x2 = max(x1+1, min(roi_x2, w))
        y2 = max(y1+1, min(roi_y2, h))
        
        # Check if ROI is valid
        if x2 <= x1 or y2 <= y1:
            print(f"[WARNING] Invalid ROI bounds: ({x1},{y1}) to ({x2},{y2})")
            # Use full frame instead
            pass
        else:
            frame = frame[y1:y2, x1:x2].copy()
            frame_offset_x = x1
            frame_offset_y = y1
    
    # Check if frame is empty
    if frame.size == 0:
        print("[ERROR] Empty frame in detect_curves!")
        return [], np.zeros((10, 10), dtype=np.uint8)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Adjust contours back to full frame coordinates
    adjusted_contours = []
    for cnt in contours:
        adjusted = cnt.copy()
        adjusted[:, 0, 0] += frame_offset_x
        adjusted[:, 0, 1] += frame_offset_y
        adjusted_contours.append(adjusted)
    
    valid_curves = [cnt for cnt in adjusted_contours if cv2.contourArea(cnt) > 100]
    return valid_curves, edges

def detect_corners(frame):
    """Detect corners using Harris corner detection."""
    
    # Store original frame dimensions for coordinate adjustment
    frame_offset_x, frame_offset_y = 0, 0
    
    # Apply ROI cropping if enabled
    if roi_enabled:
        # Validate ROI bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(roi_x1, w-1))
        y1 = max(0, min(roi_y1, h-1))
        x2 = max(x1+1, min(roi_x2, w))
        y2 = max(y1+1, min(roi_y2, h))
        
        if x2 <= x1 or y2 <= y1:
            print(f"[WARNING] Invalid ROI bounds: ({x1},{y1}) to ({x2},{y2})")
        else:
            frame = frame[y1:y2, x1:x2].copy()
            frame_offset_x = x1
            frame_offset_y = y1
    
    # Check if frame is empty
    if frame.size == 0:
        print("[ERROR] Empty frame in detect_corners!")
        return [], np.zeros((10, 10), dtype=np.float32)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    corner_threshold = 0.01 * corners.max()
    corner_locations = np.where(corners > corner_threshold)
    
    # Adjust corner coordinates back to full frame
    corner_points = list(zip(
        corner_locations[1] + frame_offset_x,
        corner_locations[0] + frame_offset_y
    ))
    
    return corner_points, corners

def detect_ellipses(frame):
    """Detect ellipses in the scene using contour fitting."""
    frame_offset_x, frame_offset_y = 0, 0
    
    if roi_enabled:
        h, w = frame.shape[:2]
        x1 = max(0, min(roi_x1, w-1))
        y1 = max(0, min(roi_y1, h-1))
        x2 = max(x1+1, min(roi_x2, w))
        y2 = max(y1+1, min(roi_y2, h))
        
        if x2 > x1 and y2 > y1:
            frame = frame[y1:y2, x1:x2].copy()
            frame_offset_x = x1
            frame_offset_y = y1
    
    if frame.size == 0:
        return [], np.zeros((10, 10), dtype=np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for cnt in contours:
        if len(cnt) >= 5 and cv2.contourArea(cnt) > 100:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), axes, angle = ellipse
            x += frame_offset_x
            y += frame_offset_y
            ellipses.append(((x, y), axes, angle))
    return ellipses, edges

def detect_cylinders(frame):
    """Detect cylindrical objects by finding long, parallel contours."""
    frame_offset_x, frame_offset_y = 0, 0
    
    if roi_enabled:
        h, w = frame.shape[:2]
        x1 = max(0, min(roi_x1, w-1))
        y1 = max(0, min(roi_y1, h-1))
        x2 = max(x1+1, min(roi_x2, w))
        y2 = max(y1+1, min(roi_y2, h))
        
        if x2 > x1 and y2 > y1:
            frame = frame[y1:y2, x1:x2].copy()
            frame_offset_x = x1
            frame_offset_y = y1
    
    if frame.size == 0:
        return [], np.zeros((10, 10), dtype=np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cylinders = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            rect = cv2.minAreaRect(cnt)
            (x, y), (w_rect, h_rect), angle = rect
            aspect_ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1e-5)
            if aspect_ratio > 3.0:
                x += frame_offset_x
                y += frame_offset_y
                cylinders.append(((x, y), (w_rect, h_rect), angle))
    return cylinders, edges

def suggest_roi_from_contrast(frame):
    """
    Automatically suggest ROI based on high-contrast regions.
    Helps find object boundaries against backdrop.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find edges (high contrast areas = likely object boundaries)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, None
    
    # Find largest contour (likely the main object)
    largest = max(contours, key=cv2.contourArea)
    
    # Get bounding box with 10% padding
    x, y, w, h = cv2.boundingRect(largest)
    
    padding_x = int(w * 0.1)
    padding_y = int(h * 0.1)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(frame.shape[1], x + w + padding_x)
    y2 = min(frame.shape[0], y + h + padding_y)
    
    return x1, y1, x2, y2

def estimate_distance_linear(dot_y):
    """Linear interpolation for distance based on Y position."""
    y_clamped = max(Y_AT_MAX_DISTANCE, min(Y_AT_MIN_DISTANCE, dot_y))
    t = (y_clamped - Y_AT_MAX_DISTANCE) / (Y_AT_MIN_DISTANCE - Y_AT_MAX_DISTANCE)
    distance_cm = DISTANCE_MAX_CM - t * (DISTANCE_MAX_CM - DISTANCE_MIN_CM)
    return distance_cm

def run_ai_analysis(frame, camera_matrix, cap):
    """Call AI analysis modules and combine results."""
    camera_info = get_camera_info(frame, camera_matrix, cap)
    quality_info = analyze_image_quality(frame)
    
    return {
        **camera_info,
        **quality_info
    }

def detect_laser_with_spectrum(frame, analyzer, brightness_threshold, min_area, max_area, sat_min, val_min):
    """Wrapper for spectrum analyzer detection."""
    return analyzer.detect_spectrum_dot(frame, brightness_threshold, min_area, max_area, sat_min, val_min)

def apply_cartoon_settings(cap, enable=True):
    """
    Apply cartoon-style camera settings for enhanced edge detection.
    
    Settings optimized for:
    - High contrast edges (great for curve detection)
    - Enhanced sharpness (better feature detection)
    - Boosted saturation (laser visibility)
    """
    if enable:
        # Cartoon mode settings
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 129)
        cap.set(cv2.CAP_PROP_EXPOSURE, -8)  # EV -8
        cap.set(cv2.CAP_PROP_CONTRAST, 138)
        cap.set(cv2.CAP_PROP_SHARPNESS, 160)
        cap.set(cv2.CAP_PROP_SATURATION, 178)
        print("[CARTOON] 🎨 Cartoon mode ON: Brightness=129 EV=-8 Contrast=138 Sharp=160 Sat=178")
    else:
        # Default/auto settings
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Auto
        cap.set(cv2.CAP_PROP_CONTRAST, 128)
        cap.set(cv2.CAP_PROP_SHARPNESS, 128)
        cap.set(cv2.CAP_PROP_SATURATION, 128)
        print("[DEFAULT] 📷 Default mode: Auto settings restored")

def show_capture_overlay(frame, message, progress=None, total=3):
    """
    Show a prominent overlay message during auto-capture.
    
    Args:
        frame: Video frame to draw on
        message: Main message text
        progress: Current capture number (1-3) or None
        total: Total captures to perform
    """
    h, w = frame.shape[:2]
    
    # Create semi-transparent dark overlay (covers entire frame)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw main message box
    box_h, box_w = 200, 600
    box_x = (w - box_w) // 2
    box_y = (h - box_h) // 2
    
    # Draw box background
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 3)
    
    # Draw message
    cv2.putText(frame, message, (box_x + 50, box_y + 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Draw progress if provided
    if progress is not None:
        # Progress text
        progress_text = f"Capture {progress}/{total}"
        cv2.putText(frame, progress_text, (box_x + 180, box_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Progress bar
        bar_w = 500
        bar_h = 30
        bar_x = box_x + 50
        bar_y = box_y + 140
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        
        # Progress fill
        fill_w = int(bar_w * (progress / total))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
        
        # Progress percentage
        percentage = int((progress / total) * 100)
        cv2.putText(frame, f"{percentage}%", (box_x + 260, bar_y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # Show "Hold Steady" message
        cv2.putText(frame, "Hold Object Steady!", (box_x + 120, box_y + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    
    return frame

def auto_capture_3_points(cap, current_mode, new_camera_matrix, points_3d, 
                         analyzer=None, brightness_threshold=None, 
                         min_area=None, max_area=None,
                         saturation_min=None, value_min=None,
                         camera_matrix=None, dist_coeffs=None, gpu_opt=None,
                         points_colors=None, point_angles=None, point_sessions=None,
                         current_angle=0.0, current_session=0):
    """Automatically capture 3 FRESH frames with 1-second intervals."""
    import time
    
    # Initialize lists if not provided
    if points_colors is None:
        points_colors = []
    if point_angles is None:
        point_angles = []
    if point_sessions is None:
        point_sessions = []
    
    window_name = "Bosch GLm 42 Scanner (635nm)"
    
    for capture_num in range(1, 4):  # 3 captures
        print(f"\n[AUTO-CAPTURE {capture_num}/3] Capturing...")
        
        # Show countdown with progress bar
        for countdown in range(10, 0, -1):  # 1 second countdown (10 frames at ~100ms each)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Undistort frame for display
            display_frame = gpu_opt.undistort_frame(frame) if gpu_opt else frame
            
            # Show progress overlay
            countdown_sec = countdown / 10.0
            message = f"Capturing {capture_num}/3 in {countdown_sec:.1f}s..."
            display_frame = show_capture_overlay(display_frame, message, capture_num, 3)
            
            cv2.imshow(window_name, display_frame)
            cv2.waitKey(100)  # 100ms delay = ~10 FPS for smooth countdown
        
        # Capture and undistort frame
        ret, frame = cap.read()
        if not ret:
            print(f"    ❌ Failed to capture frame {capture_num}")
            continue
        
        undistorted = gpu_opt.undistort_frame(frame) if gpu_opt else frame
        h, w = undistorted.shape[:2]
        
        # Show "CAPTURING" message
        display_frame = undistorted.copy()
        message = f"CAPTURING {capture_num}/3..."
        display_frame = show_capture_overlay(display_frame, message, capture_num, 3)
        cv2.imshow(window_name, display_frame)
        cv2.waitKey(500)
        
        if current_mode == MODE_LASER:
            # Detect laser dot on FRESH frame
            dot_x, dot_y, dot_area, all_dots, combined_mask, color_mask, bright_mask = \
                detect_laser_with_spectrum(
                    undistorted, analyzer, brightness_threshold, min_area, max_area,
                    saturation_min, value_min
                )
            
            if dot_x and dot_y:
                fx = new_camera_matrix[0, 0]
                fy = new_camera_matrix[1, 1]
                cx = new_camera_matrix[0, 2]
                cy = new_camera_matrix[1, 2]
                
                distance_cm = estimate_distance_linear(dot_y)
                z = distance_cm * 10
                x = (dot_x - cx) * z / fx
                y = (dot_y - cy) * z / fy
                
                points_3d.append([x, y, z])
                
                # 🎨 NEW: Capture RGB color at laser dot position
                color_bgr = undistorted[dot_y, dot_x]  # Get BGR pixel
                color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]  # Convert to RGB
                points_colors.append(color_rgb)
                
                # Add to metadata
                point_angles.append(current_angle)
                point_sessions.append(current_session)
                
                print(f"    ✓ Point {capture_num}: ({x:.1f}, {y:.1f}, {z:.1f})mm RGB({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})")
            else:
                print(f"    ⚠️  No laser detected for capture {capture_num}")
        
        elif current_mode == MODE_CURVE:
            # Detect curves on FRESH frame
            curves, _ = detect_curves(undistorted)
            count_before = len(points_3d)
            
            for curve in curves:
                for point in curve[::curve_sample_rate]:  # Use variable instead of hardcoded [::5]
                    px, py = point[0]
                    
                    # ✅ BOUNDS CHECK
                    py = max(0, min(int(py), h-1))
                    px = max(0, min(int(px), w-1))
                    
                    # Calculate 3D coordinates
                    distance_cm = estimate_distance_linear(py)
                    z = distance_cm * 10
                    x = (px - new_camera_matrix[0, 2]) * z / new_camera_matrix[0, 0]
                    y = (py - new_camera_matrix[1, 2]) * z / new_camera_matrix[1, 1]
                    points_3d.append([x, y, z])
                    
                    # 🎨 CAPTURE COLOR
                    color_bgr = undistorted[py, px]
                    color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
                    points_colors.append(color_rgb)
                    
                    # Add metadata
                    point_angles.append(current_angle)
                    point_sessions.append(current_session)
            
            added = len(points_3d) - count_before
            print(f"    ✓ Added {added} colored curve points (capture {capture_num}/3)")
        
        elif current_mode == MODE_CORNERS:
            # Detect corners on FRESH frame
            corner_points, _ = detect_corners(undistorted)
            count_before = len(points_3d)
            
            for (px, py) in corner_points[:corner_max_count]:  # Use variable
                # ✅ BOUNDS CHECK (was missing!)
                py = max(0, min(int(py), h-1))
                px = max(0, min(int(px), w-1))
                
                distance_cm = estimate_distance_linear(py)
                z = distance_cm * 10
                x = (px - new_camera_matrix[0, 2]) * z / new_camera_matrix[0, 0]
                y = (py - new_camera_matrix[1, 2]) * z / new_camera_matrix[1, 1]
                points_3d.append([x, y, z])
                
                # 🎨 CAPTURE COLOR (was missing!)
                color_bgr = undistorted[py, px]
                color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
                points_colors.append(color_rgb)
                
                # Add metadata
                point_angles.append(current_angle)
                point_sessions.append(current_session)
            
            added = len(points_3d) - count_before
            print(f"    ✓ Added {added} colored corner points (capture {capture_num}/3)")
    
    # Show completion message
    ret, frame = cap.read()
    if ret:
        display_frame = gpu_opt.undistort_frame(frame) if gpu_opt else frame
        message = "AUTO-CAPTURE COMPLETE!"
        display_frame = show_capture_overlay(display_frame, message, 3, 3)
        cv2.imshow(window_name, display_frame)
        cv2.waitKey(1000)
    
    print(f"\n✅ AUTO-CAPTURE COMPLETE! Total points: {len(points_3d)} ({len(points_colors)} colored)")
    
    return points_3d, points_colors, point_angles, point_sessions, "READY"  # ✅ RETURN COLORS TOO

# === REMOVE OR COMMENT OUT OLD FUNCTION ===
# def auto_capture_3_points(...):  # OLD FUNCTION - DELETE THIS
#     ...

# === NEW: Use AutoCaptureModule instead ===
def auto_capture_3_points_with_module(cap, points_3d, calibration_data, frame_width, frame_height,
                                      brightness_threshold, red_hue_min, red_hue_max, 
                                      saturation_min, value_min, min_area, max_area,
                                      roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2,
                                      capture_metadata, current_session, current_angle):
    """
    Uses AutoCaptureModule to capture 3 points with visual feedback.
    """
    # Initialize auto-capture module
    auto_capture = AutoCaptureModule(
        window_name="Bosch GLM 42 Scanner",
        capture_count=3,
        interval_seconds=1.0
    )
    
    # Increment session
    current_session += 1
    
    print(f"\n[AUTO-CAPTURE] Session {current_session} at angle {current_angle:.1f}°")
    
    # Define processing callback for each captured frame
    def process_frame(frame, capture_num):
        """Process each captured frame to detect laser dot and calculate 3D point."""
        # Apply ROI if enabled
        if roi_enabled:
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        else:
            roi_frame = frame
            roi_x1, roi_y1 = 0, 0
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
        # Threshold for red laser dot
        mask = cv2.inRange(hsv, 
                          np.array([red_hue_min, saturation_min, value_min]),
                          np.array([red_hue_max, 255, 255]))
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find brightest dot
        dot_x, dot_y = None, None
        max_brightness = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Get bounding rect
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                cx = x + w // 2
                cy = y + h // 2
                
                # Check brightness
                if 0 <= cy < roi_frame.shape[0] and 0 <= cx < roi_frame.shape[1]:
                    brightness = hsv[cy, cx, 2]
                    if brightness > max_brightness and brightness >= brightness_threshold:
                        max_brightness = brightness
                        dot_x = roi_x1 + cx
                        dot_y = roi_x1 + cy
        
        # If dot found, calculate 3D point
        if dot_x and dot_y:
            # Undistort point
            mtx = calibration_data['camera_matrix']
            dist = calibration_data['dist_coeffs']
            
            point_2d = np.array([[[dot_x, dot_y]]], dtype=np.float32)
            undistorted = cv2.undistortPoints(point_2d, mtx, dist, P=mtx)
            dot_x_undist, dot_y_undist = undistorted[0][0]
            
            # Calculate 3D coordinates (using existing logic)
            fx = mtx[0, 0]
            fy = mtx[1, 1]
            cx = mtx[0, 2]
            cy = mtx[1, 2]
            
            # Placeholder depth (replace with actual laser distance measurement)
            depth = 500.0  # mm
            
            x = (dot_x_undist - cx) * depth / fx
            y = (dot_y_undist - cy) * depth / fy
            z = depth
            
            # Add to points list
            points_3d.append([x, y, z])
            
            # Add metadata
            capture_metadata['angles'].append(current_angle)
            capture_metadata['sessions'].append(current_session)
            capture_metadata['timestamps'].append(time.time())
            
            print(f"   [{capture_num}/3] ✓ Point captured: ({x:.1f}, {y:.1f}, {z:.1f}) mm @ {current_angle:.1f}°")
            return True
        else:
            print(f"   [{capture_num}/3] ✗ No laser dot detected")
            return False
    
    # Execute capture sequence with progress bar
    frames, metadata, success = auto_capture.capture_sequence(cap, process_frame)
    
    # Count successful captures
    captured_count = len([m for m in metadata if 'timestamp' in str(m)])
    
    print(f"[AUTO-CAPTURE] Session {current_session} complete: {captured_count}/3 points\n")
    
    return current_session, current_angle

def save_point_cloud(points_3d, filename="scan_3d_bosch_glm42.npz", 
                     angles=None, sessions=None, rotation_step=30.0):
    """
    Save 3D point cloud to compressed NPZ file with rotation metadata.
    
    Args:
        points_3d: List of [x, y, z] points
        filename: Output filename
        angles: List of rotation angles for each point
        sessions: List of capture session IDs for each point
        rotation_step: Rotation step size in degrees
    """
    if not points_3d:
        print("[ERROR] No points to save")
        return None
    
    # Convert to numpy array
    points_array = np.array(points_3d)
    
    # Create output directory
    output_dir = Path(__file__).parent / "data" / "point_clouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    # Prepare save data
    save_data = {
        'points': points_array,
        'timestamp': datetime.now().isoformat(),
        'count': len(points_array)
    }
    
    # Add rotation metadata if provided
    if angles is not None and sessions is not None:
        save_data['angles'] = np.array(angles)
        save_data['sessions'] = np.array(sessions)
        save_data['rotation_step'] = rotation_step
        
        # Calculate coverage stats
        unique_angles = np.unique(angles)
        angle_span = angles.max() - angles.min()
        
        print(f"\n📐 ROTATION METADATA:")
        print(f"   Unique angles: {len(unique_angles)}")
        print(f"   Angular coverage: {angle_span:.1f}°")
        print(f"   Rotation step: {rotation_step}°")
        print(f"   Sessions: {len(np.unique(sessions))}")
    
    # Save with compression
    np.savez_compressed(output_path, **save_data)
    
    print(f"\n💾 Point cloud saved:")
    print(f"   Location: {output_path}")
    print(f"   Points: {len(points_array):,}")
    print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return output_path


def visualize_point_cloud_3d(points_3d, points_colors=None, window_name="3D Point Cloud Viewer", width=1280, height=720):
    """
    Launch interactive Open3D 3D viewer for current point cloud.
    
    Args:
        points_3d: Nx3 numpy array of 3D points
        points_colors: Nx3 numpy array of RGB colors (0-255)
        window_name: Name of the viewer window
        width: Window width to match camera feed
        height: Window height to match camera feed
    """
    o3d = get_open3d()
    if o3d is None:
        print("⚠️  Open3D not installed - 3D viewer unavailable")
        print("   Install with: pip install open3d")
        return
    
    if len(points_3d) == 0:
        print("⚠️  No points to visualize!")
        return
    
    try:
        print(f"\n🎨 Launching 3D viewer with {len(points_3d):,} points...")
        
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Add colors if available
        if points_colors is not None and len(points_colors) == len(points_3d):
            colors_normalized = np.array(points_colors) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        else:
            # Default gradient coloring by height (Z-axis)
            z_values = points_3d[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            if z_max > z_min:
                normalized_z = (z_values - z_min) / (z_max - z_min)
                # Blue (low) to Red (high) gradient
                colors = np.zeros((len(points_3d), 3))
                colors[:, 0] = normalized_z  # Red channel
                colors[:, 2] = 1.0 - normalized_z  # Blue channel
                pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50.0, origin=[0, 0, 0]
        )
        
        # Visualization settings
        print("\n📊 3D Viewer Controls:")
        print("   • Mouse Left: Rotate view")
        print("   • Mouse Right: Pan view")
        print("   • Mouse Wheel: Zoom in/out")
        print("   • Ctrl + Mouse Left: Roll view")
        print("   • H: Show/hide coordinate frame")
        print("   • R: Reset viewpoint")
        print("   • Q/ESC: Close viewer\n")
        
        # Position 3D viewer in top-right corner to avoid interfering with main window
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            root.destroy()
            
            # Position in top-right corner with some margin
            viewer_left = screen_width - width - 50
            viewer_top = 50
        except:
            viewer_left = 50
            viewer_top = 50
        
        # Launch visualizer
        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name=window_name,
            width=width,
            height=height,
            left=viewer_left,
            top=viewer_top,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )
        
        print("✓ 3D viewer closed")
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")


def generate_poisson_mesh(ply_path, octree_depth=9):
    """
    Generate a Poisson surface reconstruction mesh from a point cloud.
    
    Args:
        ply_path: Path to the input .ply point cloud file
        octree_depth: Poisson octree depth (8-10 typical, higher=more detail)
    
    Returns:
        Tuple of (mesh_ply_path, mesh_obj_path) or (None, None) on failure
    """
    # Lazy load Open3D
    o3d = get_open3d()
    if o3d is None:
        print("⚠️  Open3D not installed - skipping mesh generation")
        print("   Install with: pip install open3d")
        return None, None
    
    try:
        print(f"\n🔨 Generating Poisson mesh...")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(ply_path))
        num_points = len(pcd.points)
        
        if num_points < 100:
            print(f"⚠️  Too few points ({num_points}) for mesh generation")
            return None, None
        
        print(f"   Loaded {num_points:,} points")
        
        # Estimate normals
        print(f"   Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=5.0,  # Search radius in mm
                max_nn=30     # Max nearest neighbors
            )
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=10)
        
        # Poisson reconstruction
        print(f"   Running Poisson reconstruction (depth={octree_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=octree_depth
        )
        
        # Remove low-density vertices (outliers)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.005)  # Remove bottom 0.5% - preserve more detail
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        print(f"   Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
        
        # Save mesh files
        base_path = ply_path.parent / ply_path.stem
        mesh_ply_path = Path(str(base_path) + "_poisson.ply")
        mesh_obj_path = Path(str(base_path) + "_poisson.obj")
        
        # Save as PLY and OBJ
        o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
        o3d.io.write_triangle_mesh(str(mesh_obj_path), mesh)
        
        print(f"   ✓ Saved mesh to:")
        print(f"     PLY: {mesh_ply_path.name}")
        print(f"     OBJ: {mesh_obj_path.name}")
        
        return mesh_ply_path, mesh_obj_path
        
    except Exception as e:
        print(f"❌ Mesh generation failed: {e}")
        return None, None


def generate_bpa_mesh(ply_path, ball_radius=5.0):
    """
    Generate a Ball Pivoting Algorithm mesh from a point cloud.
    
    Args:
        ply_path: Path to the input .ply point cloud file
        ball_radius: Radius of the pivoting ball in mm (default 5.0mm)
    
    Returns:
        Tuple of (mesh_ply_path, mesh_obj_path) or (None, None) on failure
    """
    # Lazy load Open3D
    o3d = get_open3d()
    if o3d is None:
        print("⚠️  Open3D not installed - skipping mesh generation")
        print("   Install with: pip install open3d")
        return None, None
    
    try:
        print(f"\n🔨 Generating Ball Pivoting mesh...")
        
        # Load point cloud
        pcd = o3d.io.read_point_cloud(str(ply_path))
        num_points = len(pcd.points)
        
        if num_points < 100:
            print(f"⚠️  Too few points ({num_points}) for mesh generation")
            return None, None
        
        print(f"   Loaded {num_points:,} points")
        
        # Estimate normals
        print(f"   Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=ball_radius * 1.5,
                max_nn=30
            )
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=10)
        
        # Ball Pivoting reconstruction
        print(f"   Running Ball Pivoting (radius={ball_radius}mm)...")
        radii = [ball_radius, ball_radius * 1.5, ball_radius * 2.0]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        print(f"   Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
        
        # Save mesh files
        base_path = ply_path.parent / ply_path.stem
        mesh_ply_path = Path(str(base_path) + "_bpa.ply")
        mesh_obj_path = Path(str(base_path) + "_bpa.obj")
        
        # Save as PLY and OBJ
        o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
        o3d.io.write_triangle_mesh(str(mesh_obj_path), mesh)
        
        print(f"   ✓ Saved BPA mesh to:")
        print(f"     PLY: {mesh_ply_path.name}")
        print(f"     OBJ: {mesh_obj_path.name}")
        
        return mesh_ply_path, mesh_obj_path
        
    except Exception as e:
        print(f"❌ BPA mesh generation failed: {e}")
        return None, None


def check_system_requirements():
    """
    Check if all required dependencies and system requirements are met.
    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "="*80)
    print("🔍 SYSTEM REQUIREMENTS CHECK")
    print("="*80)
    
    all_checks_passed = True
    
    # 1. Python version check
    import sys
    import platform
    python_version = sys.version_info
    print(f"\n📌 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print(f"   ❌ Python 3.8+ required (you have {python_version.major}.{python_version.minor}.{python_version.micro})")
        all_checks_passed = False
    else:
        print("   ✅ Python version OK")
    
    # 2. System info (CPU, RAM, OS)
    print(f"\n📌 System Info")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Machine: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"   RAM: {ram_gb:.2f} GB total, {ram_available_gb:.2f} GB available")
        if ram_gb < 4:
            print("   ⚠️  Less than 4GB RAM - may run slow")
        else:
            print("   ✅ RAM sufficient")
    except ImportError:
        print("   ⚠️  psutil not installed (cannot check RAM)")
        print("   Install: pip install psutil")
    
    # CPU cores
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"   CPU Cores: {cpu_count}")
        if cpu_count < 2:
            print("   ⚠️  Only 1 CPU core - processing may be slow")
        else:
            print("   ✅ Multi-core CPU detected")
    except:
        print("   ⚠️  Cannot detect CPU cores")
    
    # 3. OpenCV check
    print("\n📌 OpenCV (cv2)")
    try:
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__} installed")
        
        # Check for CUDA support
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"   ✅ CUDA GPU support available ({cv2.cuda.getCudaEnabledDeviceCount()} device(s))")
        else:
            print("   ℹ️  No CUDA GPU support (CPU only)")
    except ImportError:
        print("   ❌ OpenCV not found")
        print("   Install: pip install opencv-python")
        all_checks_passed = False
    except:
        print("   ℹ️  OpenCV installed (no GPU support)")
    
    # 4. NumPy check
    print("\n📌 NumPy")
    try:
        import numpy as np
        print(f"   ✅ NumPy {np.__version__} installed")
    except ImportError:
        print("   ❌ NumPy not found")
        print("   Install: pip install numpy")
        all_checks_passed = False
    
    # 5. PyTorch check (for depth estimation)
    print("\n📌 PyTorch (for AI depth estimation)")
    torch_available = False
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} installed")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"   ✅ CUDA GPU support available")
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ️  No CUDA GPU (CPU only - depth estimation will be slower)")
        
        torch_available = True
    except ImportError:
        print("   ❌ PyTorch not found")
        print("   Install: pip install torch torchvision")
        print("   ⚠️  AI depth estimation will be DISABLED")
    
    # 6. torchvision check
    print("\n📌 torchvision (for depth estimation)")
    try:
        import torchvision
        print(f"   ✅ torchvision {torchvision.__version__} installed")
    except ImportError:
        print("   ❌ torchvision not found")
        print("   Install: pip install torchvision")
        print("   ⚠️  Required for depth estimation")
    
    # 7. timm check (required for DPT models)
    print("\n📌 timm (for MiDaS DPT models)")
    try:
        import timm
        print(f"   ✅ timm {timm.__version__} installed")
        print("   ✓ DPT_Large and DPT_Hybrid models supported")
    except ImportError:
        print("   ❌ timm not found")
        print("   Install: pip install timm")
        print("   ⚠️  Only MiDaS_small model will work without timm")
    
    # 8. MiDaS model accessibility check
    print("\n📌 MiDaS Depth Model")
    if torch_available:
        try:
            # Check if torch hub cache exists
            import torch
            hub_dir = torch.hub.get_dir()
            print(f"   ✓ Torch hub cache: {hub_dir}")
            
            # Check for cached models
            checkpoints_dir = Path(hub_dir) / "checkpoints"
            if checkpoints_dir.exists():
                midas_models = list(checkpoints_dir.glob("*midas*.pt"))
                if midas_models:
                    print(f"   ✅ Found {len(midas_models)} cached MiDaS model(s)")
                    for model in midas_models:
                        size_mb = model.stat().st_size / (1024**2)
                        print(f"      • {model.name} ({size_mb:.1f} MB)")
                else:
                    print("   ℹ️  No cached models - will download on first use")
                    print("   ⚠️  First run requires internet connection")
            else:
                print("   ℹ️  No model cache yet - will download on first use")
            
            print("   ✓ MiDaS will be downloaded from: https://github.com/intel-isl/MiDaS")
        except Exception as e:
            print(f"   ⚠️  Cannot check MiDaS status: {e}")
    else:
        print("   ❌ Cannot check - PyTorch not available")
        print("   ⚠️  Depth estimation features will be DISABLED")
    
    # 9. Open3D check (optional but recommended)
    print("\n📌 Open3D (optional - for mesh export)")
    o3d = get_open3d()  # Use lazy loader
    if o3d is not None:
        print(f"   ✅ Open3D {o3d.__version__} installed")
        print("   ✓ Mesh export (.ply) will be available")
    else:
        print("   ⚠️  Open3D not found (optional)")
        print("   Install for mesh export: pip install open3d")
        print("   ✓ Scanner will work without it (point clouds only)")
    
    # 10. Camera availability check
    print("\n📌 Camera Availability")
    try:
        import cv2
        test_cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            # Get camera properties
            width = test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            print(f"   ✅ Camera {WEBCAM_INDEX} accessible")
            print(f"   Resolution: {int(width)}x{int(height)} @ {int(fps)}fps")
            test_cap.release()
        else:
            test_cap = cv2.VideoCapture(WEBCAM_INDEX)
            if test_cap.isOpened():
                print(f"   ✅ Camera {WEBCAM_INDEX} accessible (without DSHOW)")
                test_cap.release()
            else:
                print(f"   ❌ Cannot access camera {WEBCAM_INDEX}")
                print("   Try: Change WEBCAM_INDEX or check camera connection")
                all_checks_passed = False
    except Exception as e:
        print(f"   ❌ Camera check failed: {e}")
        all_checks_passed = False
    
    # 11. Calibration file check
    print("\n📌 Camera Calibration")
    calibration_dir = Path(__file__).parent.parent / 'dual_checkerboard_3d' / 'calibration'
    if calibration_dir.exists():
        cal_files = list(calibration_dir.glob("camera_calibration*.npz"))
        if cal_files:
            print(f"   ✅ Found {len(cal_files)} calibration file(s)")
            for cal_file in cal_files:
                file_size_kb = cal_file.stat().st_size / 1024
                print(f"      • {cal_file.name} ({file_size_kb:.1f} KB)")
        else:
            print("   ❌ No calibration files found")
            print(f"   Location: {calibration_dir}")
            print("   Run: python dual_checkerboard_3d/checkerboard.py")
            all_checks_passed = False
    else:
        print(f"   ❌ Calibration directory not found: {calibration_dir}")
        print("   Run: python dual_checkerboard_3d/checkerboard.py")
        all_checks_passed = False
    
    # 12. Disk space check
    print("\n📌 Disk Space")
    try:
        import shutil
        data_dir = Path(__file__).parent / "data" / "point_clouds"
        stat = shutil.disk_usage(data_dir.parent if data_dir.exists() else Path.home())
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        print(f"   Total: {total_gb:.2f} GB")
        print(f"   Used: {used_gb:.2f} GB")
        print(f"   Free: {free_gb:.2f} GB")
        if free_gb < 1.0:
            print("   ⚠️  Less than 1GB free - may run out of space during scanning")
        elif free_gb < 5.0:
            print("   ⚠️  Less than 5GB free - recommended for large scans")
        else:
            print("   ✅ Sufficient disk space")
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
    
    # 13. Performance estimate
    print("\n📌 Performance Estimate")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = multiprocessing.cpu_count()
        
        if ram_gb >= 8 and cpu_count >= 4:
            print("   ✅ EXCELLENT - Fast scanning expected (8GB+ RAM, 4+ cores)")
        elif ram_gb >= 4 and cpu_count >= 2:
            print("   ✅ GOOD - Smooth scanning expected (4GB+ RAM, 2+ cores)")
        elif ram_gb >= 2:
            print("   ⚠️  ACCEPTABLE - May be slow (2-4GB RAM)")
        else:
            print("   ⚠️  LIMITED - Scanning may be very slow (<2GB RAM)")
    except:
        print("   ℹ️  Cannot estimate performance")
    
    # Summary
    print("\n" + "="*80)
    if all_checks_passed:
        print("✅ ALL SYSTEM CHECKS PASSED - Ready to scan!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        print("\nQuick fix commands:")
        print("  # Core dependencies:")
        print("  pip install opencv-python numpy psutil")
        print("  # For AI depth estimation:")
        print("  pip install torch torchvision timm")
        print("  # Optional features:")
        print("  pip install open3d  # (mesh export)")
        print("  # Calibration:")
        print("  python dual_checkerboard_3d/checkerboard.py")
    print("="*80 + "\n")
    
    return all_checks_passed

# ========== SECTION 4: MAIN SCANNER FUNCTION (Lines 1101-1450) ==========
# 🎯 THIS MUST COME **BEFORE** main()!

def scan_3d_points(project_dir=None):
    """Advanced 3D scanner with red laser beam detection + AI depth estimation."""
    global red_hue_min, red_hue_max, saturation_min, value_min
    global roi_enabled, roi_x1, roi_y1, roi_x2, roi_y2
    global info_box_visible, ai_panel_visible, cartoon_mode
    global auto_capture_mode, auto_capture_countdown, auto_capture_count
    global curve_sample_rate, corner_max_count, canny_threshold1, canny_threshold2
    
    # 🎨 Initialize Panel Display Module (lazy load)
    PanelDisplayModule = load_panel_display()
    panel_display = PanelDisplayModule(window_name="Bosch GLM 42 Scanner (635nm)")
    
    # ========================================
    # 🎯 CAMERA CALIBRATION WITH AUTO-SETUP
    # ========================================
    # Load camera calibration (lazy load)
    load_camera_calibration = load_calibration()
    calibration_result = load_camera_calibration()
    
    if calibration_result is None or calibration_result[0] is None:
        # No calibration found - offer setup wizard
        print("\n" + "="*70)
        print("⚠️  CAMERA CALIBRATION NOT DETECTED")
        print("="*70)
        
        from calibration_helper import setup_calibration, get_default_camera_matrix
        
        # Run calibration setup wizard
        should_calibrate = setup_calibration()
        
        if should_calibrate:
            # Try loading again after calibration
            calibration_result = load_camera_calibration()
            
            if calibration_result is None or calibration_result[0] is None:
                print("\n⚠️  Calibration not completed - using default parameters")
                camera_matrix, dist_coeffs = get_default_camera_matrix(1280, 720)
            else:
                camera_matrix, dist_coeffs = calibration_result
                print(f"\n✓ Calibration loaded successfully")
        else:
            # User chose to skip calibration
            print("\n⚠️  Using DEFAULT camera parameters (uncalibrated)")
            camera_matrix, dist_coeffs = get_default_camera_matrix(1280, 720)
    else:
        camera_matrix, dist_coeffs = calibration_result
        print(f"\n✓ Calibration loaded successfully")
    
    # ========================================
    
    # Initialize camera FIRST (needed for camera detection)
    print("\n🔍 Opening camera...")
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("   Trying fallback method...")
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✓ Camera opened")
    
    # ========================================
    # 🎯 CAMERA IDENTIFICATION & VERIFICATION
    # ========================================
    from camera_identifier import (
        get_camera_fingerprint, 
        check_camera_match, 
        save_camera_fingerprint,
        prompt_recalibration
    )
    
    # Get current camera fingerprint
    current_camera_fp = get_camera_fingerprint(cap)
    
    # Check if camera matches calibration (if calibration exists)
    if camera_matrix is not None:
        # Find calibration file path
        calibration_dir = Path(__file__).parent.parent / 'dual_checkerboard_3d' / 'calibration'
        calib_files = list(calibration_dir.glob("camera_calibration*.npz"))
        
        if calib_files:
            calibration_path = calib_files[0]
            
            # Check camera match
            match, confidence, message = check_camera_match(cap, calibration_path)
            
            print("\n" + "=" * 70)
            print("📷 CAMERA VERIFICATION")
            print("=" * 70)
            print(message)
            
            if match is False:  # Definite mismatch
                # Offer to recalibrate
                if prompt_recalibration():
                    print("\n🔄 Starting recalibration for new camera...")
                    
                    from calibration_helper import setup_calibration
                    setup_calibration()
                    
                    # Reload calibration after recalibrating
                    calibration_result = load_camera_calibration()
                    if calibration_result is not None:
                        camera_matrix, dist_coeffs = calibration_result
                        
                        # Save fingerprint for new camera
                        save_camera_fingerprint(current_camera_fp, calibration_path)
                        print("✓ Calibration updated for new camera")
                else:
                    print("\n⚠️  WARNING: Using mismatched calibration - results will be inaccurate!")
            elif match is None:  # Legacy calibration (no fingerprint)
                # Save fingerprint for future checks
                print("  Saving camera fingerprint for future verification...")
                save_camera_fingerprint(current_camera_fp, calibration_path)
            else:  # Match confirmed
                pass  # Already printed confirmation
            
            print("=" * 70 + "\n")
    
    # ========================================
    
    # 🎨 Depth estimator variables
    depth_estimator = None
    depth_map = None
    show_depth_viz = False
    max_depth_m = 2.0
    min_depth_m = 0.2
    downsample = 2
    
    def load_depth_model():
        """Lazy load depth estimator when Mode 4 is activated."""
        nonlocal depth_estimator
        DepthEstimator_class = get_depth_estimator()  # Call global function
        if DepthEstimator_class is None:
            print("❌ Depth estimation not available")
            return None
        
        if depth_estimator is None:
            try:
                print("\n[AI] Loading depth model...")
                depth_estimator = DepthEstimator_class("small")
                print(f"✓ Depth model loaded")
            except Exception as e:
                print(f"❌ Depth model failed: {e}")
                return None
        return depth_estimator
    
    # Point cloud data
    points_3d = []
    points_colors = []
    point_angles = []
    point_sessions = []
    
    # Scanner state
    current_mode = MODE_LASER
    current_session = 0
    current_angle = 0.0
    capture_mode = CAPTURE_MODE_PHOTO  # 📸 Photo vs Point Cloud capture
    
    # Get optimal camera matrix
    h, w = 720, 1280
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Create window - use WINDOW_NORMAL for consistent sizing control
    window_name = "Bosch GLM 42 Scanner (635nm)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Get actual camera resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set window to exact camera size
    cv2.resizeWindow(window_name, actual_width, actual_height)
    
    # Center window on screen
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Calculate center position
        center_x = (screen_width - actual_width) // 2
        center_y = (screen_height - actual_height) // 2
        
        cv2.moveWindow(window_name, center_x, center_y)
        print(f"✓ Window centered at ({center_x}, {center_y})")
    except Exception as e:
        print(f"ℹ️  Could not center window: {e}")
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Initialize modules - DEFAULT: Full Spectrum (lazy load)
    SpectrumAnalyzer = load_spectrum_analyzer()
    analyzer = SpectrumAnalyzer(wavelength_nm=None)  # None = Full Spectrum mode
    
    # Available spectrum presets for cycling
    spectrum_presets = [
        {'wavelength': None, 'name': 'Full Spectrum'},
        {'wavelength': 635, 'name': 'Red (Bosch 635nm)'},
        {'wavelength': 532, 'name': 'Green Laser'},
        {'wavelength': 450, 'name': 'Blue Laser'},
        {'wavelength': 780, 'name': 'Near-IR'},
    ]
    current_spectrum_idx = 0  # Start with Full Spectrum
    
    # ========== GPU OPTIMIZER - ONLY IF CUDA AVAILABLE ==========
    gpu_opt = None  # Default to None (CPU mode)
    
    try:
        # Check if CUDA is available
        cuda_available = False
        try:
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except AttributeError:
            # cv2.cuda module doesn't exist (OpenCV compiled without CUDA)
            cuda_available = False
        except Exception:
            # Any other CUDA check error
            cuda_available = False
        
        if cuda_available:
            # CUDA GPU found - try to initialize GPU optimizer (lazy load)
            try:
                profiler, GPUOptimizer = load_performance_modules()
                gpu_opt = GPUOptimizer()
                
                # Try different initialization methods
                if hasattr(gpu_opt, 'initialize'):
                    gpu_opt.initialize(camera_matrix, dist_coeffs, (w, h))
                elif hasattr(gpu_opt, 'setup'):
                    gpu_opt.setup(camera_matrix, dist_coeffs, (w, h))
                
                print(f"✓ GPU optimizer initialized (CUDA: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s))")
            except Exception as gpu_error:
                print(f"⚠️  GPU optimizer failed: {gpu_error}")
                print("   Falling back to CPU undistortion")
                gpu_opt = None
        else:
            # No CUDA - use CPU mode
            print("ℹ️  No CUDA GPU detected - using CPU undistortion")
            try:
                import multiprocessing
                cpu_cores = multiprocessing.cpu_count()
                print(f"   CPU mode: {cpu_cores} cores available")
            except:
                pass
    
    except Exception as e:
        print(f"⚠️  GPU check error: {e}")
        print("   Using CPU undistortion")
        gpu_opt = None
    
    # Mesh generation method
    mesh_method = "POISSON"  # Options: "POISSON" or "BPA"
    
    mode_names = {
        MODE_LASER: "RED LASER",
        MODE_CURVE: "CURVE TRACE",
        MODE_CORNERS: "CORNERS",
        MODE_DEPTH: "AI DEPTH"
    }
    
    print("\n" + "="*70)
    print("CONTROLS:")
    print("  1/2/3/4 - Mode switch")
    print("  SPACE   - Capture (points/photo/cloud based on mode)")
    print("  t       - Toggle capture mode (Photo <-> Point Cloud)")
    print("  o       - Open 3D viewer")
    print("  s       - Save PLY + mesh")
    print("  v       - Toggle depth viz (mode 4)")
    print("  p       - Toggle density (mode 4)")
    print("  +/-     - Curve sample rate")
    print("  w/e     - Min depth range (mode 4)")
    print("  q/ESC   - Quit")
    print("="*70)
    
    # ========== MAIN LOOP ==========
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Undistort
        undistorted = gpu_opt.undistort_frame(frame) if gpu_opt else frame
        h, w = undistorted.shape[:2]
        display_frame = undistorted.copy()
        
        # ========== MODE-SPECIFIC RENDERING =========
        
        # LASER MODE
        if current_mode == MODE_LASER:
            dot_x, dot_y, dot_area, all_dots, combined_mask, color_mask, bright_mask = \
                detect_laser_with_spectrum(
                    undistorted, analyzer, DEFAULT_BRIGHTNESS, 
                    LASER_MIN_AREA, LASER_MAX_AREA,
                    saturation_min, value_min
                )
            
            if dot_x and dot_y:
                cv2.drawMarker(display_frame, (dot_x, dot_y), (0, 255, 0), 
                              cv2.MARKER_CROSS, 30, 3)
                distance_cm = estimate_distance_linear(dot_y)
                cv2.putText(display_frame, f"{distance_cm:.1f}cm", 
                           (dot_x + 15, dot_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # CURVE MODE
        elif current_mode == MODE_CURVE:
            curves, edges = detect_curves(undistorted)
            cv2.drawContours(display_frame, curves, -1, (0, 255, 255), 2)
        
        # CORNER MODE
        elif current_mode == MODE_CORNERS:
            corner_points, corners_img = detect_corners(undistorted)
            for (px, py) in corner_points[:corner_max_count]:
                cv2.circle(display_frame, (int(px), int(py)), 4, (255, 0, 255), -1)
        
        # DEPTH MODE
        elif current_mode == MODE_DEPTH:
            if show_depth_viz and depth_map is not None:
                estimator = load_depth_model()
                if estimator:
                    depth_colored = estimator.visualize_depth(depth_map)
                    depth_small = cv2.resize(depth_colored, (w//3, h//3))
                    overlay_y = h - h//3 - 10
                    overlay_x = w - w//3 - 10
                    display_frame[overlay_y:overlay_y+h//3, overlay_x:overlay_x+w//3] = depth_small
                    cv2.rectangle(display_frame, (overlay_x, overlay_y),
                                (overlay_x+w//3, overlay_y+h//3), (0, 255, 255), 2)
        
        # Draw ROI rectangle if enabled
        if roi_enabled:
            cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "ROI ACTIVE", (roi_x1, roi_y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw ROI selection in progress
        if roi_selecting:
            cv2.rectangle(display_frame, (roi_start_x, roi_start_y), (roi_x2, roi_y2), (255, 255, 0), 2)
        
        # ========================================
        # 🎯 CONSOLIDATED TOP-RIGHT STATUS PANEL
        # ========================================
        # This keeps all critical info visible and unobstructed by hidable panels
        mesh_label = "Poisson" if mesh_method == "POISSON" else "BPA"
        capture_label = "POINT CLOUD" if capture_mode == CAPTURE_MODE_POINTCLOUD else "PHOTO"
        
        # Build status lines
        status_lines = []
        status_lines.append(f"Mode: {mode_names[current_mode]}")
        status_lines.append(f"Points: {len(points_3d):,}")
        status_lines.append(f"Mesh: {mesh_label}")
        status_lines.append(f"Capture: {capture_label}")  # NEW: Show active capture mode
        
        # Spectrum info
        spectrum_name = spectrum_presets[current_spectrum_idx]['name']
        status_lines.append(f"Spectrum: {spectrum_name}")
        
        # Mode-specific info
        if current_mode == MODE_LASER:
            status_lines.append("[SPACE] Capture Point")
        elif current_mode == MODE_CURVE:
            curves, _ = detect_curves(undistorted)
            status_lines.append(f"Curves: {len(curves)}")
            status_lines.append("[SPACE] Capture")
        elif current_mode == MODE_CORNERS:
            status_lines.append(f"Corners: {len(corner_points)}")
            status_lines.append("[SPACE] Capture")
        elif current_mode == MODE_DEPTH:
            status_lines.append(f"Range: {min_depth_m:.1f}-{max_depth_m:.1f}m")
            status_lines.append(f"Down: {downsample}x")
            status_lines.append("[SPACE] Capture 3D")
        
        # Control hints
        status_lines.append("")  # Blank line
        status_lines.append("[P] Spectrum")
        
        # Draw consolidated panel in TOP-RIGHT corner
        panel_width = 280
        panel_height = 25 + (len(status_lines) * 30)
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 50, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Border
        cv2.rectangle(display_frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 255, 0), 2)
        
        # Draw text lines
        text_y = panel_y + 25
        for i, line in enumerate(status_lines):
            # Color coding: Green for status, Cyan for spectrum, Yellow for mode-specific, Gray for hints
            if i == 3:  # Spectrum line
                color = (255, 255, 0)  # Cyan
            elif i >= 4 and i < len(status_lines) - 2:  # Mode-specific
                color = (0, 255, 255)  # Yellow
            elif i >= len(status_lines) - 2:  # Control hints
                color = (180, 180, 180)  # Gray
            else:  # Status info
                color = (0, 255, 0)  # Green
                
            cv2.putText(display_frame, line, 
                       (panel_x + 10, text_y + (i * 30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 if i >= len(status_lines) - 2 else 0.6, 
                       color, 1 if i >= len(status_lines) - 2 else 2)
        
        # ========================================
        
        # 🎨 DRAW PANELS ON TOP
        scanner_params = {
            'current_angle': current_angle,
            'rotation_step': 30.0,
            'current_session': current_session,
            'points_count': len(points_3d),
            'current_mode': mode_names[current_mode],
            'sensitivity_info': {
                'curve_rate': curve_sample_rate,
                'corner_max': corner_max_count,
                'canny_low': canny_threshold1,
                'canny_high': canny_threshold2
            }
        }
        
        if current_mode == MODE_CURVE:
            curves, _ = detect_curves(undistorted)
            total_curve_points = sum(len(curve) for curve in curves)
            scanner_params['curves_info'] = {
                'count': len(curves),
                'points': total_curve_points
            }
        
        # 🎨 NEW: RUN AI ANALYSIS EVERY FRAME
        ai_result = None
        try:
            # Lazy load AI modules if not already loaded
            if not _ai_modules_loaded:
                load_ai_modules()
            
            # Get camera info and image quality
            camera_info = get_camera_info(undistorted, camera_matrix, cap)
            quality_info = analyze_image_quality(undistorted)
            
            # Combine results
            ai_result = {
                **camera_info,
                **quality_info
            }
        except Exception as e:
            # If AI analysis fails, create minimal info
            print(f"⚠️  AI analysis error: {e}")  # Debug
            ai_result = {
                'resolution': f"{w}x{h}",
                'focal_length': f"{camera_matrix[0,0]:.1f}px",
                'sharpness': 0.0,
                'brightness': 0.0,
                'status': 'Error'
            }
        
        # Draw all panels (NOW WITH AI DATA!)
        display_frame = panel_display.draw_all_panels(display_frame, scanner_params, ai_result)
        
        cv2.imshow(window_name, display_frame)
        
        # ========== KEYBOARD CONTROLS ==========
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Quit
            break
        
        # ===== MODE SWITCHING =====
        elif key == ord('1'):
            current_mode = MODE_LASER
            print(f"\n[MODE] {mode_names[current_mode]}")
        
        elif key == ord('2'):
            current_mode = MODE_CURVE
            print(f"\n[MODE] {mode_names[current_mode]}")
        
        elif key == ord('3'):
            current_mode = MODE_CORNERS
            print(f"\n[MODE] {mode_names[current_mode]}")
        
        elif key == ord('4'):
            # Try to load depth estimator first
            DepthEstimator = get_depth_estimator()
            if DEPTH_AVAILABLE:
                current_mode = MODE_DEPTH
                print(f"\n[MODE] {mode_names[current_mode]}")
            else:
                print("❌ Depth unavailable - install PyTorch, torchvision, timm")
        
        # ===== CAPTURE (ALL MODES) =====
        elif key == ord(' '):
            
            # CHECK CAPTURE MODE FIRST - Photo or Point Cloud capture
            if capture_mode == CAPTURE_MODE_PHOTO:
                # Save current video frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_filename = SAVE_DIRECTORY / f"photo_{timestamp}.jpg"
                cv2.imwrite(str(photo_filename), display_frame)
                print(f"📷 Photo saved: {photo_filename.name}")
                continue  # Skip normal point capture
            
            elif capture_mode == CAPTURE_MODE_POINTCLOUD:
                # Save point cloud 3D viewer screenshot
                if len(points_3d) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print("\n📸 Capturing point cloud view...")
                    
                    try:
                        o3d_module = get_open3d()
                        if o3d_module:
                            # Create point cloud object
                            pcd = o3d_module.geometry.PointCloud()
                            pcd.points = o3d_module.utility.Vector3dVector(points_3d)
                            if points_colors:
                                pcd.colors = o3d_module.utility.Vector3dVector(points_colors)
                            
                            # Set up visualizer for screenshot (headless)
                            vis = o3d_module.visualization.Visualizer()
                            vis.create_window(visible=False, width=1920, height=1080)
                            vis.add_geometry(pcd)
                            
                            # Add coordinate frame
                            mesh_frame = o3d_module.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                            vis.add_geometry(mesh_frame)
                            
                            # Set view
                            vis.poll_events()
                            vis.update_renderer()
                            
                            # Capture image
                            screenshot_path = SAVE_DIRECTORY / f"pointcloud_view_{timestamp}.png"
                            vis.capture_screen_image(str(screenshot_path))
                            vis.destroy_window()
                            
                            print(f"✓ Point cloud view saved: {screenshot_path.name}")
                        else:
                            print("❌ Open3D not available - cannot capture point cloud view")
                    except Exception as e:
                        print(f"❌ Error capturing point cloud: {e}")
                else:
                    print("⚠️  No points captured yet - scan some points first!")
                continue  # Skip normal point capture
            
            # NORMAL POINT CAPTURE MODE - proceed with regular scanning
            # QUALITY WARNING BEFORE CAPTURE (non-blocking)
            if ai_result:
                fps = ai_result.get('fps', 0)
                status = ai_result.get('status', 'Unknown')
                
                # Warn if quality is low (but don't block)
                if fps < 10:
                    print(f"\n⚠️  WARNING: Very low FPS ({fps:.1f}) - results may be inaccurate")
                elif status == 'Poor':
                    print(f"\n⚠️  WARNING: Poor image quality - consider improving lighting/focus")
                else:
                    print(f"\n✓ Quality check: FPS={fps:.1f}, Status={status}")
            
            # LASER MODE CAPTURE (single precise point)
            if current_mode == MODE_LASER:
                dot_x, dot_y, dot_area, all_dots, combined_mask, color_mask, bright_mask = \
                    detect_laser_with_spectrum(
                        undistorted, analyzer, DEFAULT_BRIGHTNESS, 
                        LASER_MIN_AREA, LASER_MAX_AREA,
                        saturation_min, value_min
                    )
                
                if dot_x and dot_y:
                    fx = new_camera_matrix[0, 0]
                    fy = new_camera_matrix[1, 1]
                    cx = new_camera_matrix[0, 2]
                    cy = new_camera_matrix[1, 2]
                    
                    distance_cm = estimate_distance_linear(dot_y)
                    z = distance_cm * 10
                    x = (dot_x - cx) * z / fx
                    y = (dot_y - cy) * z / fy
                    
                    points_3d.append([x, y, z])
                    color_bgr = undistorted[dot_y, dot_x]
                    color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
                    points_colors.append(color_rgb)
                    point_angles.append(current_angle)
                    point_sessions.append(current_session)
                    
                    print(f"✓ Laser: ({x:.1f}, {y:.1f}, {z:.1f})mm RGB{color_rgb}")
                else:
                    print("⚠️  No laser detected!")
            
            # CURVE MODE CAPTURE (3 snapshots with averaging)
            elif current_mode == MODE_CURVE:
                print("\n📸 Taking 3 snapshots for curves...")
                all_snapshot_points = []
                
                for snapshot_num in range(1, 4):
                    # Capture fresh frame
                    ret, frame_snap = cap.read()
                    if not ret:
                        print(f"  ⚠️  Snapshot {snapshot_num}/3: Frame capture failed")
                        continue
                    
                    # Undistort
                    undistorted_snap = cv2.undistort(frame_snap, camera_matrix, dist_coeffs, None, new_camera_matrix)
                    
                    # VISUAL PROGRESS BAR
                    progress_frame = undistorted_snap.copy()
                    bar_width = 400
                    bar_height = 30
                    bar_x = (w - bar_width) // 2
                    bar_y = h - 100
                    
                    cv2.rectangle(progress_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
                    cv2.rectangle(progress_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                    fill_width = int((snapshot_num / 3) * bar_width)
                    cv2.rectangle(progress_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
                    
                    progress_text = f"Curve Snapshot {snapshot_num}/3..."
                    text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = bar_x + (bar_width - text_size[0]) // 2
                    text_y = bar_y + (bar_height + text_size[1]) // 2
                    cv2.putText(progress_frame, progress_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow(window_name, progress_frame)
                    cv2.waitKey(1)
                    
                    # Detect curves in this snapshot
                    curves, _ = detect_curves(undistorted_snap)
                    snapshot_curve_points = []
                    
                    for curve in curves:
                        for point in curve[::curve_sample_rate]:
                            px, py = point[0]
                            py = max(0, min(int(py), h-1))
                            px = max(0, min(int(px), w-1))
                            
                            distance_cm = estimate_distance_linear(py)
                            z = distance_cm * 10
                            x = (px - new_camera_matrix[0, 2]) * z / new_camera_matrix[0, 0]
                            y = (py - new_camera_matrix[1, 2]) * z / new_camera_matrix[1, 1]
                            
                            color_bgr = undistorted_snap[py, px]
                            color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
                            
                            snapshot_curve_points.append(([x, y, z], color_rgb))
                    
                    all_snapshot_points.append(snapshot_curve_points)
                    print(f"  ✓ Snapshot {snapshot_num}/3: {len(snapshot_curve_points)} curve points")
                    
                    import time
                    time.sleep(0.1)
                
                # Show completion
                complete_frame = display_frame.copy()
                cv2.rectangle(complete_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
                cv2.putText(complete_frame, "Curve Capture Complete!", (bar_x + 50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window_name, complete_frame)
                cv2.waitKey(500)
                
                # Use points from the snapshot with most detections
                if all_snapshot_points:
                    best_snapshot = max(all_snapshot_points, key=len)
                    count_before = len(points_3d)
                    
                    for point_data in best_snapshot:
                        points_3d.append(point_data[0])
                        points_colors.append(point_data[1])
                        point_angles.append(current_angle)
                        point_sessions.append(current_session)
                    
                    added = len(points_3d) - count_before
                    print(f"\n✓ Curves: Added {added} points (from best of 3 snapshots)")
                else:
                    print("\n❌ No curves detected in any snapshot!")
            
            # CORNER MODE CAPTURE (3 snapshots with averaging)
            elif current_mode == MODE_CORNERS:
                print("\n📸 Taking 3 snapshots for corners...")
                all_snapshot_corners = []
                
                for snapshot_num in range(1, 4):
                    # Capture fresh frame
                    ret, frame_snap = cap.read()
                    if not ret:
                        print(f"  ⚠️  Snapshot {snapshot_num}/3: Frame capture failed")
                        continue
                    
                    # Undistort
                    undistorted_snap = cv2.undistort(frame_snap, camera_matrix, dist_coeffs, None, new_camera_matrix)
                    
                    # VISUAL PROGRESS BAR
                    progress_frame = undistorted_snap.copy()
                    bar_width = 400
                    bar_height = 30
                    bar_x = (w - bar_width) // 2
                    bar_y = h - 100
                    
                    cv2.rectangle(progress_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
                    cv2.rectangle(progress_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                    fill_width = int((snapshot_num / 3) * bar_width)
                    cv2.rectangle(progress_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
                    
                    progress_text = f"Corner Snapshot {snapshot_num}/3..."
                    text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = bar_x + (bar_width - text_size[0]) // 2
                    text_y = bar_y + (bar_height + text_size[1]) // 2
                    cv2.putText(progress_frame, progress_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow(window_name, progress_frame)
                    cv2.waitKey(1)
                    
                    # Detect corners in this snapshot
                    corner_points, _ = detect_corners(undistorted_snap)
                    snapshot_corner_points = []
                    
                    for (px, py) in corner_points[:corner_max_count]:
                        py = max(0, min(int(py), h-1))
                        px = max(0, min(int(px), w-1))
                        
                        distance_cm = estimate_distance_linear(py)
                        z = distance_cm * 10
                        x = (px - new_camera_matrix[0, 2]) * z / new_camera_matrix[0, 0]
                        y = (py - new_camera_matrix[1, 2]) * z / new_camera_matrix[1, 1]
                        
                        color_bgr = undistorted_snap[py, px]
                        color_rgb = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
                        
                        snapshot_corner_points.append(([x, y, z], color_rgb))
                    
                    all_snapshot_corners.append(snapshot_corner_points)
                    print(f"  ✓ Snapshot {snapshot_num}/3: {len(snapshot_corner_points)} corners")
                    
                    import time
                    time.sleep(0.1)
                
                # Show completion
                complete_frame = display_frame.copy()
                cv2.rectangle(complete_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
                cv2.putText(complete_frame, "Corner Capture Complete!", (bar_x + 40, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window_name, complete_frame)
                cv2.waitKey(500)
                
                # Use points from the snapshot with most detections
                if all_snapshot_corners:
                    best_snapshot = max(all_snapshot_corners, key=len)
                    count_before = len(points_3d)
                    
                    for point_data in best_snapshot:
                        points_3d.append(point_data[0])
                        points_colors.append(point_data[1])
                        point_angles.append(current_angle)
                        point_sessions.append(current_session)
                    
                    added = len(points_3d) - count_before
                    print(f"\n✓ Corners: Added {added} points (from best of 3 snapshots)")
                else:
                    print("\n❌ No corners detected in any snapshot!")
            
            # DEPTH MODE CAPTURE
            elif current_mode == MODE_DEPTH:
                estimator = load_depth_model()
                if estimator:
                    print("\n[DEPTH] Capturing...")
                    import time
                    start = time.perf_counter()
                    
                    try:
                        # Apply ROI if enabled
                        frame_to_process = undistorted
                        camera_matrix_roi = new_camera_matrix.copy()
                        
                        if roi_enabled:
                            # Crop to ROI region
                            x1 = max(0, min(roi_x1, w-1))
                            y1 = max(0, min(roi_y1, h-1))
                            x2 = max(x1+1, min(roi_x2, w))
                            y2 = max(y1+1, min(roi_y2, h))
                            
                            frame_to_process = undistorted[y1:y2, x1:x2].copy()
                            
                            # Adjust camera intrinsics for ROI offset
                            camera_matrix_roi = new_camera_matrix.copy()
                            camera_matrix_roi[0, 2] -= x1  # Adjust cx
                            camera_matrix_roi[1, 2] -= y1  # Adjust cy
                            
                            roi_width = x2 - x1
                            roi_height = y2 - y1
                            print(f"[DEPTH] Processing ROI: {roi_width}x{roi_height} at ({x1},{y1})")
                        
                        depth_map = estimator.estimate_depth(frame_to_process)
                        new_points, new_colors = estimator.depth_to_point_cloud(
                            frame_to_process, depth_map, camera_matrix_roi,
                            max_depth_m, min_depth_m, downsample
                        )
                        
                        points_3d.extend(new_points.tolist())
                        points_colors.extend(new_colors.tolist())
                        
                        for _ in range(len(new_points)):
                            point_angles.append(current_angle)
                            point_sessions.append(current_session)
                        
                        elapsed = time.perf_counter() - start
                        roi_msg = f" (ROI: {roi_width}x{roi_height})" if roi_enabled else ""
                        print(f"✓ Added {len(new_points):,} points in {elapsed:.2f}s{roi_msg}")
                        show_depth_viz = True
                        
                    except Exception as e:
                        print(f"❌ Error: {e}")
        
        # ===== TOGGLE CAPTURE MODE =====
        elif key == ord('t'):
            capture_mode = 1 - capture_mode  # Toggle between 0 and 1
            mode_name = "POINT CLOUD" if capture_mode == CAPTURE_MODE_POINTCLOUD else "PHOTO"
            print(f"📸 Capture mode: {mode_name}")
        
        # ===== 3D VIEWER =====
        elif key == ord('o'):
            if len(points_3d) > 0:
                print("\n[3D VIEWER] Opening visualization...")
                
                # Get actual camera resolution
                cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"[3D VIEWER] Camera resolution: {cam_width}x{cam_height}")
                
                # Lock the video window size BEFORE opening 3D viewer
                cv2.resizeWindow(window_name, cam_width, cam_height)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                # Open 3D viewer (positioned in top-right corner automatically)
                visualize_point_cloud_3d(
                    points_3d, points_colors, 
                    window_name=f"Scanner - {len(points_3d):,} points",
                    width=cam_width,
                    height=cam_height
                )
                
                print("[3D VIEWER] Closed - restoring scanner window...")
                
                # Re-lock window size and position
                cv2.resizeWindow(window_name, cam_width, cam_height)
                
                # Re-center window on screen
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    screen_width = root.winfo_screenwidth()
                    screen_height = root.winfo_screenheight()
                    root.destroy()
                    
                    center_x = (screen_width - cam_width) // 2
                    center_y = (screen_height - cam_height) // 2
                    cv2.moveWindow(window_name, center_x, center_y)
                except:
                    pass
                
                # Force refresh with new frames
                for i in range(10):
                    ret, frame = cap.read()
                    if ret:
                        temp_frame = gpu_opt.undistort_frame(frame) if gpu_opt else frame
                        cv2.imshow(window_name, temp_frame)
                        if cv2.waitKey(10) & 0xFF == 27:  # Allow ESC to break
                            break
                
                print(f"[3D VIEWER] Window restored and centered at {cam_width}x{cam_height}")
            else:
                print("⚠️  No points captured yet - scan some points first!")
        
        # ===== SAVE =====
        elif key == ord('s'):
            if len(points_3d) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scan_3d_{timestamp}.ply"
                
                # Use the global save directory set at startup
                output_dir = SAVE_DIRECTORY
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / filename
                
                print(f"\n💾 Saving {len(points_3d):,} points...")
                
                with open(output_path, 'w') as f:
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
                    
                    for i in range(len(points_3d)):
                        x, y, z = points_3d[i]
                        r, g, b = points_colors[i] if i < len(points_colors) else [128, 128, 128]
                        f.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")
                
                print(f"✓ Saved to {output_path}")
                
                # Automatically generate mesh using selected method
                if mesh_method == "POISSON":
                    generate_poisson_mesh(output_path, octree_depth=9)
                elif mesh_method == "BPA":
                    generate_bpa_mesh(output_path, ball_radius=5.0)
            else:
                print("⚠️  No points to save!")
        
        # ===== PANEL TOGGLES (NO CONFLICTS!) =====
        elif key == ord('b'):
            info_box_visible = panel_display.toggle_info_box()
            print(f"[PANEL] Controls: {'VISIBLE' if info_box_visible else 'HIDDEN'}")
        
        elif key == ord('i'):
            ai_panel_visible = panel_display.toggle_ai_panel()
            print(f"[PANEL] AI panel: {'VISIBLE' if ai_panel_visible else 'HIDDEN'}")
        
        # ===== CARTOON MODE (ORIGINAL KEY!) =====
        elif key == ord('v'):
            cartoon_mode = not cartoon_mode
            apply_cartoon_settings(cap, cartoon_mode)
            print(f"[CARTOON] {'ON' if cartoon_mode else 'OFF'}")
        
        # ===== MESH METHOD TOGGLE =====
        elif key == ord('m'):
            mesh_method = "BPA" if mesh_method == "POISSON" else "POISSON"
            method_desc = "Ball Pivoting (faithful to data)" if mesh_method == "BPA" else "Poisson (watertight/smooth)"
            print(f"[MESH] Method: {mesh_method} - {method_desc}")
        
        # ===== DEPTH MODE CONTROLS (NEW KEYS - NO CONFLICTS!) =====
        elif key == ord('z'):  # NEW: Depth visualization toggle
            if current_mode == MODE_DEPTH:
                show_depth_viz = not show_depth_viz
                print(f"[DEPTH VIZ] {'ON' if show_depth_viz else 'OFF'}")
        
        elif key == ord('x'):  # NEW: Depth density toggle
            if current_mode == MODE_DEPTH:
                downsample = 4 if downsample == 2 else 2
                print(f"[DEPTH] Downsample: {downsample}x ({'SPARSE' if downsample == 4 else 'DENSE'})")
        
        elif key == ord('w'):  # Depth min range UP
            if current_mode == MODE_DEPTH:
                min_depth_m = min(max_depth_m - 0.2, min_depth_m + 0.1)
                print(f"[DEPTH] Min: {min_depth_m:.1f}m")
        
        elif key == ord('e'):  # Depth min range DOWN
            if current_mode == MODE_DEPTH:
                min_depth_m = max(0.1, min_depth_m - 0.1)
                print(f"[DEPTH] Min: {min_depth_m:.1f}m")
        
        elif key == ord('r'):  # NEW: Depth max range UP
            if current_mode == MODE_DEPTH:
                max_depth_m = min(10.0, max_depth_m + 0.2)
                print(f"[DEPTH] Max: {max_depth_m:.1f}m")
        
        elif key == ord('f'):  # NEW: Depth max range DOWN
            if current_mode == MODE_DEPTH:
                max_depth_m = max(min_depth_m + 0.2, max_depth_m - 0.2)
                print(f"[DEPTH] Max: {max_depth_m:.1f}m")
        
        # ===== CURVE/CORNER/EDGE SENSITIVITY (ORIGINAL KEYS!) =====
        elif key == ord('+') or key == ord('='):
            curve_sample_rate = max(1, curve_sample_rate - 1)
            print(f"[CURVE] Sample rate: 1/{curve_sample_rate}")
        
        elif key == ord('-') or key == ord('_'):
            curve_sample_rate = min(20, curve_sample_rate + 1)
            print(f"[CURVE] Sample rate: 1/{curve_sample_rate}")
        
        elif key == ord('['):
            corner_max_count = max(10, corner_max_count - 10)
            print(f"[CORNERS] Max count: {corner_max_count}")
        
        elif key == ord(']'):
            corner_max_count = min(500, corner_max_count + 10)
            print(f"[CORNERS] Max count: {corner_max_count}")
        
        elif key == ord(',') or key == ord('<'):
            canny_threshold1 = max(10, canny_threshold1 - 10)
            canny_threshold2 = max(canny_threshold1 + 20, canny_threshold2 - 10)
            print(f"[EDGES] Thresholds: {canny_threshold1}/{canny_threshold2}")
        
        elif key == ord('.') or key == ord('>'):
            canny_threshold1 = min(200, canny_threshold1 + 10)
            canny_threshold2 = min(300, canny_threshold2 + 10)
            print(f"[EDGES] Thresholds: {canny_threshold1}/{canny_threshold2}")
        
        # ========== SPECTRUM ANALYZER CONTROLS ==========
        elif key == ord('w') and key != ord('w'):  # Prevent conflict with depth controls
            pass  # Depth control handled above
        
        elif key == ord('p'):  # Cycle through spectrum presets
            current_spectrum_idx = (current_spectrum_idx + 1) % len(spectrum_presets)
            preset = spectrum_presets[current_spectrum_idx]
            SpectrumAnalyzer = load_spectrum_analyzer()
            analyzer = SpectrumAnalyzer(wavelength_nm=preset['wavelength'])
            print(f"\n🌈 [SPECTRUM] {preset['name']}")
            print(f"   {analyzer.get_info_text()}")
        
        elif key == ord('g'):  # Show spectrum guide
            SpectrumAnalyzer.show_spectrum_guide()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Scanner closed")


# ========== SECTION 5: MAIN ENTRY POINT (Lines 1451-1470) ==========
# 🎯 THIS MUST COME **AFTER** scan_3d_points()!

def main():
    """Main entry point for the scanner."""
    # Check system requirements FIRST
    if not check_system_requirements():
        print("\n⚠️  System requirements not met!")
        print("Fix the issues above and try again.\n")
        response = input("Continue anyway? (y/N): ")
        
        if response.strip().lower() != 'y':
            print("Exiting...")
            sys.exit(1)
        
        print("\n⚠️  Proceeding without all requirements - may crash!\n")
    
    parser = argparse.ArgumentParser(description="Advanced 3D Scanner for Bosch GLM 42 (635nm)")
    parser.add_argument('--project', type=str, help='Project directory for saving scans')
    args = parser.parse_args()
    
    project_dir = args.project if args.project else None
    
    print("\n" + "="*80)
    print("🚀 STARTING BOSCH GLM 42 SCANNER")
    print("="*80)
    
    scan_3d_points(project_dir)  # ✅ NOW THIS WILL WORK!


if __name__ == "__main__":
    main()



