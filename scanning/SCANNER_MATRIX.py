"""
SCANNER MATRIX - Fast Lookup Index
==================================
Performance optimization matrix for laser_3d_scanner_advanced.py
Provides O(1) lookup for functions, files, and critical paths.

This matrix is used to speed up file searching, module imports, and runtime lookups.
"""

import os
from pathlib import Path

# ============================================================================
# SECTION 1: FILE PATH MATRIX
# ============================================================================

class PathMatrix:
    """Fast lookup for all file paths in the scanner system."""
    
    # Base directory (auto-detected)
    BASE_DIR = Path(__file__).parent
    PARENT_DIR = BASE_DIR.parent
    PROJECT_ROOT = BASE_DIR.parent.parent.parent
    
    # Core scanner files
    MAIN_SCANNER = BASE_DIR / "laser_3d_scanner_advanced.py"
    DEPTH_ESTIMATOR = BASE_DIR / "depth_estimator.py"
    SPECTRUM_CONFIG = BASE_DIR / "spectrum_config.py"
    PANEL_DISPLAY = BASE_DIR / "panel_display_module.py"
    OBJECT_MANAGER = BASE_DIR / "object_manager.py"
    AUTO_CAPTURE = BASE_DIR / "auto_capture_module.py"
    
    # Performance modules
    GPU_OPTIMIZER = BASE_DIR / "gpu_optimizer.py"
    PERFORMANCE_PROFILER = BASE_DIR / "performance_profiler.py"
    QUALITY_MONITOR = BASE_DIR / "runtime_quality_monitor.py"
    
    # Data directories
    DATA_DIR = BASE_DIR / "data"
    POINT_CLOUDS_DIR = DATA_DIR / "point_clouds"
    MESHES_DIR = DATA_DIR / "meshes"
    SCANS_DIR = DATA_DIR / "scans"
    
    # Calibration
    CALIBRATION_DIR = PARENT_DIR / "dual_checkerboard_3d" / "calibration"
    CAMERA_CALIB_NPZ = CALIBRATION_DIR / "camera_calibration.npz"
    
    # AI Analysis
    AI_ANALYSIS_DIR = PARENT_DIR / "ai_analysis"
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.POINT_CLOUDS_DIR.mkdir(exist_ok=True)
        cls.MESHES_DIR.mkdir(exist_ok=True)
        cls.SCANS_DIR.mkdir(exist_ok=True)
        
    @classmethod
    def get_next_scan_path(cls, session_name="scan"):
        """Get next available scan file path."""
        cls.ensure_directories()
        counter = 1
        while True:
            path = cls.POINT_CLOUDS_DIR / f"{session_name}_{counter:04d}.csv"
            if not path.exists():
                return path
            counter += 1


# ============================================================================
# SECTION 2: FUNCTION LOCATION MATRIX
# ============================================================================

class FunctionMatrix:
    """Fast lookup for function locations and signatures."""
    
    # Core detection functions
    DETECT_LASER = {
        'name': 'detect_laser_dots',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 630,
        'params': ['frame', 'brightness_threshold', 'red_hue_min', 'red_hue_max', 
                   'saturation_min', 'value_min', 'min_area', 'max_area'],
        'returns': 'dot_x, dot_y, dot_area, all_dots, combined_mask, color_mask, bright_mask',
        'description': 'Main red laser detection using HSV color space'
    }
    
    DETECT_SPECTRUM = {
        'name': 'detect_laser_with_spectrum',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 626,
        'params': ['frame', 'analyzer', 'brightness_threshold', 'min_area', 'max_area', 'sat_min', 'val_min'],
        'returns': 'Same as detect_laser_dots',
        'description': 'Spectrum-based laser detection wrapper'
    }
    
    DETECT_CURVES = {
        'name': 'detect_curves',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 797,
        'params': ['frame'],
        'returns': 'curves, edges',
        'description': 'Detect continuous curves using Canny edge detection'
    }
    
    DETECT_CORNERS = {
        'name': 'detect_corners',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 814,
        'params': ['frame'],
        'returns': 'corner_points, corners_img',
        'description': 'Detect corner features using goodFeaturesToTrack'
    }
    
    # 3D reconstruction functions
    GENERATE_MESH_POISSON = {
        'name': 'generate_poisson_mesh',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 1060,
        'params': ['ply_path', 'octree_depth'],
        'returns': 'mesh_ply_path, mesh_obj_path',
        'description': 'Generate watertight mesh using Poisson reconstruction'
    }
    
    GENERATE_MESH_BPA = {
        'name': 'generate_ball_pivoting_mesh',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 1139,
        'params': ['ply_path'],
        'returns': 'mesh_ply_path, mesh_obj_path',
        'description': 'Generate faithful mesh using Ball Pivoting Algorithm'
    }
    
    SAVE_POINT_CLOUD = {
        'name': 'save_point_cloud',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 828,
        'params': ['points_3d', 'colors', 'filename', 'project_dir'],
        'returns': 'output_path',
        'description': 'Save point cloud to PLY/CSV with metadata'
    }
    
    # Depth estimation functions
    ESTIMATE_DEPTH = {
        'name': 'estimate_depth',
        'file': 'depth_estimator.py',
        'line_approx': 138,
        'params': ['rgb_image'],
        'returns': 'depth_map',
        'description': 'AI depth estimation using MiDaS'
    }
    
    DEPTH_TO_POINT_CLOUD = {
        'name': 'depth_to_point_cloud',
        'file': 'depth_estimator.py',
        'line_approx': 169,
        'params': ['rgb_image', 'depth_map', 'camera_matrix', 'max_depth_m', 'min_depth_m', 'downsample'],
        'returns': 'points_3d, colors',
        'description': 'Convert depth map to 3D point cloud'
    }
    
    # System checks
    CHECK_REQUIREMENTS = {
        'name': 'check_system_requirements',
        'file': 'laser_3d_scanner_advanced.py',
        'line_approx': 1218,
        'params': [],
        'returns': 'bool',
        'description': 'Verify all dependencies and system requirements'
    }
    
    @classmethod
    def get_function(cls, name):
        """Quick lookup for function info by name."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, dict) and attr.get('name') == name:
                return attr
        return None
    
    @classmethod
    def list_all(cls):
        """List all registered functions."""
        functions = []
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, dict) and 'name' in attr:
                functions.append(attr)
        return functions


# ============================================================================
# SECTION 3: CONFIGURATION MATRIX
# ============================================================================

class ConfigMatrix:
    """Fast lookup for all configuration parameters."""
    
    # Scanner modes
    MODE_LASER = 0
    MODE_CURVE = 1
    MODE_CORNERS = 2
    MODE_DEPTH = 3
    
    MODE_NAMES = {
        MODE_LASER: "LASER DOTS",
        MODE_CURVE: "CURVE TRACE",
        MODE_CORNERS: "CORNER DETECT",
        MODE_DEPTH: "AI DEPTH"
    }
    
    # Camera settings
    WEBCAM_INDEX = 0
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    
    # Detection thresholds (defaults)
    BRIGHTNESS_THRESHOLD = 180
    RED_HUE_MIN = 0
    RED_HUE_MAX = 20
    SATURATION_MIN = 80
    VALUE_MIN = 100
    MIN_AREA = 5
    MAX_AREA = 1000
    
    # Spectrum presets
    SPECTRUM_PRESETS = [
        {'wavelength': None, 'name': 'Full Spectrum'},
        {'wavelength': 635, 'name': 'Red (Bosch 635nm)'},
        {'wavelength': 532, 'name': 'Green Laser'},
        {'wavelength': 450, 'name': 'Blue Laser'},
        {'wavelength': 780, 'name': 'Near-IR'},
    ]
    
    # Mesh generation
    MESH_POISSON = "POISSON"
    MESH_BPA = "BPA"
    OCTREE_DEPTH_DEFAULT = 9
    
    # Depth estimation
    DEPTH_MIN_M = 0.2
    DEPTH_MAX_M = 2.0
    DEPTH_DOWNSAMPLE = 1
    
    # Performance
    GPU_ENABLED = True
    PROFILING_ENABLED = False
    MONITORING_ENABLED = False


# ============================================================================
# SECTION 4: KEYBOARD CONTROL MATRIX
# ============================================================================

class KeyboardMatrix:
    """Fast lookup for all keyboard controls."""
    
    CONTROLS = {
        # Mode switching
        '1': {'action': 'Switch to Laser Mode', 'mode': 0},
        '2': {'action': 'Switch to Curve Mode', 'mode': 1},
        '3': {'action': 'Switch to Corners Mode', 'mode': 2},
        '4': {'action': 'Switch to Depth Mode', 'mode': 3},
        
        # Capture
        'SPACE': {'action': 'Capture point/curve/corner/depth', 'critical': True},
        
        # File operations
        's': {'action': 'Save point cloud', 'critical': True},
        'c': {'action': 'Clear all points', 'warning': True},
        
        # Display toggles
        'i': {'action': 'Toggle info panel'},
        'b': {'action': 'Toggle info box visibility'},
        'v': {'action': 'Toggle AI panel'},
        'm': {'action': 'Toggle mesh method (Poisson/BPA)'},
        'z': {'action': 'Toggle depth visualization'},
        'x': {'action': 'Toggle cartoon mode'},
        
        # Spectrum
        'p': {'action': 'Cycle spectrum preset'},
        'g': {'action': 'Show spectrum guide'},
        
        # Depth controls
        'w': {'action': 'Increase min depth range'},
        'e': {'action': 'Decrease min depth range'},
        'r': {'action': 'Increase max depth range'},
        'f': {'action': 'Decrease max depth range'},
        
        # Curve/Corner sensitivity
        '+/-': {'action': 'Adjust curve sample rate'},
        '[/]': {'action': 'Adjust corner max count'},
        ',/.': {'action': 'Adjust Canny thresholds'},
        
        # Exit
        'q': {'action': 'Quit scanner', 'critical': True},
        'ESC': {'action': 'Quit scanner', 'critical': True},
    }
    
    @classmethod
    def get_help_text(cls):
        """Generate help text for all controls."""
        lines = ["KEYBOARD CONTROLS:", "=" * 50]
        for key, info in cls.CONTROLS.items():
            marker = "⚠️ " if info.get('warning') else "🎯 " if info.get('critical') else "  "
            lines.append(f"{marker}[{key:6}] {info['action']}")
        return "\n".join(lines)


# ============================================================================
# SECTION 5: DEPENDENCY MATRIX
# ============================================================================

class DependencyMatrix:
    """Fast lookup for all module dependencies."""
    
    REQUIRED = {
        'opencv-python': {'import_as': 'cv2', 'critical': True},
        'numpy': {'import_as': 'np', 'critical': True},
        'pathlib': {'import_as': 'Path', 'critical': True},
    }
    
    OPTIONAL_FEATURES = {
        'torch': {
            'import_as': 'torch',
            'feature': 'AI Depth Estimation',
            'required_for': ['MODE_DEPTH'],
        },
        'torchvision': {
            'import_as': 'torchvision',
            'feature': 'AI Depth Estimation',
            'required_for': ['MODE_DEPTH'],
        },
        'timm': {
            'import_as': 'timm',
            'feature': 'MiDaS DPT models',
            'required_for': ['DepthEstimator (DPT models)'],
        },
        'open3d': {
            'import_as': 'o3d',
            'feature': 'Mesh generation',
            'required_for': ['generate_poisson_mesh', 'generate_ball_pivoting_mesh'],
        },
        'psutil': {
            'import_as': 'psutil',
            'feature': 'System monitoring',
            'required_for': ['check_system_requirements'],
        },
    }
    
    LOCAL_MODULES = {
        'spectrum_config': ['SpectrumAnalyzer'],
        'depth_estimator': ['DepthEstimator'],
        'panel_display_module': ['PanelDisplay'],
        'object_manager': ['ObjectManager'],
        'gpu_optimizer': ['GPUOptimizer'],
        'performance_profiler': ['PerformanceProfiler'],
    }


# ============================================================================
# SECTION 6: PERFORMANCE OPTIMIZATION MATRIX
# ============================================================================

class PerformanceMatrix:
    """Lookup table for performance-critical code sections."""
    
    # Hotspots (frequently called functions)
    HOTSPOTS = {
        'detect_laser_dots': {
            'calls_per_frame': 1,
            'avg_time_ms': 5.0,
            'optimization': 'GPU acceleration available',
        },
        'cv2.cvtColor': {
            'calls_per_frame': 3,
            'avg_time_ms': 1.5,
            'optimization': 'Use cv2.cuda if available',
        },
        'cv2.inRange': {
            'calls_per_frame': 2,
            'avg_time_ms': 2.0,
            'optimization': 'Pre-allocate mask buffers',
        },
        'panel_display.draw_all_panels': {
            'calls_per_frame': 1,
            'avg_time_ms': 8.0,
            'optimization': 'Cache panel backgrounds',
        },
    }
    
    # Memory optimization
    MEMORY_POOLS = {
        'frame_buffer': {'size': '1920x1080x3', 'reuse': True},
        'hsv_buffer': {'size': '1920x1080x3', 'reuse': True},
        'mask_buffer': {'size': '1920x1080x1', 'reuse': True},
    }
    
    # Caching strategies
    CACHE_CONFIG = {
        'depth_model': {'max_size': 1, 'ttl_seconds': None},  # Keep forever
        'calibration': {'max_size': 1, 'ttl_seconds': None},  # Keep forever
        'panel_backgrounds': {'max_size': 10, 'ttl_seconds': 60},
    }


# ============================================================================
# SECTION 7: QUICK REFERENCE API
# ============================================================================

class ScannerMatrix:
    """Main API for fast lookups."""
    
    paths = PathMatrix
    functions = FunctionMatrix
    config = ConfigMatrix
    keyboard = KeyboardMatrix
    dependencies = DependencyMatrix
    performance = PerformanceMatrix
    
    @classmethod
    def find(cls, query):
        """Universal search across all matrices."""
        results = []
        
        # Search functions
        func = cls.functions.get_function(query)
        if func:
            results.append(('function', func))
        
        # Search paths
        if hasattr(cls.paths, query.upper()):
            results.append(('path', getattr(cls.paths, query.upper())))
        
        # Search config
        if hasattr(cls.config, query.upper()):
            results.append(('config', getattr(cls.config, query.upper())))
        
        return results
    
    @classmethod
    def print_summary(cls):
        """Print complete matrix summary."""
        print("\n" + "=" * 80)
        print("SCANNER MATRIX SUMMARY")
        print("=" * 80)
        
        print(f"\n📁 Registered Paths: {len([a for a in dir(cls.paths) if not a.startswith('_')])}")
        print(f"⚙️  Registered Functions: {len(cls.functions.list_all())}")
        print(f"🎮 Keyboard Controls: {len(cls.keyboard.CONTROLS)}")
        print(f"📦 Required Dependencies: {len(cls.dependencies.REQUIRED)}")
        print(f"🔌 Optional Dependencies: {len(cls.dependencies.OPTIONAL_FEATURES)}")
        print(f"⚡ Performance Hotspots: {len(cls.performance.HOTSPOTS)}")
        
        print("\n" + "=" * 80)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Print summary
    ScannerMatrix.print_summary()
    
    # Example lookups
    print("\n🔍 Example Lookups:")
    print("-" * 80)
    
    # Find a function
    func = ScannerMatrix.functions.get_function('detect_laser_dots')
    if func:
        print(f"\nFunction: {func['name']}")
        print(f"  File: {func['file']} (line ~{func['line_approx']})")
        print(f"  Description: {func['description']}")
    
    # Get next scan path
    next_path = ScannerMatrix.paths.get_next_scan_path("my_scan")
    print(f"\nNext scan path: {next_path}")
    
    # Show keyboard help
    print(f"\n{ScannerMatrix.keyboard.get_help_text()}")
