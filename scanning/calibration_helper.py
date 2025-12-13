"""
Camera Calibration Helper
=========================
Assists users with camera calibration setup when no calibration file is found.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Standard checkerboard dimensions (auto-filled)
DEFAULT_CHECKERBOARD = {
    'rows': 7,  # Internal corners (9x7 pattern = 8 rows, 6 cols internal)
    'cols': 10,
    'square_size_mm': 25.0,  # 25mm squares (standard)
    'pattern_type': 'CHECKERBOARD',
    'description': 'Standard 9x7 checkerboard (25mm squares)'
}

def generate_checkerboard_pattern(output_path=None, dpi=300):
    """
    Generate a printable checkerboard calibration pattern.
    
    Args:
        output_path: Where to save the pattern (default: current directory)
        dpi: Print resolution (300 for high quality)
    
    Returns:
        Path to saved checkerboard image
    """
    if output_path is None:
        output_path = Path(__file__).parent / "calibration_checkerboard.png"
    else:
        output_path = Path(output_path)
    
    # Checkerboard dimensions
    rows = 9  # Outer squares
    cols = 7
    square_size_pixels = 100  # Each square is 100 pixels (for 25mm at 300 DPI)
    
    # Calculate image size
    width = cols * square_size_pixels
    height = rows * square_size_pixels
    
    # Create checkerboard pattern
    pattern = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                # White square
                y1 = i * square_size_pixels
                y2 = (i + 1) * square_size_pixels
                x1 = j * square_size_pixels
                x2 = (j + 1) * square_size_pixels
                pattern[y1:y2, x1:x2] = 255
    
    # Add border and instructions
    border = 50
    full_height = height + border * 2 + 150  # Extra space for text
    full_width = width + border * 2
    full_pattern = np.ones((full_height, full_width), dtype=np.uint8) * 255
    
    # Place checkerboard in center
    y_offset = border + 100
    full_pattern[y_offset:y_offset+height, border:border+width] = pattern
    
    # Convert to BGR for colored text
    full_pattern_bgr = cv2.cvtColor(full_pattern, cv2.COLOR_GRAY2BGR)
    
    # Add title and instructions
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(full_pattern_bgr, "CAMERA CALIBRATION CHECKERBOARD", 
                (border, 40), font, 0.8, (0, 0, 0), 2)
    cv2.putText(full_pattern_bgr, f"Pattern: {rows}x{cols} squares | Square size: 25mm", 
                (border, 70), font, 0.5, (100, 100, 100), 1)
    
    # Instructions at bottom
    y_bottom = y_offset + height + 20
    cv2.putText(full_pattern_bgr, "INSTRUCTIONS:", 
                (border, y_bottom), font, 0.6, (0, 0, 255), 2)
    cv2.putText(full_pattern_bgr, "1. Print this page at 100% scale (NO SCALING)", 
                (border, y_bottom + 25), font, 0.4, (0, 0, 0), 1)
    cv2.putText(full_pattern_bgr, "2. Attach to a flat, rigid surface", 
                (border, y_bottom + 45), font, 0.4, (0, 0, 0), 1)
    cv2.putText(full_pattern_bgr, "3. Run the calibration script and capture 20-30 images", 
                (border, y_bottom + 65), font, 0.4, (0, 0, 0), 1)
    
    # Save
    cv2.imwrite(str(output_path), full_pattern_bgr)
    
    print(f"\n✓ Checkerboard pattern saved to: {output_path}")
    return output_path


def check_calibration_exists(calibration_dir=None):
    """
    Check if camera calibration file exists.
    
    Args:
        calibration_dir: Directory to check (default: auto-detect)
    
    Returns:
        Tuple of (exists: bool, calibration_path: Path or None)
    """
    if calibration_dir is None:
        # Auto-detect calibration directory
        calibration_dir = Path(__file__).parent.parent / 'dual_checkerboard_3d' / 'calibration'
    else:
        calibration_dir = Path(calibration_dir)
    
    if not calibration_dir.exists():
        return False, None
    
    # Look for calibration files
    calib_files = list(calibration_dir.glob("camera_calibration*.npz"))
    
    if calib_files:
        return True, calib_files[0]
    else:
        return False, None


def prompt_calibration_setup():
    """
    Interactive prompt to guide user through calibration setup.
    
    Returns:
        Choice: 'calibrate', 'skip', or 'exit'
    """
    print("\n" + "=" * 70)
    print("⚠️  CAMERA CALIBRATION NOT FOUND")
    print("=" * 70)
    
    print("\nCamera calibration improves 3D measurement accuracy.")
    print("\nYou have two options:")
    print("\n  [1] CALIBRATE NOW (Recommended)")
    print("      • Generates a printable checkerboard pattern")
    print("      • Takes 5-10 minutes")
    print("      • Provides accurate measurements")
    
    print("\n  [2] SKIP CALIBRATION (Not Recommended)")
    print("      • Uses generic camera parameters")
    print("      • Lower accuracy")
    print("      • 3D coordinates may be incorrect")
    
    print("\n  [3] EXIT")
    print("      • Exit the program")
    
    while True:
        choice = input("\n👉 Select option (1-3): ").strip()
        
        if choice == '1':
            return 'calibrate'
        elif choice == '2':
            confirm = input("\n⚠️  Are you sure? This will reduce accuracy. (y/N): ").strip().lower()
            if confirm == 'y':
                return 'skip'
        elif choice == '3':
            return 'exit'
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def setup_calibration():
    """
    Complete calibration setup workflow.
    
    Returns:
        True if calibration will be performed, False if skipped
    """
    # Check if calibration exists
    exists, calib_path = check_calibration_exists()
    
    if exists:
        print(f"\n✓ Calibration found: {calib_path.name}")
        return True
    
    # Prompt user
    choice = prompt_calibration_setup()
    
    if choice == 'exit':
        print("\n👋 Exiting...")
        sys.exit(0)
    
    elif choice == 'skip':
        print("\n⚠️  WARNING: Proceeding WITHOUT calibration")
        print("   3D measurements will be LESS ACCURATE")
        return False
    
    elif choice == 'calibrate':
        print("\n" + "=" * 70)
        print("🎯 SETTING UP CALIBRATION")
        print("=" * 70)
        
        # Generate checkerboard pattern
        pattern_path = generate_checkerboard_pattern()
        
        print("\n📋 CALIBRATION STEPS:")
        print("-" * 70)
        print(f"\n1. PRINT the checkerboard pattern:")
        print(f"   File: {pattern_path}")
        print(f"   • Print at 100% scale (no scaling/fit to page)")
        print(f"   • Use white paper, black ink")
        print(f"   • Attach to flat, rigid surface (cardboard/foam board)")
        
        print("\n2. RUN the calibration script:")
        calib_script = Path(__file__).parent.parent / 'dual_checkerboard_3d' / 'checkerboard.py'
        print(f"   python {calib_script.name}")
        
        print("\n3. CAPTURE 20-30 images:")
        print("   • Hold pattern at different angles")
        print("   • Cover all areas of camera view")
        print("   • Keep pattern flat and fully visible")
        print("   • Press SPACE to capture each image")
        print("   • Press Q when done")
        
        print("\n4. RESTART this scanner after calibration")
        
        print("\n" + "=" * 70)
        
        # Ask if user wants to run calibration now
        run_now = input("\n👉 Run calibration script now? (Y/n): ").strip().lower()
        
        if run_now != 'n':
            # Import and run calibration with auto-filled parameters
            print("\n🔄 Starting calibration...")
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent / 'dual_checkerboard_3d'))
                
                # Set default parameters
                import checkerboard
                
                # Auto-fill checkerboard dimensions
                checkerboard.CHECKERBOARD_ROWS = DEFAULT_CHECKERBOARD['rows']
                checkerboard.CHECKERBOARD_COLS = DEFAULT_CHECKERBOARD['cols']
                checkerboard.SQUARE_SIZE = DEFAULT_CHECKERBOARD['square_size_mm']
                
                print(f"\n✓ Using standard checkerboard: {DEFAULT_CHECKERBOARD['description']}")
                print(f"  Rows: {DEFAULT_CHECKERBOARD['rows']} | Cols: {DEFAULT_CHECKERBOARD['cols']}")
                print(f"  Square size: {DEFAULT_CHECKERBOARD['square_size_mm']}mm")
                
                # Run calibration
                checkerboard.main()
                
                print("\n✓ Calibration complete!")
                return True
                
            except Exception as e:
                print(f"\n❌ Calibration failed: {e}")
                print("\nPlease run calibration manually:")
                print(f"   python {calib_script}")
                return False
        else:
            print("\n📌 Remember to run calibration before using the scanner!")
            print(f"   python {calib_script}")
            sys.exit(0)
    
    return False


def get_default_camera_matrix(width=1920, height=1080):
    """
    Generate a default camera matrix when calibration is skipped.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Tuple of (camera_matrix, dist_coeffs)
    """
    print("\n⚠️  Using DEFAULT camera parameters (uncalibrated)")
    
    # Estimate focal length (typical for webcams)
    # Focal length ≈ image_width (in pixels) for standard webcams
    fx = fy = width
    cx = width / 2
    cy = height / 2
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # No distortion correction
    dist_coeffs = np.zeros(5, dtype=np.float32)
    
    print(f"  Focal length: {fx:.1f}px")
    print(f"  Principal point: ({cx:.1f}, {cy:.1f})")
    print("  Distortion: None (assuming perfect lens)")
    
    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    # Test the calibration helper
    print("Camera Calibration Helper Test")
    print("=" * 70)
    
    # Generate pattern
    pattern_path = generate_checkerboard_pattern()
    print(f"\n✓ Pattern generated: {pattern_path}")
    
    # Check calibration
    exists, path = check_calibration_exists()
    if exists:
        print(f"\n✓ Calibration exists: {path}")
    else:
        print("\n⚠️  No calibration found")
        
        # Run setup
        setup_calibration()
