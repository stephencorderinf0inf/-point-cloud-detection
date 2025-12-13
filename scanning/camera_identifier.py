"""
Camera Identifier
=================
Detects and identifies connected cameras to ensure calibration matches.
"""

import cv2
import hashlib
import json
from pathlib import Path


def get_camera_fingerprint(cap):
    """
    Generate a unique fingerprint for the connected camera.
    
    Args:
        cap: OpenCV VideoCapture object
    
    Returns:
        Dictionary with camera properties
    """
    if not cap.isOpened():
        return None
    
    fingerprint = {
        # Resolution capabilities
        'max_width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'max_height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        
        # Camera backend
        'backend': cap.getBackendName() if hasattr(cap, 'getBackendName') else 'unknown',
        
        # FPS capability
        'fps': int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30,
        
        # Format info
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)) if cap.get(cv2.CAP_PROP_FOURCC) else 0,
        
        # Additional properties (some may not be supported)
        'brightness': int(cap.get(cv2.CAP_PROP_BRIGHTNESS)) if cap.get(cv2.CAP_PROP_BRIGHTNESS) != -1 else None,
        'contrast': int(cap.get(cv2.CAP_PROP_CONTRAST)) if cap.get(cv2.CAP_PROP_CONTRAST) != -1 else None,
    }
    
    # Generate hash from key properties
    key_string = f"{fingerprint['max_width']}x{fingerprint['max_height']}_{fingerprint['backend']}_{fingerprint['fourcc']}"
    fingerprint['hash'] = hashlib.md5(key_string.encode()).hexdigest()[:8]
    
    return fingerprint


def get_camera_description(fingerprint):
    """Get human-readable camera description."""
    if fingerprint is None:
        return "Unknown Camera"
    
    desc = f"{fingerprint['max_width']}x{fingerprint['max_height']}"
    if fingerprint['fps']:
        desc += f" @ {fingerprint['fps']}fps"
    desc += f" ({fingerprint['backend']})"
    
    return desc


def save_camera_fingerprint(fingerprint, calibration_path):
    """
    Save camera fingerprint alongside calibration file.
    
    Args:
        fingerprint: Camera fingerprint dict
        calibration_path: Path to calibration .npz file
    """
    if fingerprint is None:
        return
    
    # Save as JSON next to calibration file
    fingerprint_path = Path(str(calibration_path).replace('.npz', '_camera.json'))
    
    with open(fingerprint_path, 'w') as f:
        json.dump(fingerprint, f, indent=2)
    
    print(f"✓ Camera fingerprint saved: {fingerprint_path.name}")


def load_camera_fingerprint(calibration_path):
    """
    Load camera fingerprint from calibration directory.
    
    Args:
        calibration_path: Path to calibration .npz file
    
    Returns:
        Camera fingerprint dict or None
    """
    fingerprint_path = Path(str(calibration_path).replace('.npz', '_camera.json'))
    
    if not fingerprint_path.exists():
        return None
    
    try:
        with open(fingerprint_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load camera fingerprint: {e}")
        return None


def compare_cameras(current_fp, calibration_fp):
    """
    Compare current camera with calibration camera.
    
    Args:
        current_fp: Current camera fingerprint
        calibration_fp: Calibration camera fingerprint
    
    Returns:
        Tuple of (match: bool, confidence: float, differences: list)
    """
    if current_fp is None or calibration_fp is None:
        return False, 0.0, ["Missing fingerprint data"]
    
    differences = []
    score = 0
    max_score = 0
    
    # Resolution check (critical)
    max_score += 10
    if (current_fp['max_width'] == calibration_fp['max_width'] and 
        current_fp['max_height'] == calibration_fp['max_height']):
        score += 10
    else:
        differences.append(
            f"Resolution: {current_fp['max_width']}x{current_fp['max_height']} vs "
            f"{calibration_fp['max_width']}x{calibration_fp['max_height']}"
        )
    
    # Backend check (important)
    max_score += 5
    if current_fp['backend'] == calibration_fp['backend']:
        score += 5
    else:
        differences.append(
            f"Backend: {current_fp['backend']} vs {calibration_fp['backend']}"
        )
    
    # FPS check (minor)
    max_score += 3
    if abs(current_fp['fps'] - calibration_fp['fps']) <= 5:
        score += 3
    else:
        differences.append(
            f"FPS: {current_fp['fps']} vs {calibration_fp['fps']}"
        )
    
    # FOURCC check (minor)
    max_score += 2
    if current_fp['fourcc'] == calibration_fp['fourcc']:
        score += 2
    
    # Calculate confidence
    confidence = score / max_score if max_score > 0 else 0
    match = confidence >= 0.8  # 80% threshold
    
    return match, confidence, differences


def check_camera_match(cap, calibration_path):
    """
    Check if current camera matches the calibration file.
    
    Args:
        cap: Current VideoCapture object
        calibration_path: Path to calibration file
    
    Returns:
        Tuple of (match: bool, confidence: float, message: str)
    """
    # Get current camera fingerprint
    current_fp = get_camera_fingerprint(cap)
    
    if current_fp is None:
        return False, 0.0, "Cannot detect current camera"
    
    # Load calibration camera fingerprint
    calibration_fp = load_camera_fingerprint(calibration_path)
    
    if calibration_fp is None:
        # No fingerprint saved - legacy calibration
        return None, 0.5, "⚠️  Legacy calibration (no camera fingerprint)"
    
    # Compare cameras
    match, confidence, differences = compare_cameras(current_fp, calibration_fp)
    
    # Build message
    current_desc = get_camera_description(current_fp)
    calib_desc = get_camera_description(calibration_fp)
    
    if match:
        message = f"✓ Camera matches calibration ({confidence*100:.0f}% confidence)\n"
        message += f"  Camera: {current_desc}"
    else:
        message = f"⚠️  CAMERA MISMATCH ({confidence*100:.0f}% confidence)\n"
        message += f"  Current camera: {current_desc}\n"
        message += f"  Calibrated for: {calib_desc}\n"
        message += f"  Differences:\n"
        for diff in differences:
            message += f"    • {diff}\n"
    
    return match, confidence, message


def prompt_recalibration():
    """
    Prompt user to recalibrate for different camera.
    
    Returns:
        True if user wants to recalibrate, False otherwise
    """
    print("\n" + "=" * 70)
    print("⚠️  DIFFERENT CAMERA DETECTED")
    print("=" * 70)
    print("\nThe current camera does not match the calibration file.")
    print("Using incorrect calibration will result in:")
    print("  • Incorrect 3D measurements")
    print("  • Distorted point clouds")
    print("  • Poor mesh quality")
    
    print("\nYou have three options:")
    print("\n  [1] RECALIBRATE for this camera (Recommended)")
    print("      • Takes 5-10 minutes")
    print("      • Ensures accurate measurements")
    
    print("\n  [2] Continue with EXISTING calibration (Not Recommended)")
    print("      • May produce incorrect results")
    print("      • Use only for testing/preview")
    
    print("\n  [3] EXIT")
    
    while True:
        choice = input("\n👉 Select option (1-3): ").strip()
        
        if choice == '1':
            return True
        elif choice == '2':
            confirm = input("\n⚠️  Are you sure? Results will be inaccurate. (y/N): ").strip().lower()
            if confirm == 'y':
                return False
        elif choice == '3':
            print("\n👋 Exiting...")
            import sys
            sys.exit(0)
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    # Test camera detection
    print("Camera Identifier Test")
    print("=" * 70)
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        # Get fingerprint
        fp = get_camera_fingerprint(cap)
        
        print("\nCamera Fingerprint:")
        print(json.dumps(fp, indent=2))
        
        print(f"\nDescription: {get_camera_description(fp)}")
        print(f"Hash: {fp['hash']}")
        
        cap.release()
    else:
        print("❌ Cannot open camera")
