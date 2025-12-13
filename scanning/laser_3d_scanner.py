import cv2
import numpy as np
import os
from camera_distance_detector_calibrated import load_camera_calibration

WEBCAM_INDEX = 0
CALIBRATION_FILE = "D:/Users/Planet UI/camera_calibration.npz"

# OPTIMIZED LASER DETECTION SETTINGS (adjustable during runtime)
LASER_BRIGHTNESS_THRESHOLD = 230
LASER_MIN_AREA = 20
LASER_MAX_AREA = 500

# MEASURED: Distance between camera lens center and laser lens center
LASER_CAMERA_OFFSET_MM = 24.6  # 31/32 inch

# Simple linear distance estimation (adjust based on your setup)
# These are rough estimates - you can calibrate by measuring a few points
DISTANCE_MIN_CM = 20  # Minimum detectable distance
DISTANCE_MAX_CM = 150  # Maximum detectable distance
Y_AT_MIN_DISTANCE = 500  # Y pixel position when object is at min distance (lower in frame)
Y_AT_MAX_DISTANCE = 100  # Y pixel position when object is at max distance (higher in frame)

def detect_laser_dot(frame, brightness_threshold, min_area, max_area):
    """Detect the laser dot in the frame using adjustable settings."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find brightest regions with adjustable threshold
    _, thresh = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find ALL valid dots and return the brightest one
    valid_dots = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                brightness = gray[cy, cx]
                
                valid_dots.append({
                    'x': cx,
                    'y': cy,
                    'area': area,
                    'brightness': brightness
                })
    
    if len(valid_dots) == 0:
        return None, None, None, []
    
    # Return the brightest dot (most likely to be laser)
    brightest = max(valid_dots, key=lambda d: d['brightness'])
    return brightest['x'], brightest['y'], brightest['area'], valid_dots

def estimate_distance_linear(dot_y):
    """
    Simple linear interpolation for distance based on Y position.
    Lower in frame (higher Y) = closer
    Higher in frame (lower Y) = farther
    """
    # Clamp to valid range
    y_clamped = max(Y_AT_MAX_DISTANCE, min(Y_AT_MIN_DISTANCE, dot_y))
    
    # Linear interpolation
    t = (y_clamped - Y_AT_MAX_DISTANCE) / (Y_AT_MIN_DISTANCE - Y_AT_MAX_DISTANCE)
    distance_cm = DISTANCE_MAX_CM - t * (DISTANCE_MAX_CM - DISTANCE_MIN_CM)
    
    return distance_cm

def scan_3d_points():
    """
    Scan 3D points using laser and calibrated camera.
    """
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    if camera_matrix is None:
        print("\n❌ Cannot run without camera calibration!")
        print("   Please run checkerboard.py first")
        return
    
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Trying default backend...")
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Adjustable parameters
    brightness_threshold = LASER_BRIGHTNESS_THRESHOLD
    min_area = LASER_MIN_AREA
    max_area = LASER_MAX_AREA
    
    print("\n" + "=" * 80)
    print("3D LASER SCANNER - SIMPLE LINEAR DISTANCE")
    print("=" * 80)
    print("\nOptimized Laser Detection:")
    print(f"   Brightness Threshold: {brightness_threshold} (adjust with i/k)")
    print(f"   Area Range: {min_area}-{max_area} pixels (adjust with j/l)")
    print(f"   Distance Range: {DISTANCE_MIN_CM}-{DISTANCE_MAX_CM}cm")
    print("\nControls:")
    print("   SPACE - Capture 3D point")
    print("   i/k   - Adjust brightness threshold")
    print("   j/l   - Adjust min area")
    print("   +/-   - Adjust max area")
    print("   'c'   - Clear points")
    print("   's'   - Save point cloud")
    print("   'd'   - Toggle debug view")
    print("   'q'   - Quit")
    print("=" * 80 + "\n")
    
    points_3d = []
    show_debug = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Undistort frame
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Detect laser dot with current settings
        dot_x, dot_y, dot_area, all_dots = detect_laser_dot(
            undistorted, brightness_threshold, min_area, max_area
        )
        
        display_frame = undistorted.copy()
        
        # Draw ALL detected dots if debug mode
        if show_debug:
            for i, dot in enumerate(all_dots):
                color = (0, 255, 0) if (dot['x'] == dot_x and dot['y'] == dot_y) else (100, 100, 255)
                cv2.circle(display_frame, (dot['x'], dot['y']), 10, color, 2)
                cv2.putText(display_frame, f"{i+1}: B={dot['brightness']}", 
                           (dot['x'] + 15, dot['y']), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw selected laser dot and estimate distance
        if dot_x and dot_y:
            cv2.circle(display_frame, (dot_x, dot_y), 15, (0, 255, 0), 3)
            cv2.circle(display_frame, (dot_x, dot_y), 3, (0, 0, 255), -1)
            
            # Simple linear distance estimation
            distance_cm = estimate_distance_linear(dot_y)
            
            cv2.putText(display_frame, f"Distance: {distance_cm:.1f}cm", 
                       (dot_x + 20, dot_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Y={dot_y}", 
                       (dot_x + 20, dot_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Info overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, overlay)
        display_frame = overlay
        
        cv2.putText(display_frame, f"Points: {len(points_3d)} | Threshold: {brightness_threshold}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Area: {min_area}-{max_area} | Dots: {len(all_dots)}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if dot_x and dot_y:
            cv2.putText(display_frame, f"✅ Laser detected at ({dot_x},{dot_y})", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(display_frame, f"❌ No laser (found {len(all_dots)} dots)", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(display_frame, "Adjust: i/k=brightness j/l=area d=debug", 
                       (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        
        cv2.putText(display_frame, "SPACE=capture | s=save | d=debug | q=quit", 
                   (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("3D Laser Scanner", display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 32:  # SPACE key code
            if dot_x and dot_y:
                # Capture 3D point
                fx = new_camera_matrix[0, 0]
                fy = new_camera_matrix[1, 1]
                cx = new_camera_matrix[0, 2]
                cy = new_camera_matrix[1, 2]
                
                # Estimate distance
                distance_cm = estimate_distance_linear(dot_y)
                z = distance_cm * 10  # cm to mm
                
                # Calculate 3D coordinates
                x = (dot_x - cx) * z / fx
                y = (dot_y - cy) * z / fy
                
                points_3d.append([x, y, z])
                print(f"✅ Point {len(points_3d)}: ({x:.1f}, {y:.1f}, {z:.1f})mm @ {distance_cm:.1f}cm")
            else:
                print("   ❌ No laser detected")
        
        elif key == ord('i'):
            brightness_threshold = min(brightness_threshold + 5, 255)
            print(f"Brightness threshold: {brightness_threshold}")
        elif key == ord('k'):
            brightness_threshold = max(brightness_threshold - 5, 0)
            print(f"Brightness threshold: {brightness_threshold}")
        elif key == ord('j'):
            min_area = max(min_area - 5, 1)
            print(f"Min area: {min_area}")
        elif key == ord('l'):
            min_area += 5
            print(f"Min area: {min_area}")
        elif key == ord('+') or key == ord('='):
            max_area += 50
            print(f"Max area: {max_area}")
        elif key == ord('-'):
            max_area = max(max_area - 50, min_area + 1)
            print(f"Max area: {max_area}")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug view: {'ON' if show_debug else 'OFF'}")
        elif key == ord('c'):
            points_3d = []
            print("🗑️  Cleared all points")
        elif key == ord('s') and len(points_3d) > 0:
            filename = "D:/Users/Planet UI/scan_3d.npz"
            np.savez(filename, points=np.array(points_3d))
            print(f"💾 Saved {len(points_3d)} points to: {filename}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Session complete. Captured {len(points_3d)} 3D points")
    print(f"Final settings: Threshold={brightness_threshold}, Area={min_area}-{max_area}")

if __name__ == "__main__":
    scan_3d_points()