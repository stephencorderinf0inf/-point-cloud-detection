"""
3D Scanner optimized for INFRARED laser dots
- Detects bright spots with low color saturation (IR appears gray/white)
- Uses brightness + size filtering
- Curve/edge tracing
- Corner detection
"""

import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration.camera_distance_detector_calibrated import load_camera_calibration

WEBCAM_INDEX = 0

# Detection modes
MODE_LASER = 0
MODE_CURVE = 1
MODE_CORNERS = 2

# IR laser detection settings
LASER_BRIGHTNESS_THRESHOLD = 230
LASER_MIN_AREA = 10
LASER_MAX_AREA = 500
MAX_SATURATION = 80  # IR appears with LOW saturation (gray/white)

# Distance estimation
DISTANCE_MIN_CM = 20
DISTANCE_MAX_CM = 150
Y_AT_MIN_DISTANCE = 500
Y_AT_MAX_DISTANCE = 100

def detect_ir_laser_dot(frame, brightness_threshold, min_area, max_area, max_saturation):
    """Detect INFRARED laser dot - bright + low saturation (gray/white)."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # IR appears BRIGHT
    _, bright_mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    # IR has LOW saturation (appears gray/white, not colored)
    h, s, v = cv2.split(hsv)
    low_sat_mask = cv2.threshold(s, max_saturation, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Combine: BRIGHT + LOW SATURATION = IR laser
    ir_mask = cv2.bitwise_and(bright_mask, low_sat_mask)
    
    # Find contours
    contours, _ = cv2.findContours(ir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_dots = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                brightness = gray[cy, cx]
                saturation = s[cy, cx]
                
                valid_dots.append({
                    'x': cx,
                    'y': cy,
                    'area': area,
                    'brightness': brightness,
                    'saturation': saturation
                })
    
    if len(valid_dots) == 0:
        return None, None, None, [], ir_mask, bright_mask, low_sat_mask
    
    # Return the brightest dot with lowest saturation
    best = max(valid_dots, key=lambda d: d['brightness'] * (255 - d['saturation']))
    return best['x'], best['y'], best['area'], valid_dots, ir_mask, bright_mask, low_sat_mask

def detect_curves(frame):
    """Detect curves/edges in the scene."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_curves = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    return valid_curves, edges

def detect_corners(frame):
    """Detect corners using Harris corner detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    corner_threshold = 0.01 * corners.max()
    corner_locations = np.where(corners > corner_threshold)
    corner_points = list(zip(corner_locations[1], corner_locations[0]))
    return corner_points, corners

def estimate_distance_linear(dot_y):
    """Linear interpolation for distance based on Y position."""
    y_clamped = max(Y_AT_MAX_DISTANCE, min(Y_AT_MIN_DISTANCE, dot_y))
    t = (y_clamped - Y_AT_MAX_DISTANCE) / (Y_AT_MIN_DISTANCE - Y_AT_MAX_DISTANCE)
    distance_cm = DISTANCE_MAX_CM - t * (DISTANCE_MAX_CM - DISTANCE_MIN_CM)
    return distance_cm

def scan_3d_points():
    """3D scanner optimized for INFRARED laser."""
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    if camera_matrix is None:
        print("\n[X] Cannot run without camera calibration!")
        print("   Please run checkerboard.py first")
        return
    
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("[X] Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Parameters
    brightness_threshold = LASER_BRIGHTNESS_THRESHOLD
    min_area = LASER_MIN_AREA
    max_area = LASER_MAX_AREA
    max_saturation = MAX_SATURATION
    current_mode = MODE_LASER
    show_debug = False
    
    mode_names = {
        MODE_LASER: "IR LASER DOT",
        MODE_CURVE: "CURVE TRACING",
        MODE_CORNERS: "CORNER DETECTION"
    }
    
    print("\n" + "=" * 80)
    print("3D SCANNER - INFRARED LASER OPTIMIZED")
    print("=" * 80)
    print("\nDetection Modes:")
    print("   1 - IR Laser Dot (bright + low saturation)")
    print("   2 - Curve/Edge Tracing")
    print("   3 - Corner Detection")
    print("\nLaser Controls (Mode 1):")
    print("   i/k   - Brightness threshold")
    print("   u/o   - Max saturation (lower = more IR-like)")
    print("   j/l   - Area filtering")
    print("\nGeneral Controls:")
    print("   SPACE - Capture point(s)")
    print("   'd'   - Toggle debug view")
    print("   'c'   - Clear points")
    print("   's'   - Save point cloud")
    print("   'q'   - Quit")
    print("=" * 80)
    print("\nInfo: IR lasers appear as bright WHITE/GRAY spots (low color saturation)")
    print("      If not detecting, lower brightness (k) or increase max saturation (o)\n")
    
    points_3d = []
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
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
        
        display_frame = undistorted.copy()
        
        # MODE: IR LASER DOT
        if current_mode == MODE_LASER:
            dot_x, dot_y, dot_area, all_dots, ir_mask, bright_mask, low_sat_mask = detect_ir_laser_dot(
                undistorted, brightness_threshold, min_area, max_area, max_saturation
            )
            
            # Show masks in debug mode
            if show_debug:
                bright_colored = cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2BGR)
                low_sat_colored = cv2.cvtColor(low_sat_mask, cv2.COLOR_GRAY2BGR)
                ir_colored = cv2.cvtColor(ir_mask, cv2.COLOR_GRAY2BGR)
                
                masks = np.hstack([bright_colored, low_sat_colored, ir_colored])
                
                cv2.putText(masks, "BRIGHT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(masks, "LOW SAT (GRAY)", (640 + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(masks, "IR COMBINED", (1280 + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                masks_resized = cv2.resize(masks, (1920, 360))
                cv2.imshow("Debug: Bright | Low-Saturation | IR-Combined", masks_resized)
            
            # Draw all detected dots
            for dot in all_dots:
                color = (0, 255, 0) if (dot['x'] == dot_x and dot['y'] == dot_y) else (100, 100, 255)
                cv2.circle(display_frame, (dot['x'], dot['y']), 10, color, 2)
                cv2.putText(display_frame, f"B:{dot['brightness']} S:{dot['saturation']}", 
                           (dot['x'] + 15, dot['y']), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw selected laser
            if dot_x and dot_y:
                cv2.circle(display_frame, (dot_x, dot_y), 15, (0, 255, 0), 3)
                cv2.circle(display_frame, (dot_x, dot_y), 3, (0, 0, 255), -1)
                distance_cm = estimate_distance_linear(dot_y)
                cv2.putText(display_frame, f"{distance_cm:.1f}cm", 
                           (dot_x + 20, dot_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # MODE: CURVE TRACING
        elif current_mode == MODE_CURVE:
            curves, edges = detect_curves(undistorted)
            
            if show_debug:
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                display_frame = np.hstack([display_frame, edges_colored])
            
            cv2.drawContours(display_frame, curves, -1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Curves: {len(curves)}", 
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # MODE: CORNER DETECTION
        elif current_mode == MODE_CORNERS:
            corner_points, corner_map = detect_corners(undistorted)
            
            for (x, y) in corner_points[:100]:
                cv2.circle(display_frame, (x, y), 5, (255, 0, 255), 2)
            
            cv2.putText(display_frame, f"Corners: {len(corner_points)}", 
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Info overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        cv2.putText(display_frame, f"Mode: {mode_names[current_mode]} | Points: {len(points_3d)}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_mode == MODE_LASER:
            cv2.putText(display_frame, f"Bright>={brightness_threshold} Sat<={max_saturation} Area:{min_area}-{max_area}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Dots detected: {len(all_dots) if 'all_dots' in locals() else 0}", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "i/k=bright u/o=saturation j/l=area", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(display_frame, "1=Laser 2=Curves 3=Corners | SPACE=capture d=debug", 
                   (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("IR 3D Scanner", display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 32:  # SPACE
            if current_mode == MODE_LASER and dot_x and dot_y:
                fx = new_camera_matrix[0, 0]
                fy = new_camera_matrix[1, 1]
                cx = new_camera_matrix[0, 2]
                cy = new_camera_matrix[1, 2]
                
                distance_cm = estimate_distance_linear(dot_y)
                z = distance_cm * 10
                x = (dot_x - cx) * z / fx
                y = (dot_y - cy) * z / fy
                
                points_3d.append([x, y, z])
                print(f"[CHECK] IR point {len(points_3d)}: ({x:.1f}, {y:.1f}, {z:.1f})mm @ {distance_cm:.1f}cm")
            
            elif current_mode == MODE_CURVE:
                for curve in curves:
                    for point in curve[::5]:
                        px, py = point[0]
                        distance_cm = estimate_distance_linear(py)
                        z = distance_cm * 10
                        x = (px - new_camera_matrix[0, 2]) * z / new_camera_matrix[0, 0]
                        y = (py - new_camera_matrix[1, 2]) * z / new_camera_matrix[1, 1]
                        points_3d.append([x, y, z])
                print(f"[CHECK] Curves: {len(curves)} - total points: {len(points_3d)}")
            
            elif current_mode == MODE_CORNERS:
                for (px, py) in corner_points[:50]:
                    distance_cm = estimate_distance_linear(py)
                    z = distance_cm * 10
                    x = (px - new_camera_matrix[0, 2]) * z / new_camera_matrix[0, 0]
                    y = (py - new_camera_matrix[1, 2]) * z / new_camera_matrix[1, 1]
                    points_3d.append([x, y, z])
                print(f"[CHECK] Corners: {min(50, len(corner_points))} - total: {len(points_3d)}")
        
        elif key == ord('1'):
            current_mode = MODE_LASER
            print(f"Mode: {mode_names[current_mode]}")
        elif key == ord('2'):
            current_mode = MODE_CURVE
            print(f"Mode: {mode_names[current_mode]}")
        elif key == ord('3'):
            current_mode = MODE_CORNERS
            print(f"Mode: {mode_names[current_mode]}")
        
        elif key == ord('i'):
            brightness_threshold = min(brightness_threshold + 5, 255)
            print(f"Brightness >= {brightness_threshold}")
        elif key == ord('k'):
            brightness_threshold = max(brightness_threshold - 5, 0)
            print(f"Brightness >= {brightness_threshold}")
        elif key == ord('u'):
            max_saturation = max(max_saturation - 10, 0)
            print(f"Max saturation <= {max_saturation}")
        elif key == ord('o'):
            max_saturation = min(max_saturation + 10, 255)
            print(f"Max saturation <= {max_saturation}")
        elif key == ord('j'):
            min_area = max(min_area - 5, 1)
            print(f"Min area: {min_area}")
        elif key == ord('l'):
            min_area += 5
            print(f"Min area: {min_area}")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug: {'ON' if show_debug else 'OFF'}")
            if not show_debug:
                cv2.destroyWindow("Debug: Bright | Low-Saturation | IR-Combined")
        elif key == ord('c'):
            points_3d = []
            print("[TRASH] Cleared")
        elif key == ord('s') and len(points_3d) > 0:
            filename = os.path.join(data_dir, "scan_3d_ir.npz")
            np.savez(filename, points=np.array(points_3d))
            print(f"[SAVE] Saved {len(points_3d)} points to: {filename}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[CHECK] Captured {len(points_3d)} 3D points")
    print(f"Final settings: Bright>={brightness_threshold}, Sat<={max_saturation}, Area:{min_area}-{max_area}")

if __name__ == "__main__":
    scan_3d_points()