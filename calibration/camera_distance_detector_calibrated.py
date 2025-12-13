import cv2
import numpy as np
import os

WEBCAM_INDEX = 0
CALIBRATION_FILE = "D:/Users/Planet UI/camera_calibration.npz"

# MEASURED: Distance between camera lens center and laser lens center
LASER_CAMERA_OFFSET_MM = 24.6  # 15/16 inch + 1/32 inch ≈ 24.6mm

def load_camera_calibration():
    """Load camera calibration from checkerboard calibration."""
    if not os.path.exists(CALIBRATION_FILE):
        print(f"⚠️  Camera calibration file not found: {CALIBRATION_FILE}")
        print(f"   Run checkerboard.py first to calibrate camera")
        return None, None
    
    data = np.load(CALIBRATION_FILE)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    
    print("✅ Loaded camera calibration:")
    print(f"   Focal length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
    print(f"   Principal point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
    print(f"   Laser-Camera offset: {LASER_CAMERA_OFFSET_MM:.1f} mm")
    
    return camera_matrix, dist_coeffs

def get_distance_with_calibration(frame, camera_matrix, dist_coeffs):
    """
    Measure distance using calibrated camera (corrects lens distortion).
    """
    # 1. Undistort the frame
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # 2. Detect laser dot in UNDISTORTED image
    dot_x_pixel, dot_y_pixel = detect_laser_dot(undistorted)
    
    if dot_x_pixel is None or dot_y_pixel is None:
        return None, None, None, undistorted
    
    # 3. Convert to mm using CALIBRATED focal length
    fx = new_camera_matrix[0, 0]
    fy = new_camera_matrix[1, 1]
    cx = new_camera_matrix[0, 2]
    cy = new_camera_matrix[1, 2]
    
    # Use your existing calibration curve (now with undistorted image)
    a = 0.006140
    b = -1.353207
    c = 419.711156 - dot_y_pixel
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, dot_y_pixel, "Discriminant < 0", undistorted
    
    distance1 = (-b + discriminant**0.5) / (2*a)
    distance2 = (-b - discriminant**0.5) / (2*a)
    
    # Return valid distance
    for d in [distance1, distance2]:
        if 19 <= d <= 112:
            return d, dot_y_pixel, None, undistorted
    
    return None, dot_y_pixel, f"Out of range: {distance1:.1f}, {distance2:.1f}", undistorted

def detect_laser_dot(frame):
    """Detect the laser dot in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find brightest regions
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 500:  # Dot size range
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy
    
    return None, None

def main():
    """Run the calibrated camera distance detector."""
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_camera_calibration()
    
    if camera_matrix is None:
        print("\n❌ Cannot run without camera calibration!")
        print("   Please run checkerboard.py first to calibrate your camera")
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
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    print("\n" + "=" * 80)
    print("CALIBRATED CAMERA DISTANCE DETECTOR")
    print("Using lens distortion correction from checkerboard calibration")
    print(f"Laser-Camera offset: {LASER_CAMERA_OFFSET_MM:.1f} mm (31/32 inch)")
    print("Range: 19-112 cm")
    print("Press 'q' to quit")
    print("=" * 80 + "\n")
    
    last_distance = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Get distance with calibration
        distance, dot_y_mm, error, undistorted = get_distance_with_calibration(
            frame, camera_matrix, dist_coeffs
        )
        
        if distance is not None:
            last_distance = distance
        
        # Draw on undistorted frame
        display_frame = undistorted.copy()
        h, w = display_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Crosshairs
        cv2.line(display_frame, (center_x, 0), (center_x, h), (0, 255, 0), 1)
        cv2.line(display_frame, (0, center_y), (w, center_y), (0, 255, 0), 1)
        
        # Info box
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        if last_distance is not None:
            cv2.putText(display_frame, f"Distance (Calibrated):", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"{last_distance:.1f} cm", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            cv2.putText(display_frame, f"Lens distortion corrected ✅", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display_frame, f"Waiting for laser dot...", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
        # Draw dot
        dot_x, dot_y = detect_laser_dot(undistorted)
        if dot_x and dot_y:
            cv2.circle(display_frame, (dot_x, dot_y), 15, (0, 0, 255), 3)
            cv2.line(display_frame, (dot_x - 20, dot_y), (dot_x + 20, dot_y), (0, 255, 255), 2)
            cv2.line(display_frame, (dot_x, dot_y - 20), (dot_x, dot_y + 20), (0, 255, 255), 2)
        
        cv2.imshow("Calibrated Distance Detector", display_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()