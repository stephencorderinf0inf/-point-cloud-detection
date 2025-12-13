import cv2
import numpy as np
import os

# YOUR CHECKERBOARD MEASUREMENTS
CHECKERBOARD_SIZE = (8, 6)  # Inner corners (width, height)
SQUARE_SIZE_MM = 27.19  # 1 1/16 1/2 inches in mm

CALIBRATION_FILE = "D:/Users/Planet UI/camera_calibration.npz"

def calibrate_camera():
    """
    Calibrate camera using your physical checkerboard.
    Corrects lens distortion for accurate 3D measurements.
    """
    
    print("=" * 80)
    print("CAMERA CALIBRATION - CHECKERBOARD METHOD")
    print("=" * 80)
    print("\nYour Checkerboard:")
    print(f"   Size: 7⅝\" × 9¾½\" (193.7mm × 247.7mm)")
    print(f"   Squares: 1 1/16½\" (27.19mm each)")
    print(f"   Pattern: 9×7 squares = {CHECKERBOARD_SIZE[0]}×{CHECKERBOARD_SIZE[1]} inner corners")
    print("\nInstructions:")
    print("1. Print this checkerboard pattern (ensure accurate scale!)")
    print("2. Mount on flat, rigid surface")
    print("3. Show checkerboard to camera from different angles")
    print("4. Press SPACE when checkerboard is detected (green corners)")
    print("5. Capture 15-20 images from various angles/distances")
    print("6. Press 'q' when done to calculate calibration")
    print("=" * 80 + "\n")
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 
                            0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_MM
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane
    
    print("🎥 Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Failed with CAP_DSHOW, trying default backend...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera!")
        print("   Is your webcam connected?")
        print("   Is another program using it?")
        return
    
    print("✅ Camera opened successfully!")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📷 Camera resolution: {int(actual_width)}×{int(actual_height)}")
    
    captured_count = 0
    frame_count = 0
    
    print("\n🔍 Starting detection loop...")
    print("   Waiting for frames...\n")
    
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print(f"⚠️  Failed to read frame {frame_count}")
            if frame_count > 100:  # Give up after 100 failed attempts
                print("❌ Too many failed frame reads. Exiting.")
                break
            continue
        
        if frame_count == 1:
            print(f"✅ First frame received! Shape: {frame.shape}")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(
            gray, 
            CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        display_frame = frame.copy()
        
        if ret_corners:
            if frame_count % 30 == 0:  # Print every second
                print(f"✅ Checkerboard detected at frame {frame_count}")
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Draw corners
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD_SIZE, corners_refined, ret_corners)
            
            # Status
            cv2.putText(display_frame, "Checkerboard detected! Press SPACE to capture", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Capture on spacebar
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                captured_count += 1
                print(f"📸 Captured image {captured_count}/20")
        else:
            cv2.putText(display_frame, "Checkerboard not detected", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, "Move checkerboard closer or adjust angle", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            key = cv2.waitKey(30) & 0xFF
        
        # Info overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, display_frame.shape[0] - 100), 
                     (500, display_frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        cv2.putText(display_frame, f"Images captured: {captured_count}/20", 
                   (20, display_frame.shape[0] - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to finish calibration", 
                   (20, display_frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Camera Calibration", display_frame)
        
        if key == ord('q'):
            print("\n🛑 User pressed 'q' - stopping capture")
            if captured_count < 10:
                print(f"\n⚠️  Only {captured_count} images captured.")
                print("   Recommend at least 10-15 images for good calibration.")
                print("   Continue anyway? (y/n): ", end='')
                response = input().lower()
                if response != 'y':
                    print("   Continuing capture...")
                    continue
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released")
    
    if captured_count == 0:
        print("\n❌ No images captured. Calibration aborted.")
        return
    
    print(f"\n🔄 Calculating calibration from {captured_count} images...")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    if ret:
        print("\n" + "=" * 80)
        print("✅ CALIBRATION SUCCESSFUL!")
        print("=" * 80)
        print("\nCamera Matrix (Intrinsic Parameters):")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        print(f"\nFocal Length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f}) pixels")
        print(f"Principal Point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f}) pixels")
        print(f"Radial Distortion (k1, k2, k3): {dist_coeffs[0][:3]}")
        print(f"Tangential Distortion (p1, p2): {dist_coeffs[0][3:5]}")
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                              camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        print(f"\nReprojection Error: {mean_error:.3f} pixels")
        
        if mean_error < 0.5:
            print("   ✅ Excellent calibration!")
        elif mean_error < 1.0:
            print("   ✅ Good calibration")
        else:
            print("   ⚠️  Calibration acceptable but could be improved")
            print("   Consider recalibrating with more varied angles/positions")
        
        # Save calibration
        np.savez(CALIBRATION_FILE, 
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs,
                 rvecs=rvecs,
                 tvecs=tvecs,
                 square_size_mm=SQUARE_SIZE_MM,
                 board_size=CHECKERBOARD_SIZE)
        
        print(f"\n💾 Calibration saved to: {CALIBRATION_FILE}")
        print("=" * 80)
        print("\n✅ You can now use calibrated distance detection!")
        print("   Run: camera_distance_detector_calibrated.py")
        print("=" * 80)
    else:
        print("\n❌ Calibration failed. Try again with better images.")

if __name__ == "__main__":
    try:
        calibrate_camera()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()