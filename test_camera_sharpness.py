"""
Test camera sharpness without undistortion.
"""
import cv2
import numpy as np

def calculate_sharpness(image):
    """Calculate sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    # ADJUSTED: Normalize for low-quality cameras (50-500 instead of 0-2000)
    normalized = min(variance / 500.0, 1.0)  # Changed from 2000 to 500
    return variance, normalized

print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera!")
    print("Make sure no other application is using the camera.")
    input("Press Enter to exit...")
    exit()

print("Camera opened successfully!")
print("Testing camera sharpness...")
print("Press 'q' to quit")
print("\nInstructions:")
print("1. Try moving closer/farther from camera")
print("2. Try tapping the camera lens to trigger autofocus")
print("3. Clean the lens if values are consistently low")
print("\n[ADJUSTED] Normalization changed from /2000 to /500 for low-quality cameras")
print("Starting in 2 seconds...")

import time
time.sleep(2)

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_count}")
            continue
        
        frame_count += 1
        
        # Calculate sharpness on raw frame
        raw_variance, raw_normalized = calculate_sharpness(frame)
        
        # Display
        cv2.putText(frame, f"Sharpness: {raw_normalized:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Raw Variance: {raw_variance:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Color code the quality (adjusted thresholds)
        if raw_normalized > 0.5:
            color = (0, 255, 0)  # Green - Good
            status = "GOOD"
        elif raw_normalized > 0.3:
            color = (0, 255, 255)  # Yellow - Fair
            status = "FAIR"
        else:
            color = (0, 0, 255)  # Red - Poor
            status = "POOR"
        
        cv2.putText(frame, f"Status: {status}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.putText(frame, "[ADJUSTED for low-quality camera]", (10, 680),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Camera Sharpness Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\nQuitting after {frame_count} frames...")
            break

except Exception as e:
    print(f"\nERROR: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"Test complete! Total frames: {frame_count}")