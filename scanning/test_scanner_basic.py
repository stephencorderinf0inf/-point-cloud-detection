"""Minimal scanner test"""
import cv2
import numpy as np
import sys

print("=" * 60)
print("BASIC SCANNER TEST")
print("=" * 60)

# Test 1: OpenCV
print("\n[1/5] Testing OpenCV import...")
print(f"✓ OpenCV version: {cv2.__version__}")

# Test 2: Camera
print("\n[2/5] Testing camera access...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera index 0")
    print("   Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Cannot open camera index 1")
        sys.exit(1)

print("✓ Camera opened successfully")

# Test 3: Get camera properties
print("\n[3/5] Getting camera properties...")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"✓ Resolution: {width}x{height}")
print(f"✓ FPS: {fps}")

# Test 4: Frame capture
print("\n[4/5] Testing frame capture...")
ret, frame = cap.read()
if not ret or frame is None:
    print("❌ Cannot read frame from camera")
    cap.release()
    sys.exit(1)

print(f"✓ Frame captured: {frame.shape}")
print(f"✓ Frame type: {frame.dtype}")

# Test 5: Display
print("\n[5/5] Testing window display...")
cv2.imshow("Basic Test - Press ANY KEY to close", frame)
print("✓ Window created")
print("\n>>> Press ANY KEY in the video window to continue <<<")
cv2.waitKey(0)

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour camera and OpenCV are working correctly.")
print("Now let's test the advanced scanner imports...\n")