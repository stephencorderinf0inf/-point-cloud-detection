"""
3D Scanner with Object Management Integration
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from object_manager import ObjectManager

def scanner_with_object_management():
    """Run 3D scanner with object management."""
    manager = ObjectManager()
    
    print("\n" + "="*70)
    print("3D SCANNER WITH OBJECT MANAGEMENT")
    print("="*70)
    
    # Step 1: Select or create object
    print("\n1. Select object:")
    print("   A) Use existing object")
    print("   B) Create new object")
    
    choice = input("\nChoice (A/B): ").strip().upper()
    
    if choice == 'B':
        name = input("Object name: ").strip()
        category = input("Category: ").strip() or "other"
        manager.create_object_folder(name, category)
        object_name = name
    else:
        objects = manager.list_objects()
        if not objects:
            print("❌ No objects found! Create one first.")
            return
        
        idx = int(input("\nSelect object number: ")) - 1
        object_name = objects[idx]['name']
    
    # Step 2: Start capture session
    session_info = manager.start_capture_session(object_name)
    
    if not session_info:
        return
    
    # Step 3: Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n" + "="*70)
    print("CONTROLS:")
    print("  SPACE - Capture image")
    print("  Q - End session and quit")
    print("="*70)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display
            display = frame.copy()
            cv2.putText(display, f"Object: {object_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Images: {session_info['image_count']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press SPACE to capture", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("3D Scanner", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                manager.save_image(frame, session_info)
    
    finally:
        manager.end_capture_session(session_info)
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Scanner closed")


if __name__ == "__main__":
    scanner_with_object_management()