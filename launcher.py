"""
Camera Tools Launcher
Main menu to run any camera tool
"""

import os
import sys

def print_menu():
    print("\n" + "=" * 80)
    print("CAMERA TOOLS LAUNCHER")
    print("=" * 80)
    print("\n[CAMERA] CALIBRATION:")
    print("  1. Checkerboard Calibration (calibration/checkerboard.py)")
    print("  2. Distance Calibration (calibration/camera_distance_detector_calibrated.py)")
    
    print("\n[SEARCH] DETECTION:")
    print("  3. Laser Detection Test (detection/laser_detection_test.py)")
    print("  4. Laser + Distance Detection (detection/detect_laser_and_distance.py)")
    
    print("\n[RULER] 3D SCANNING:")
    print("  5. Basic 3D Scanner (scanning/laser_3d_scanner.py)")
    print("  6. Advanced Red Laser Scanner (scanning/laser_3d_scanner_advanced.py)")
    print("  7. Infrared Laser Scanner (scanning/laser_3d_scanner_infrared.py)")
    
    print("\n[FOLDER] DATA:")
    print("  8. View saved scans (data/)")
    
    print("\n  q. Quit")
    print("=" * 80)

def run_script(script_path):
    """Run a Python script"""
    if os.path.exists(script_path):
        print(f"\n[ROCKET] Running: {script_path}\n")
        os.system(f'py -3.12 "{script_path}"')
    else:
        print(f"\n[X] Script not found: {script_path}")
        input("\nPress Enter to continue...")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        print_menu()
        choice = input("\nSelect option: ").strip().lower()
        
        if choice == '1':
            run_script(os.path.join(base_dir, "calibration", "checkerboard.py"))
        elif choice == '2':
            run_script(os.path.join(base_dir, "calibration", "camera_distance_detector_calibrated.py"))
        elif choice == '3':
            run_script(os.path.join(base_dir, "detection", "laser_detection_test.py"))
        elif choice == '4':
            run_script(os.path.join(base_dir, "detection", "detect_laser_and_distance.py"))
        elif choice == '5':
            run_script(os.path.join(base_dir, "scanning", "laser_3d_scanner.py"))
        elif choice == '6':
            run_script(os.path.join(base_dir, "scanning", "laser_3d_scanner_advanced.py"))
        elif choice == '7':
            run_script(os.path.join(base_dir, "scanning", "laser_3d_scanner_infrared.py"))
        elif choice == '8':
            data_dir = os.path.join(base_dir, "data")
            print(f"\n[FOLDER] Data folder: {data_dir}")
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                if files:
                    print("\nSaved files:")
                    for f in files:
                        print(f"  * {f}")
                else:
                    print("  (No files yet)")
            else:
                print("  (Folder not found)")
            input("\nPress Enter to continue...")
        elif choice == 'q':
            print("\n[WAVE] Goodbye!")
            break
        else:
            print("\n[X] Invalid choice")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
