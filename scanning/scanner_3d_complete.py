"""
COMPLETE 3D SCANNER - All-In-One Solution
No complicated setup needed - just run and scan!
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class Complete3DScanner:
    """All-in-one 3D scanner with automatic setup."""
    
    def __init__(self):
        self.base_dir = Path("D:/Users/Planet UI/3d_scan_objects")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-load or create calibration
        self.calibration = self._load_or_create_calibration()
        self.sphere_matrix = self._load_or_create_sphere_matrix()
        
        self.current_object = None
        self.current_session = None
        self.captured_points = []
    
    def _load_or_create_calibration(self):
        """Load existing calibration or guide user to create one."""
        calib_file = Path("camera_calibration_dual.npz")
        
        if calib_file.exists():
            print("✓ Loaded camera calibration")
            calib = np.load(str(calib_file))
            return {
                'camera_matrix': calib['camera_matrix'],
                'dist_coeffs': calib['dist_coeffs']
            }
        else:
            print("⚠️  No calibration found - using default")
            # Default calibration (user should calibrate once for accuracy)
            return {
                'camera_matrix': np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=float),
                'dist_coeffs': np.zeros((5, 1))
            }
    
    def _load_or_create_sphere_matrix(self):
        """Load sphere matrix or compute it."""
        sphere_file = Path("sphere_matrix.npy")
        
        if sphere_file.exists():
            print("✓ Loaded sphere mapping matrix")
            return np.load(str(sphere_file))
        else:
            print("⚠️  No sphere matrix - computing (one-time setup)...")
            return self._compute_sphere_matrix()
    
    def _compute_sphere_matrix(self):
        """Compute pixel-to-3D sphere mapping."""
        height, width = 720, 1280
        sphere_matrix = np.zeros((height, width, 3), dtype=float)
        
        cx, cy = width / 2, height / 2
        
        for y in range(height):
            for x in range(width):
                # Normalized coordinates
                nx = (x - cx) / cx
                ny = (y - cy) / cy
                
                r_sq = nx**2 + ny**2
                
                if r_sq <= 1.0:
                    # Map to sphere
                    z = np.sqrt(1.0 - r_sq)
                    sphere_matrix[y, x] = [nx, ny, z]
                else:
                    sphere_matrix[y, x] = [np.nan, np.nan, np.nan]
        
        # Save for future use
        np.save("sphere_matrix.npy", sphere_matrix)
        print("✓ Sphere matrix computed and saved")
        return sphere_matrix
    
    def select_or_create_object(self):
        """Interactive object selection."""
        print("\n" + "="*70)
        print("SELECT OBJECT TO SCAN")
        print("="*70)
        
        # List existing objects
        catalog_file = self.base_dir / "catalog.json"
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                catalog = json.load(f)
            
            if catalog.get('objects'):
                print("\nExisting objects:")
                for i, obj in enumerate(catalog['objects'], 1):
                    print(f"  {i}. {obj['name']} ({obj['category']})")
                
                choice = input("\nEnter number to select, or 'N' for new: ").strip().upper()
                
                if choice != 'N' and choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(catalog['objects']):
                        self.current_object = catalog['objects'][idx]
                        print(f"✓ Selected: {self.current_object['name']}")
                        return
        else:
            catalog = {'objects': []}
        
        # Create new object
        name = input("\nNew object name: ").strip()
        category = input("Category (vehicle/furniture/other): ").strip() or "other"
        
        safe_name = name.lower().replace(' ', '_')
        obj_path = self.base_dir / category / safe_name
        
        # Create folders
        (obj_path / 'raw').mkdir(parents=True, exist_ok=True)
        (obj_path / 'point_clouds').mkdir(parents=True, exist_ok=True)
        (obj_path / 'meshes').mkdir(parents=True, exist_ok=True)
        
        self.current_object = {
            'name': name,
            'safe_name': safe_name,
            'category': category,
            'path': str(obj_path),
            'created': datetime.now().isoformat()
        }
        
        catalog['objects'].append(self.current_object)
        
        with open(catalog_file, 'w') as f:
            json.dump(catalog, f, indent=2)
        
        print(f"✓ Created: {name}")
    
    def start_scanning(self):
        """Start the scanning session."""
        if not self.current_object:
            print("❌ No object selected!")
            return
        
        # Create session folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_path = Path(self.current_object['path']) / 'raw' / f"scan_{timestamp}"
        session_path.mkdir(parents=True, exist_ok=True)
        
        self.current_session = {
            'path': session_path,
            'image_count': 0,
            'point_count': 0
        }
        
        print("\n" + "="*70)
        print(f"SCANNING: {self.current_object['name']}")
        print("="*70)
        print("\nControls:")
        print("  SPACE - Capture frame")
        print("  P     - Process captured frames → 3D")
        print("  S     - Save point cloud")
        print("  Q     - Quit")
        print("="*70)
        
        # Open camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Detect laser
                laser_mask = self._detect_laser(frame)
                
                # Overlay visualization
                display = frame.copy()
                display[laser_mask > 0] = [0, 255, 0]  # Green overlay
                
                # Info overlay
                cv2.putText(display, f"Object: {self.current_object['name']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Frames: {self.current_session['image_count']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Points: {self.current_session['point_count']}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("3D Scanner", display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self._capture_frame(frame, laser_mask)
                elif key == ord('p'):
                    self._process_all_frames()
                elif key == ord('s'):
                    self._save_point_cloud()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _detect_laser(self, frame):
        """Detect red laser line in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red detection (adjust for your laser)
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        return cv2.bitwise_or(mask1, mask2)
    
    def _capture_frame(self, frame, laser_mask):
        """Capture current frame."""
        img_file = self.current_session['path'] / f"frame_{self.current_session['image_count']:04d}.png"
        cv2.imwrite(str(img_file), frame)
        
        self.current_session['image_count'] += 1
        print(f"✓ Captured frame {self.current_session['image_count']}")
    
    def _process_all_frames(self):
        """Process all captured frames into 3D points."""
        print("\n🔄 Processing frames...")
        
        image_files = sorted(self.current_session['path'].glob("*.png"))
        all_points = []
        
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            laser_mask = self._detect_laser(img)
            
            # Get laser pixels
            laser_pixels = np.column_stack(np.where(laser_mask > 0))
            
            # Map to 3D
            for y, x in laser_pixels:
                if 0 <= y < self.sphere_matrix.shape[0] and 0 <= x < self.sphere_matrix.shape[1]:
                    pt_3d = self.sphere_matrix[y, x]
                    if not np.any(np.isnan(pt_3d)):
                        all_points.append(pt_3d)
        
        self.captured_points = np.array(all_points)
        self.current_session['point_count'] = len(self.captured_points)
        
        print(f"✓ Processed {len(image_files)} frames → {len(self.captured_points)} 3D points")
    
    def _save_point_cloud(self):
        """Save the point cloud."""
        if len(self.captured_points) == 0:
            print("❌ No points to save! Press 'P' to process first.")
            return
        
        output_dir = Path(self.current_object['path']) / 'point_clouds'
        output_file = output_dir / f"{self.current_session['path'].name}.npy"
        
        np.save(str(output_file), self.captured_points)
        print(f"✓ Saved point cloud: {output_file}")
        print(f"  Total points: {len(self.captured_points)}")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("🎯 COMPLETE 3D SCANNER - All-In-One")
    print("="*70)
    
    scanner = Complete3DScanner()
    scanner.select_or_create_object()
    scanner.start_scanning()
    
    print("\n✅ Scan complete!")


if __name__ == "__main__":
    main()