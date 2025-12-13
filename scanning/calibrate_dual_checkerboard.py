"""
Enhanced Camera Calibration with Dual 45° Checkerboard
Uses YOUR EXACT checkerboard: 8x6 pattern, 27.19mm squares
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys  # ← ADD THIS

# ← ADD THIS ENTIRE BLOCK (IMMEDIATE DEBUG OUTPUT)
print("=" * 70, flush=True)
print("🔧 SCRIPT STARTING...", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Script location: {Path(__file__).parent}", flush=True)
print(f"Current directory: {Path.cwd()}", flush=True)
print("=" * 70, flush=True)

# --- YOUR EXACT CHECKERBOARD SETTINGS ---
SQUARE_SIZE = 27.19  # 1 1/16½ inches in mm (from your checkerboard.py)
PATTERN_SIZE = (8, 6)  # Inner corners (width, height) - from your checkerboard.py
THETA_DEG = 90.0  # rotation angle for second board

print(f"✓ Constants loaded: SQUARE_SIZE={SQUARE_SIZE}, PATTERN_SIZE={PATTERN_SIZE}", flush=True)

# Calibration file paths
script_dir = Path(__file__).parent
EXISTING_CALIBRATION_NPZ = script_dir.parent.parent / "camera_calibration.npz"
EXISTING_CALIBRATION_FILE = script_dir / "camera_calibration.npz"

print(f"✓ Paths set:", flush=True)
print(f"  EXISTING_CALIBRATION_NPZ: {EXISTING_CALIBRATION_NPZ}", flush=True)
print(f"  EXISTING_CALIBRATION_FILE: {EXISTING_CALIBRATION_FILE}", flush=True)
print(f"  NPZ exists: {EXISTING_CALIBRATION_NPZ.exists()}", flush=True)
print(f"  FILE exists: {EXISTING_CALIBRATION_FILE.exists()}", flush=True)

print("=" * 70)
print("DUAL 45° CHECKERBOARD CALIBRATION")
print("=" * 70)
print("\n📐 Using YOUR checkerboard:")
print(f"   Physical size: 7⅝\" × 9¾½\" (193.7mm × 247.7mm)")
print(f"   Square size: 1 1/16½\" = {SQUARE_SIZE} mm")
print(f"   Pattern: 9×7 squares = {PATTERN_SIZE[0]}×{PATTERN_SIZE[1]} inner corners")
print(f"   Rotation angle: {THETA_DEG}°")
print("=" * 70)

class DualCheckerboardCalibrator:
    """Calibrate camera using dual 45° checkerboard setup."""
    
    def __init__(self, square_size=SQUARE_SIZE, pattern_size=PATTERN_SIZE, theta_deg=THETA_DEG):
        self.square_size = square_size
        self.pattern_size = pattern_size
        self.theta = np.deg2rad(theta_deg)
        
        # Camera parameters (will be calculated)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.laser_distance_mm = None
        self.calibration_quality = None
        
        # Try to load existing single-board calibration as starting point
        self.load_existing_calibration()
        
    def load_existing_calibration(self):
        """Load existing single checkerboard calibration if available."""
        for calib_path in [EXISTING_CALIBRATION_NPZ, EXISTING_CALIBRATION_FILE]:
            if Path(calib_path).exists():
                try:
                    data = np.load(calib_path)
                    self.camera_matrix = data['camera_matrix']
                    self.dist_coeffs = data['dist_coeffs']
                    print(f"\n✓ Loaded existing calibration from: {calib_path}")
                    print("  This will be used as initial estimate for dual-board calibration")
                    return True
                except Exception as e:
                    print(f"⚠️  Could not load {calib_path}: {e}")
        
        print("\n⚠️  No existing calibration found - will calculate from scratch")
        return False
        
    def rotation_about_axis(self, axis, angle):
        """Generate rotation matrix about arbitrary axis."""
        axis = axis / np.linalg.norm(axis)
        ux, uy, uz = axis
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
            [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
            [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
        ])
        return R
    
    def generate_board1_points(self):
        """Generate 3D object points for flat board 1."""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def generate_board2_points(self, board1_points):
        """Generate 3D object points for rotated board 2 (45° along hinge)."""
        # Hinge line: use last column of corners (right edge)
        hinge_indices = [i * self.pattern_size[1] + (self.pattern_size[1] - 1) 
                        for i in range(self.pattern_size[0])]
        
        hinge_points = board1_points[hinge_indices]
        hinge_origin = hinge_points[0]
        hinge_axis = hinge_points[-1] - hinge_points[0]
        
        # Rotation matrix about hinge axis
        R = self.rotation_about_axis(hinge_axis, self.theta)
        
        # Build second board in local coordinates (same pattern as board 1)
        objp2_local = np.zeros_like(board1_points)
        objp2_local[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp2_local *= self.square_size
        
        # Transform to world coordinates (rotate about hinge)
        objp2 = (R @ objp2_local.T).T + hinge_origin
        return objp2
    
    def calibrate_from_image(self, img_path):
        """Calibrate camera from dual checkerboard image with aggressive detection."""
        
        print("\n" + "=" * 70)
        print("PROCESSING DUAL CHECKERBOARD IMAGE")
        print("=" * 70)
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Could not load image: {img_path}")
            return False
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        print(f"\n✓ Loaded image: {w}x{h} pixels")
        
        # Generate object points
        objp1 = self.generate_board1_points()
        objp2 = self.generate_board2_points(objp1)
        
        # ===== AGGRESSIVE MULTI-STRATEGY DETECTION =====
        
        all_corners = []
        all_ids = []
        
        print("\n[1/6] Strategy 1: Full image detection with multiple configs...")
        
        # Try multiple detection passes
        detection_configs = [
            {'adaptiveThreshold': True, 'normalizeImage': True, 'filterQuads': True},
            {'adaptiveThreshold': True, 'normalizeImage': True, 'filterQuads': False},
            {'adaptiveThreshold': False, 'normalizeImage': True, 'filterQuads': True},
            {'adaptiveThreshold': True, 'normalizeImage': False, 'filterQuads': False},
        ]
        
        for config_idx, config in enumerate(detection_configs):
            flags = 0
            if config['adaptiveThreshold']:
                flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            if config['normalizeImage']:
                flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
            if config['filterQuads']:
                flags |= cv2.CALIB_CB_FILTER_QUADS
            
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Check if this is a new detection (not duplicate)
                is_duplicate = False
                for existing_corners in all_corners:
                    center_dist = np.linalg.norm(np.mean(corners_refined, axis=0) - np.mean(existing_corners, axis=0))
                    if center_dist < 20:  # Less than 20 pixels apart
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_corners.append(corners_refined)
                    all_ids.append(f"full_config_{config_idx}")
                    print(f"   ✓ Found pattern with config {config_idx}")
        
        print(f"\n[2/6] Strategy 2: Left/Right split detection...")
        
        # Left half
        left_gray = gray[:, :w//2]
        ret_left, corners_left = cv2.findChessboardCorners(
            left_gray, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret_left:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_refined = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
            all_corners.append(corners_left_refined)
            all_ids.append("left_half")
            print(f"   ✓ Found pattern in LEFT half")
        
        # Right half
        right_gray = gray[:, w//2:]
        ret_right, corners_right = cv2.findChessboardCorners(
            right_gray, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret_right:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_right_offset = corners_right.copy()
            corners_right_offset[:, :, 0] += w//2
            corners_right_refined = cv2.cornerSubPix(gray, corners_right_offset, (11, 11), (-1, -1), criteria)
            all_corners.append(corners_right_refined)
            all_ids.append("right_half")
            print(f"   ✓ Found pattern in RIGHT half")
        
        print(f"\n[3/6] Strategy 3: Top/Bottom split detection...")
        
        # Top half
        top_gray = gray[:h//2, :]
        ret_top, corners_top = cv2.findChessboardCorners(
            top_gray, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret_top:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_top_refined = cv2.cornerSubPix(top_gray, corners_top, (11, 11), (-1, -1), criteria)
            
            is_duplicate = False
            for existing_corners in all_corners:
                center_dist = np.linalg.norm(np.mean(corners_top_refined, axis=0) - np.mean(existing_corners, axis=0))
                if center_dist < 20:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_corners.append(corners_top_refined)
                all_ids.append("top_half")
                print(f"   ✓ Found pattern in TOP half")
        
        # Bottom half
        bottom_gray = gray[h//2:, :]
        ret_bottom, corners_bottom = cv2.findChessboardCorners(
            bottom_gray, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret_bottom:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_bottom_offset = corners_bottom.copy()
            corners_bottom_offset[:, :, 1] += h//2
            corners_bottom_refined = cv2.cornerSubPix(gray, corners_bottom_offset, (11, 11), (-1, -1), criteria)
            
            is_duplicate = False
            for existing_corners in all_corners:
                center_dist = np.linalg.norm(np.mean(corners_bottom_refined, axis=0) - np.mean(existing_corners, axis=0))
                if center_dist < 20:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_corners.append(corners_bottom_refined)
                all_ids.append("bottom_half")
                print(f"   ✓ Found pattern in BOTTOM half")
        
        print(f"\n[4/6] Strategy 4: Quadrant-based detection...")
        
        # Try each quadrant
        quadrants = [
            ("top_left", gray[:h//2, :w//2], 0, 0),
            ("top_right", gray[:h//2, w//2:], 0, w//2),
            ("bottom_left", gray[h//2:, :w//2], h//2, 0),
            ("bottom_right", gray[h//2:, w//2:], h//2, w//2),
        ]
        
        for quad_name, quad_gray, offset_y, offset_x in quadrants:
            ret_quad, corners_quad = cv2.findChessboardCorners(
                quad_gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_quad:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_quad_offset = corners_quad.copy()
                corners_quad_offset[:, :, 0] += offset_x
                corners_quad_offset[:, :, 1] += offset_y
                corners_quad_refined = cv2.cornerSubPix(gray, corners_quad_offset, (11, 11), (-1, -1), criteria)
                
                is_duplicate = False
                for existing_corners in all_corners:
                    center_dist = np.linalg.norm(np.mean(corners_quad_refined, axis=0) - np.mean(existing_corners, axis=0))
                    if center_dist < 20:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_corners.append(corners_quad_refined)
                    all_ids.append(f"quad_{quad_name}")
                    print(f"   ✓ Found pattern in {quad_name.upper()} quadrant")
        
        print(f"\n[5/6] Strategy 5: Enhanced preprocessing...")
        
        # Try with enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        ret_enh, corners_enh = cv2.findChessboardCorners(
            enhanced, self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret_enh:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_enh_refined = cv2.cornerSubPix(gray, corners_enh, (11, 11), (-1, -1), criteria)
            
            is_duplicate = False
            for existing_corners in all_corners:
                center_dist = np.linalg.norm(np.mean(corners_enh_refined, axis=0) - np.mean(existing_corners, axis=0))
                if center_dist < 20:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_corners.append(corners_enh_refined)
                all_ids.append("enhanced_contrast")
                print(f"   ✓ Found pattern with enhanced contrast")
        
        # Analyze detected patterns
        print(f"\n[6/6] Analyzing {len(all_corners)} unique pattern(s)...")
        
        if len(all_corners) == 0:
            print("❌ No checkerboard patterns detected!")
            print("\n⚠️  TROUBLESHOOTING:")
            print("   1. Check lighting - ensure both boards are well-lit")
            print("   2. Check focus - image should be sharp")
            print("   3. Check visibility - all squares should be clear")
            print("   4. Try recapturing with better positioning")
            
            # Save debug image
            debug_path = Path(img_path).parent / "calibration_debug.jpg"
            cv2.imwrite(str(debug_path), enhanced)
            print(f"\n📸 Debug image saved: {debug_path}")
            print("   (Enhanced contrast version - check if pattern is visible)")
            
            return False
        
        elif len(all_corners) == 1:
            print("⚠️  Only ONE unique checkerboard pattern detected")
            print("   This will use single-board calibration (less accurate depth)")
            print(f"   Detected using: {all_ids[0]}")
            
            corners1_refined = all_corners[0]
            ret1 = True
            ret2 = False
            corners2_refined = None
            
        else:
            # Multiple patterns detected - find the two most distinct
            print(f"   ✓ Multiple patterns detected, selecting best two...")
            
            # Calculate center points
            centers = [np.mean(corners, axis=0)[0] for corners in all_corners]
            
            # Find two patterns with maximum distance
            max_dist = 0
            best_pair = (0, 1)
            
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (i, j)
            
            print(f"   ✓ Selected: '{all_ids[best_pair[0]]}' and '{all_ids[best_pair[1]]}'")
            print(f"   ✓ Distance between patterns: {max_dist:.1f} pixels")
            
            if max_dist < 50:
                print(f"   ⚠️  Patterns are very close ({max_dist:.1f} px)")
                print("      This might be the same board detected twice")
                print("      Proceeding with single-board calibration...")
                corners1_refined = all_corners[best_pair[0]]
                ret1 = True
                ret2 = False
                corners2_refined = None
            else:
                corners1_refined = all_corners[best_pair[0]]
                corners2_refined = all_corners[best_pair[1]]
                ret1 = True
                ret2 = True
        
        # Rest of calibration process (same as before)
        print("\n[Calibration] Processing Board 1...")
        
        if self.camera_matrix is not None:
            print("   Using existing calibration as initial estimate...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                [objp1], [corners1_refined], gray.shape[::-1], 
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
        else:
            print("   Calculating from scratch...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                [objp1], [corners1_refined], gray.shape[::-1], None, None
            )
        
        if not ret:
            print("❌ Camera calibration failed!")
            return False
        
        print("✓ Camera matrix estimated")
        
        # Refine with dual boards if available
        if ret2:
            print("\n[Calibration] Refining with dual boards...")
            obj_points = [objp1, objp2]
            img_points = [corners1_refined, corners2_refined]
            
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, gray.shape[::-1], mtx, dist,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
            
            print("✓ Dual-board calibration complete!")
            calibration_quality = "EXCELLENT (dual board)"
        else:
            print("\n[Calibration] Using single board calibration")
            calibration_quality = "GOOD (single board)"
        
        # Calculate laser distance
        rvec1, tvec1 = rvecs[0], tvecs[0]
        laser_distance = np.linalg.norm(tvec1)
        
        # Store results
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.laser_distance_mm = float(laser_distance)
        self.calibration_quality = calibration_quality
        
        # Display results
        print("\n" + "=" * 70)
        print("CALIBRATION RESULTS")
        print("=" * 70)
        print(f"\n✓ Quality: {calibration_quality}")
        print(f"\n📷 Camera Matrix:")
        print(f"   fx = {mtx[0,0]:.2f} pixels")
        print(f"   fy = {mtx[1,1]:.2f} pixels")
        print(f"   cx = {mtx[0,2]:.2f} pixels")
        print(f"   cy = {mtx[1,2]:.2f} pixels")
        
        print(f"\n📏 Laser Distance: {laser_distance:.1f} mm ({laser_distance/10:.1f} cm)")
        
        print(f"\n🔧 Distortion Coefficients:")
        print(f"   k1 = {dist[0,0]:.6f}")
        print(f"   k2 = {dist[0,1]:.6f}")
        print(f"   p1 = {dist[0,2]:.6f}")
        print(f"   p2 = {dist[0,3]:.6f}")
        print(f"   k3 = {dist[0,4]:.6f}")
        
        # Save visualization
        vis_img = img.copy()
        if ret1:
            cv2.drawChessboardCorners(vis_img, self.pattern_size, corners1_refined, ret1)
        if ret2:
            # Draw second board in different color
            for corner in corners2_refined:
                cv2.circle(vis_img, tuple(corner[0].astype(int)), 5, (0, 255, 255), -1)
        
        output_path = Path(img_path).parent / "calibration_visualization_dual.jpg"
        cv2.imwrite(str(output_path), vis_img)
        print(f"\n✓ Visualization saved: {output_path}")
        
        if not ret2:
            print("\n⚠️  TIPS FOR BETTER DUAL BOARD DETECTION:")
            print("   1. Ensure boards are well-separated (15+ cm apart)")
            print("   2. Both boards should be fully visible in frame")
            print("   3. Try more lighting on Board 2 (the angled one)")
            print("   4. Board 2 should be clearly at 45° angle")
            print("   5. Consider re-capturing with boards further apart")
        
        return True
    
    def save_calibration(self, output_path):
        """Save calibration to both JSON and NPZ formats."""
        if self.camera_matrix is None:
            print("❌ No calibration data to save!")
            return False
        
        # Save as JSON (for easy reading)
        calib_data_json = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "laser_distance_mm": self.laser_distance_mm,
            "quality": self.calibration_quality,
            "square_size_mm": self.square_size,
            "pattern_size": list(self.pattern_size),
            "rotation_angle_deg": np.rad2deg(self.theta),
            "calibration_type": "dual_45deg_checkerboard"
        }
        
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(calib_data_json, f, indent=2)
        
        print(f"\n✓ JSON calibration saved: {json_path}")
        
        # Save as NPZ (compatible with your existing code)
        npz_path = Path(output_path).with_suffix('.npz')
        np.savez(
            npz_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            square_size_mm=self.square_size,
            board_size=self.pattern_size,
            laser_distance_mm=self.laser_distance_mm,
            calibration_quality=self.calibration_quality
        )
        
        print(f"✓ NPZ calibration saved: {npz_path}")
        
        return True
    
    def interactive_calibration(self, num_captures=10):
        """Interactive calibration using webcam with multiple captures."""
        print("\n" + "=" * 70)
        print("INTERACTIVE DUAL CHECKERBOARD CALIBRATION")
        print("=" * 70)
        print("\n📐 Setup Instructions:")
        print(f"   1. Print TWO copies of your {self.pattern_size[0]}x{self.pattern_size[1]} checkerboard")
        print(f"      Physical size: 7⅝\" × 9¾½\" (193.7mm × 247.7mm)")
        print(f"      Square size: 1 1/16½\" ({self.square_size} mm)")
        print("   2. Connect them with tape/hinge along one vertical edge")
        print("   3. Fold second board to exactly 45° angle")
        print("\n📷 Camera Instructions:")
        print(f"   1. Capture {num_captures} images from different angles/distances")
        print("   2. Green corners = board detected")
        print("   3. Yellow/Cyan = second board detected!")
        print("   4. Press SPACE to capture each image")
        print("   5. Press ESC when done capturing (or after 10 images)")
        print("   6. Press Q to quit without calibration")
        print("\n" + "=" * 70)
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("❌ Failed with CAP_DSHOW, trying default...")
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera!")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"\n✅ Camera opened at {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        # Storage for multiple captures
        captured_images = []
        capture_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            
            # Try to detect MULTIPLE boards in different regions
            boards_detected = []
            
            # 1. Try full image
            ret_full, corners_full = cv2.findChessboardCorners(
                gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_full:
                boards_detected.append(("full", corners_full))
                cv2.drawChessboardCorners(frame, self.pattern_size, corners_full, ret_full)
            
            # 2. Try left half
            left_gray = gray[:, :w//2]
            ret_left, corners_left = cv2.findChessboardCorners(
                left_gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_left:
                is_different = True
                if ret_full:
                    center_dist = np.linalg.norm(
                        np.mean(corners_left, axis=0) - np.mean(corners_full, axis=0)[:, :w//2]
                    )
                    if center_dist < 20:
                        is_different = False
                
                if is_different:
                    boards_detected.append(("left", corners_left))
                    for corner in corners_left:
                        cv2.circle(frame, tuple(corner[0].astype(int)), 5, (0, 255, 255), -1)
            
            # 3. Try right half
            right_gray = gray[:, w//2:]
            ret_right, corners_right = cv2.findChessboardCorners(
                right_gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_right:
                corners_right_offset = corners_right.copy()
                corners_right_offset[:, :, 0] += w//2
                
                is_different = True
                for name, existing_corners in boards_detected:
                    center_dist = np.linalg.norm(
                        np.mean(corners_right_offset, axis=0) - np.mean(existing_corners, axis=0)
                    )
                    if center_dist < 20:
                        is_different = False
                        break
                
                if is_different:
                    boards_detected.append(("right", corners_right_offset))
                    for corner in corners_right_offset:
                        cv2.circle(frame, tuple(corner[0].astype(int)), 5, (255, 255, 0), -1)
            
            # Status text
            num_boards = len(boards_detected)
            
            # Progress bar
            progress_width = w - 40
            progress_filled = int((capture_count / num_captures) * progress_width)
            cv2.rectangle(frame, (20, h - 60), (20 + progress_width, h - 40), (100, 100, 100), -1)
            cv2.rectangle(frame, (20, h - 60), (20 + progress_filled, h - 40), (0, 255, 0), -1)
            cv2.putText(frame, f"Captured: {capture_count}/{num_captures}", 
                       (w//2 - 100, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if num_boards >= 2:
                cv2.putText(frame, f"✓ {num_boards} BOARDS! Press SPACE to capture ({capture_count}/{num_captures})", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (10, 10), (w-10, 80), (0, 255, 0), 3)
            elif num_boards == 1:
                cv2.putText(frame, f"⚠ Only 1 board - position both in frame ({capture_count}/{num_captures})", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(frame, f"Position dual checkerboard... ({capture_count}/{num_captures})", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Info overlay
            cv2.putText(frame, f"Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} | Boards: {num_boards} | SPACE=Capture ESC=Done Q=Quit", 
                       (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Dual Checkerboard Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and num_boards >= 1:  # Allow capture even with 1 board
                # Save calibration image
                capture_count += 1
                calib_img_path = Path(f"calibration_capture_{capture_count:03d}.jpg")
                cv2.imwrite(str(calib_img_path), frame)
                captured_images.append(str(calib_img_path))
                
                if num_boards >= 2:
                    print(f"📸 [{capture_count}/{num_captures}] ✓ Captured with {num_boards} boards!")
                else:
                    print(f"📸 [{capture_count}/{num_captures}] ⚠ Captured with {num_boards} board")
                
                # Auto-finish if reached target
                if capture_count >= num_captures:
                    print(f"\n✓ Reached {num_captures} captures! Processing...")
                    break
                    
            elif key == 27:  # ESC key
                if capture_count > 0:
                    print(f"\n✓ Finishing with {capture_count} captures...")
                    break
                else:
                    print("\n⚠️  No images captured yet!")
                    
            elif key == ord('q'):
                print("\n❌ Calibration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        if capture_count == 0:
            print("❌ No images captured!")
            return False
        
        # Process all captured images
        print(f"\n{'=' * 70}")
        print(f"PROCESSING {capture_count} CALIBRATION IMAGES")
        print(f"{'=' * 70}")
        
        return self.calibrate_from_multiple_images(captured_images)
    
    def calibrate_from_multiple_images(self, image_paths):
        """Calibrate camera from multiple dual checkerboard images."""
        
        objp1 = self.generate_board1_points()
        objp2 = self.generate_board2_points(objp1)
        
        # Storage for all detected corners
        all_objpoints = []  # 3D points in real world space
        all_imgpoints = []  # 2D points in image plane
        
        image_size = None
        successful_images = 0
        dual_board_images = 0
        
        for img_idx, img_path in enumerate(image_paths, 1):
            print(f"\n[{img_idx}/{len(image_paths)}] Processing {Path(img_path).name}...")
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ❌ Could not load image")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            # Detect boards using aggressive multi-strategy
            all_corners = []
            all_ids = []
            
            h, w = gray.shape[:2]
            
            # Strategy 1: Full image
            ret_full, corners_full = cv2.findChessboardCorners(
                gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_full:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners_full, (11, 11), (-1, -1), criteria)
                all_corners.append(corners_refined)
                all_ids.append("full")
            
            # Strategy 2: Left/Right split
            left_gray = gray[:, :w//2]
            ret_left, corners_left = cv2.findChessboardCorners(
                left_gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_left:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left_refined = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
                all_corners.append(corners_left_refined)
                all_ids.append("left")
            
            right_gray = gray[:, w//2:]
            ret_right, corners_right = cv2.findChessboardCorners(
                right_gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret_right:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_right_offset = corners_right.copy()
                corners_right_offset[:, :, 0] += w//2
                corners_right_refined = cv2.cornerSubPix(gray, corners_right_offset, (11, 11), (-1, -1), criteria)
                all_corners.append(corners_right_refined)
                all_ids.append("right")
            
            # Analyze detections
            if len(all_corners) == 0:
                print(f"  ❌ No boards detected")
                continue
            
            # Filter duplicates
            unique_corners = []
            unique_ids = []
            
            for i, corners in enumerate(all_corners):
                is_duplicate = False
                for existing in unique_corners:
                    center_dist = np.linalg.norm(np.mean(corners, axis=0) - np.mean(existing, axis=0))
                    if center_dist < 20:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_corners.append(corners)
                    unique_ids.append(all_ids[i])
            
            if len(unique_corners) >= 2:
                # Find two most distant boards
                centers = [np.mean(c, axis=0)[0] for c in unique_corners]
                max_dist = 0
                best_pair = (0, 1)
                
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if dist > max_dist:
                            max_dist = dist
                            best_pair = (i, j)
                
                if max_dist > 50:  # Boards are sufficiently separated
                    # Add both boards
                    all_objpoints.append(objp1)
                    all_imgpoints.append(unique_corners[best_pair[0]])
                    all_objpoints.append(objp2)
                    all_imgpoints.append(unique_corners[best_pair[1]])
                    
                    print(f"  ✓ Dual board detection! ({unique_ids[best_pair[0]]} + {unique_ids[best_pair[1]]}, {max_dist:.0f}px apart)")
                    dual_board_images += 1
                    successful_images += 1
                else:
                    # Too close, use single board
                    all_objpoints.append(objp1)
                    all_imgpoints.append(unique_corners[0])
                    print(f"  ⚠ Single board only ({unique_ids[0]})")
                    successful_images += 1
            
            elif len(unique_corners) == 1:
                # Single board
                all_objpoints.append(objp1)
                all_imgpoints.append(unique_corners[0])
                print(f"  ⚠ Single board only ({unique_ids[0]})")
                successful_images += 1
        
        # Summary
        print(f"\n{'=' * 70}")
        print(f"DETECTION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total images processed: {len(image_paths)}")
        print(f"Successful detections: {successful_images}")
        print(f"Dual-board images: {dual_board_images}")
        print(f"Total calibration views: {len(all_objpoints)}")
        
        if len(all_objpoints) < 3:
            print("\n❌ Not enough successful detections for calibration!")
            print("   Need at least 3 views. Try capturing more images.")
            return False
        
        # Calibrate camera with all detected patterns
        print(f"\n{'=' * 70}")
        print("CAMERA CALIBRATION")
        print(f"{'=' * 70}")
        
        # Convert to proper format - OpenCV needs np.float32
        all_objpoints = [pts.astype(np.float32) for pts in all_objpoints]
        all_imgpoints = [pts.astype(np.float32) for pts in all_imgpoints]
        
        if self.camera_matrix is not None:
            print(f"Using existing calibration as initial estimate...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_objpoints, all_imgpoints, image_size,
                self.camera_matrix.copy(), self.dist_coeffs.copy(),
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
            )
        else:
            print(f"Calculating from scratch...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_objpoints, all_imgpoints, image_size, None, None
            )
        
        if not ret:
            print("❌ Camera calibration failed!")
            return False
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(all_objpoints)):
            imgpoints2, _ = cv2.projectPoints(all_objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(all_imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(all_objpoints)
        
        # Determine quality
        if dual_board_images >= len(image_paths) * 0.5:
            quality = f"EXCELLENT (dual board, {dual_board_images}/{len(image_paths)} images)"
        elif dual_board_images > 0:
            quality = f"GOOD (mixed, {dual_board_images} dual + {successful_images - dual_board_images} single)"
        else:
            quality = f"ACCEPTABLE (single board only)"
        
        # Calculate average laser distance
        laser_distances = [np.linalg.norm(tvec) for tvec in tvecs]
        avg_laser_distance = np.mean(laser_distances)
        
        # Store results
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.laser_distance_mm = float(avg_laser_distance)
        self.calibration_quality = quality
        
        # Display results
        print(f"\n{'=' * 70}")
        print("CALIBRATION RESULTS")
        print(f"{'=' * 70}")
        print(f"\n✓ Quality: {quality}")
        print(f"✓ Reprojection error: {mean_error:.3f} pixels")
        print(f"\n📷 Camera Matrix:")
        print(f"   fx = {mtx[0,0]:.2f} pixels")
        print(f"   fy = {mtx[1,1]:.2f} pixels")
        print(f"   cx = {mtx[0,2]:.2f} pixels")
        print(f"   cy = {mtx[1,2]:.2f} pixels")
        
        print(f"\n📏 Average Laser Distance: {avg_laser_distance:.1f} mm ({avg_laser_distance/10:.1f} cm)")
        print(f"   Distance range: {min(laser_distances):.1f} - {max(laser_distances):.1f} mm")
        
        print(f"\n🔧 Distortion Coefficients:")
        print(f"   k1 = {dist[0,0]:.6f}")
        print(f"   k2 = {dist[0,1]:.6f}")
        print(f"   p1 = {dist[0,2]:.6f}")
        print(f"   p2 = {dist[0,3]:.6f}")
        print(f"   k3 = {dist[0,4]:.6f}")
        
        return True

def recalibrate_from_existing(angle_deg=90.0, output_name='camera_calibration_dual'):
    """Recalibrate using existing captured images with corrected angle."""
    
    print("\n" + "=" * 70)
    print(f"RECALIBRATION WITH {angle_deg}° ANGLE")
    print("=" * 70)
    
    # Create calibrator with corrected angle
    calibrator = DualCheckerboardCalibrator(
        square_size=SQUARE_SIZE,
        pattern_size=PATTERN_SIZE,
        theta_deg=angle_deg  # Use corrected angle
    )
    
    # Find all captured images
    import glob
    captured_images = sorted(glob.glob("calibration_capture_*.jpg"))
    
    if not captured_images:
        print("❌ No captured images found!")
        print("   Expected files like: calibration_capture_001.jpg")
        print("   Run interactive calibration first:")
        print("   python calibrate_dual_checkerboard.py --interactive")
        return False
    
    print(f"\n✓ Found {len(captured_images)} captured images:")
    for img in captured_images[:5]:  # Show first 5
        print(f"   - {img}")
    if len(captured_images) > 5:
        print(f"   ... and {len(captured_images) - 5} more")
    
    # Recalibrate with corrected angle
    success = calibrator.calibrate_from_multiple_images(captured_images)
    
    if success:
        # Save with angle in filename
        output_with_angle = f"{output_name}_{int(angle_deg)}deg"
        calibrator.save_calibration(output_with_angle)
        
        print(f"\n✅ Recalibration complete!")
        print(f"   New files created:")
        print(f"   - {output_with_angle}.npz")
        print(f"   - {output_with_angle}.json")
        
        # Ask if user wants to replace main calibration
        print("\n" + "=" * 70)
        response = input("❓ Replace main camera_calibration.npz with this? (y/n): ")
        if response.lower() == 'y':
            import shutil
            npz_src = Path(output_with_angle + ".npz")
            json_src = Path(output_with_angle + ".json")
            
            # Copy to script directory
            npz_dest = script_dir / "camera_calibration.npz"
            json_dest = script_dir / "camera_calibration.json"
            
            shutil.copy(npz_src, npz_dest)
            shutil.copy(json_src, json_dest)
            print(f"✓ Copied to: {npz_dest}")
            print(f"✓ Copied to: {json_dest}")
            
            # Also copy to parent directory if it exists
            parent_npz = script_dir.parent.parent / "camera_calibration.npz"
            parent_json = script_dir.parent.parent / "camera_calibration.json"
            
            if parent_npz.parent.exists():
                shutil.copy(npz_src, parent_npz)
                shutil.copy(json_src, parent_json)
                print(f"✓ Copied to: {parent_npz}")
                print(f"✓ Copied to: {parent_json}")
            
            print("\n✅ Main calibration files replaced!")
            print("   Your scanner will now use the new calibration.")
        else:
            print(f"\n✓ Calibration saved as: {output_with_angle}.npz")
            print("   To use it, manually copy to camera_calibration.npz")
        
        return True
    
    return False


def main():
    """Main entry point for dual checkerboard calibration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dual Checkerboard Camera Calibration"
    )
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Run interactive calibration with webcam'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to calibration image file'
    )
    parser.add_argument(
        '--recalibrate',  # 🎨 NEW OPTION
        action='store_true',
        help='Recalibrate using existing captured images'
    )
    parser.add_argument(
        '--angle',  # 🎨 NEW OPTION
        type=float,
        default=90.0,
        help='Angle between boards in degrees (default: 90.0)'
    )
    parser.add_argument(
        '--num-captures',
        type=int,
        default=30,
        help='Number of images to capture in interactive mode (default: 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='camera_calibration_dual',
        help='Output filename (without extension, default: camera_calibration_dual)'
    )
    
    args = parser.parse_args()
    
    # 🎨 NEW: Handle recalibration mode FIRST
    if args.recalibrate:
        success = recalibrate_from_existing(
            angle_deg=args.angle,
            output_name=args.output
        )
        return 0 if success else 1
    
    # Create calibrator with specified angle
    calibrator = DualCheckerboardCalibrator(
        square_size=SQUARE_SIZE,
        pattern_size=PATTERN_SIZE,
        theta_deg=args.angle  # Use angle from command line
    )
    
    # Run calibration
    success = False
    
    if args.interactive:
        # Interactive mode with webcam
        success = calibrator.interactive_calibration(num_captures=args.num_captures)
        
    elif args.image:
        # Single image mode
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"❌ Image file not found: {args.image}")
            return 1
        
        success = calibrator.calibrate_from_image(img_path)
    
    else:
        print("❌ Please specify --interactive, --image <path>, or --recalibrate")
        parser.print_help()
        return 1
    
    # Save results if successful
    if success:
        # Add angle to output filename if not default
        if args.angle != 90.0:
            output_name = f"{args.output}_{int(args.angle)}deg"
        else:
            output_name = args.output
            
        calibrator.save_calibration(output_name)
        
        print("\n" + "=" * 70)
        print("✅ CALIBRATION COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Calibration files saved:")
        print(f"   JSON: {output_name}.json")
        print(f"   NPZ:  {output_name}.npz")
        print("\n🔧 Integration with scanner:")
        print("   Your scanner will automatically use this calibration!")
        print("   Just run: python laser_3d_scanner_advanced.py")
        print("\n📊 Expected improvements:")
        print("   ✓ Accurate depth measurements (correct Z-axis)")
        print("   ✓ Proper object scale (no more oversized objects!)")
        print("   ✓ Better point cloud density calculations")
        print("=" * 70)
        
        return 0
    else:
        print("\n❌ Calibration failed!")
        return 1


# 🎨 ENTRY POINT - ADD THIS:
if __name__ == "__main__":
    import sys
    sys.exit(main())