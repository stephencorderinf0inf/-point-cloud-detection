"""
Enhanced Ellipse Detection Using Sphere Matrix
Maps detected ellipse points to 3D sphere coordinates for more accurate fitting
"""

import numpy as np
import cv2
import sys
from pathlib import Path

def load_sphere_matrix(filepath="sphere_matrix.npy"):
    """Load the pre-computed sphere matrix."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Sphere matrix not found: {filepath}")
    return np.load(filepath)

def detect_ellipses_2d(image, min_area=500, max_area=50000):
    """
    Detect ellipses in 2D image with STRICT filtering.
    Only returns high-quality circular objects.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply stronger blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use Canny edge detection for better quality
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges slightly to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    ellipses = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # STRICT area filtering
        if area < min_area or area > max_area:
            continue
        
        # Need at least 10 points (not just 5!)
        if len(contour) < 10:
            continue
        
        # Compute perimeter
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        # Circularity test: 4π·area / perimeter²
        # Perfect circle = 1.0, lower = more elongated
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # REJECT if not circular enough
        if circularity < 0.4:  # Adjust this threshold (0.4 = fairly circular)
            continue
        
        try:
            # Fit ellipse in 2D
            ellipse_2d = cv2.fitEllipse(contour)
            (x, y), (w, h), angle = ellipse_2d
            
            # STRICT aspect ratio check (2D)
            aspect_ratio_2d = max(w, h) / (min(w, h) + 1e-6)
            
            # REJECT if too elongated (> 3:1 ratio)
            if aspect_ratio_2d > 3.0:
                continue
            
            # REJECT if too small
            if max(w, h) < 20:  # Minimum 20 pixels diameter
                continue
            
            ellipses.append({
                'ellipse_2d': ellipse_2d,
                'contour': contour,
                'area': area,
                'circularity': circularity,
                'aspect_ratio_2d': aspect_ratio_2d
            })
        except:
            continue
    
    print(f"   Filtered to {len(ellipses)} high-quality candidates")
    return ellipses

def map_contour_to_3d(contour, sphere_matrix):
    """
    Map 2D contour points to 3D sphere coordinates.
    """
    points_3d = []
    valid_indices = []
    
    for i, point in enumerate(contour):
        u, v = int(point[0][0]), int(point[0][1])
        
        # Check bounds
        if v < 0 or v >= sphere_matrix.shape[0] or u < 0 or u >= sphere_matrix.shape[1]:
            continue
        
        point_3d = sphere_matrix[v, u]
        
        # Skip NaN points
        if not np.isnan(point_3d).any():
            points_3d.append(point_3d)
            valid_indices.append(i)
    
    return np.array(points_3d), valid_indices

def fit_ellipse_3d(points_3d):
    """
    Fit an ellipse to 3D points on the sphere.
    Returns center, axes, and rotation in 3D.
    """
    if len(points_3d) < 5:
        return None
    
    # Compute centroid
    center = np.mean(points_3d, axis=0)
    
    # Center the points
    centered = points_3d - center
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalue (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Semi-axes lengths (2 sigma)
    axes = 2 * np.sqrt(eigenvalues)
    
    return {
        'center_3d': center,
        'axes_3d': axes,
        'rotation_3d': eigenvectors,
        'eigenvalues': eigenvalues
    }

def compute_ellipse_quality(ellipse_2d, ellipse_3d, contour_3d):
    """
    Compute quality metrics comparing 2D and 3D ellipse fits.
    """
    if ellipse_3d is None:
        return {
            'is_valid': False,
            'quality_score': 0.0,
            'metrics': {}
        }
    
    # 2D ellipse parameters
    (x, y), (w, h), angle = ellipse_2d
    aspect_ratio_2d = max(w, h) / (min(w, h) + 1e-6)
    
    # 3D ellipse parameters
    axes_3d = ellipse_3d['axes_3d']
    aspect_ratio_3d = max(axes_3d[:2]) / (min(axes_3d[:2]) + 1e-6)
    
    # Check if third axis is too large (indicates non-planar ellipse)
    axis_ratio = axes_3d[2] / axes_3d[0]  # z-axis / x-axis
    
    # REJECT if z-axis is too large (not flat enough)
    if axis_ratio > 0.1:  # More than 10% of major axis
        return {
            'is_valid': False,
            'quality_score': 0.0,
            'metrics': {
                'axis_ratio': float(axis_ratio),
                'reason': 'Non-planar (z-axis too large)'
            }
        }
    
    # Compute residuals (distance from points to ellipse plane)
    center = ellipse_3d['center_3d']
    normal = ellipse_3d['rotation_3d'][:, 2]  # Third eigenvector
    
    distances = []
    for point in contour_3d:
        dist = abs(np.dot(point - center, normal))
        distances.append(dist)
    
    avg_residual = np.mean(distances)
    max_residual = np.max(distances)
    std_residual = np.std(distances)
    
    # Quality score (lower residuals = better)
    quality_score = 1.0 / (1.0 + avg_residual * 100)
    
    # Circularity (closer to 1.0 = more circular)
    circularity = 1.0 / aspect_ratio_3d
    
    metrics = {
        'aspect_ratio_2d': float(aspect_ratio_2d),
        'aspect_ratio_3d': float(aspect_ratio_3d),
        'axis_ratio': float(axis_ratio),
        'avg_residual': float(avg_residual),
        'max_residual': float(max_residual),
        'std_residual': float(std_residual),
        'circularity': float(circularity),
        'num_points': int(len(contour_3d))
    }
    
    return {
        'is_valid': True,
        'quality_score': float(quality_score),
        'metrics': metrics
    }

def visualize_ellipses(image, ellipses_data, show_2d=True, show_3d=True):
    """
    Visualize detected ellipses with quality scores.
    """
    vis = image.copy()
    
    for i, data in enumerate(ellipses_data):
        ellipse_2d = data['ellipse_2d']
        quality = data['quality']
        
        # Color based on quality
        if quality['quality_score'] > 0.8:
            color = (0, 255, 0)  # Green = excellent
        elif quality['quality_score'] > 0.6:
            color = (0, 255, 255)  # Yellow = good
        elif quality['quality_score'] > 0.4:
            color = (0, 165, 255)  # Orange = fair
        else:
            color = (0, 0, 255)  # Red = poor
        
        # Draw 2D ellipse
        if show_2d:
            cv2.ellipse(vis, ellipse_2d, color, 2)
        
        # Draw contour points
        cv2.drawContours(vis, [data['contour']], -1, color, 1)
        
        # Add quality text
        center = (int(ellipse_2d[0][0]), int(ellipse_2d[0][1]))
        text = f"#{i+1}: {quality['quality_score']:.2f}"
        cv2.putText(vis, text, (center[0]-30, center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add detailed metrics
        y_offset = center[1] + 20
        metrics = quality['metrics']
        cv2.putText(vis, f"AR: {metrics['aspect_ratio_3d']:.2f}", 
                    (center[0]-30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        y_offset += 15
        cv2.putText(vis, f"Res: {metrics['avg_residual']:.3f}", 
                    (center[0]-30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis

def process_image(image, sphere_matrix, min_area=100, max_area=50000):
    """
    Complete pipeline: detect ellipses, map to 3D, fit, and compute quality.
    """
    print("\n" + "=" * 70)
    print("PROCESSING IMAGE FOR ELLIPSE DETECTION")
    print("=" * 70)
    
    # Detect ellipses in 2D
    print("\n1. Detecting ellipses in 2D...")
    ellipses_2d = detect_ellipses_2d(image, min_area, max_area)
    print(f"   Found {len(ellipses_2d)} candidate ellipses")
    
    if len(ellipses_2d) == 0:
        print("   ⚠️  No ellipses detected!")
        return []
    
    # Process each ellipse
    ellipses_data = []
    for i, ellipse_data in enumerate(ellipses_2d):
        print(f"\n2. Processing ellipse #{i+1}...")
        
        # Map contour to 3D
        print(f"   Mapping contour to 3D sphere...")
        contour_3d, valid_indices = map_contour_to_3d(
            ellipse_data['contour'], sphere_matrix
        )
        
        if len(contour_3d) < 5:
            print(f"   ⚠️  Not enough valid 3D points ({len(contour_3d)})")
            continue
        
        print(f"   ✓ Mapped {len(contour_3d)} points to 3D")
        
        # Fit ellipse in 3D
        print(f"   Fitting ellipse in 3D...")
        ellipse_3d = fit_ellipse_3d(contour_3d)
        
        if ellipse_3d is None:
            print(f"   ⚠️  3D fit failed")
            continue
        
        print(f"   ✓ 3D ellipse fitted")
        print(f"     Center: {ellipse_3d['center_3d']}")
        print(f"     Axes: {ellipse_3d['axes_3d']}")
        
        # Compute quality
        quality = compute_ellipse_quality(
            ellipse_data['ellipse_2d'], ellipse_3d, contour_3d
        )
        
        # SKIP if invalid
        if not quality.get('is_valid', False):
            print(f"   ❌ Rejected: Invalid 3D geometry")
            continue
        
        # SKIP if quality too low
        if quality['quality_score'] < 0.5:
            print(f"   ❌ Rejected: Quality too low ({quality['quality_score']:.3f})")
            continue
        
        print(f"   Quality score: {quality['quality_score']:.3f}")
        print(f"     Aspect ratio (3D): {quality['metrics']['aspect_ratio_3d']:.2f}")
        print(f"     Axis ratio (z/x): {quality['metrics']['axis_ratio']:.4f}")
        print(f"     Avg residual: {quality['metrics']['avg_residual']:.4f}")
        
        ellipses_data.append({
            'ellipse_2d': ellipse_data['ellipse_2d'],
            'ellipse_3d': ellipse_3d,
            'contour': ellipse_data['contour'],
            'contour_3d': contour_3d,
            'quality': quality
        })
    
    # Sort by quality
    ellipses_data.sort(key=lambda x: x['quality']['quality_score'], reverse=True)
    
    print("\n" + "=" * 70)
    print(f"✅ Processed {len(ellipses_data)} ellipses successfully")
    print("=" * 70)
    
    return ellipses_data

def main():
    print("=" * 70)
    print("ENHANCED ELLIPSE DETECTION WITH SPHERE MATRIX")
    print("=" * 70)
    
    # Load sphere matrix
    try:
        print("\nLoading sphere matrix...")
        sphere_matrix = load_sphere_matrix()
        print(f"✓ Sphere matrix loaded: {sphere_matrix.shape}")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\nPlease run 'pixel_to_sphere.py' first to generate the sphere matrix.")
        return 1
    
    # Load calibration for camera access
    calib_file = Path("camera_calibration_dual.npz")
    if not calib_file.exists():
        print(f"❌ ERROR: Calibration file not found: {calib_file}")
        return 1
    
    print(f"✓ Calibration file found: {calib_file}")
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera")
        return 1
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera opened at 1280×720")
    print("\n" + "=" * 70)
    print("CONTROLS:")
    print("  SPACE - Capture and analyze frame")
    print("  Q - Quit")
    print("=" * 70)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read frame")
            break
        
        # Display live feed
        cv2.imshow("Ellipse Detection (Press SPACE to analyze)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            # Process frame
            ellipses_data = process_image(frame, sphere_matrix)
            
            if len(ellipses_data) > 0:
                # Visualize results
                vis = visualize_ellipses(frame, ellipses_data)
                cv2.imshow("Detected Ellipses (Press any key)", vis)
                cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program ended")
    return 0

if __name__ == "__main__":
    sys.exit(main())