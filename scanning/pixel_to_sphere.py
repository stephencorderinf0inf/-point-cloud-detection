import numpy as np
import cv2
import sys
from pathlib import Path

def pixel_to_sphere_matrix(K, d, R, t, img_shape, sphere_radius=1.0, O=np.zeros(3)):
    """
    For each pixel, compute the intersection with a sphere in the world frame.
    Returns a (H, W, 3) array of 3D points on the sphere.
    """
    print(f"Building sphere matrix for image shape {img_shape}...")
    h, w = img_shape
    
    # Build pixel grid
    print("  Creating pixel grid...")
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack([u, v], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    
    # Undistort pixels
    print("  Undistorting pixels...")
    undist = cv2.undistortPoints(pixels, K, d, P=K).reshape(-1, 2)
    
    # Create rays in camera frame
    print("  Creating camera rays...")
    rays_c = np.concatenate([undist, np.ones((undist.shape[0], 1))], axis=1)
    rays_c /= np.linalg.norm(rays_c, axis=1, keepdims=True)
    
    # Transform rays to world frame
    print("  Transforming to world frame...")
    rays_w = (R.T @ rays_c.T).T
    C_w = -R.T @ t.reshape(3)
    
    # Intersect with sphere
    print("  Computing sphere intersections...")
    sphere_points = []
    for i, r_w in enumerate(rays_w):
        if i % 100000 == 0:
            print(f"    Progress: {i}/{len(rays_w)} rays ({100*i/len(rays_w):.1f}%)")
        
        b = np.dot(r_w, C_w - O)
        c = np.sum((C_w - O)**2) - sphere_radius**2
        delta = b**2 - c
        
        if delta < 0:
            sphere_points.append([np.nan, np.nan, np.nan])
            continue
        
        lam = -b + np.sqrt(delta)
        P = C_w + lam * r_w
        sphere_points.append(P)
    
    print("  Reshaping result...")
    return np.array(sphere_points).reshape(h, w, 3)

def main():
    print("=" * 70)
    print("PIXEL-TO-SPHERE MATRIX GENERATOR")
    print("=" * 70)
    
    # Check if calibration file exists
    calib_file = Path("camera_calibration_dual.npz")
    if not calib_file.exists():
        print(f"❌ ERROR: Calibration file not found: {calib_file}")
        print(f"   Current directory: {Path.cwd()}")
        return 1
    
    print(f"✓ Found calibration file: {calib_file}")
    
    try:
        # Load calibration
        print("\nLoading calibration data...")
        calib = np.load(str(calib_file))
        K = calib["camera_matrix"]
        d = calib["dist_coeffs"]
        
        print(f"  Camera matrix K:")
        print(f"    fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
        print(f"    cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
        print(f"  Distortion coefficients: {d.shape}")
        
        # Use identity extrinsics (camera at world origin)
        R = np.eye(3)
        t = np.zeros(3)
        print(f"\n✓ Using identity extrinsics (camera at world origin)")
        
        # Image shape from your calibration (720p)
        img_shape = (720, 1280)
        print(f"✓ Image shape: {img_shape[0]}×{img_shape[1]}")
        
        # Sphere parameters
        sphere_radius = 1.0
        print(f"✓ Sphere radius: {sphere_radius}")
        
        # Build sphere matrix
        print("\n" + "=" * 70)
        sphere_matrix = pixel_to_sphere_matrix(K, d, R, t, img_shape, sphere_radius)
        print("=" * 70)
        
        # Check for NaN values
        nan_count = np.isnan(sphere_matrix).sum()
        total_pixels = sphere_matrix.shape[0] * sphere_matrix.shape[1]
        valid_pixels = total_pixels - nan_count // 3
        
        print(f"\n✓ Sphere matrix computed:")
        print(f"    Shape: {sphere_matrix.shape}")
        print(f"    Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
        print(f"    NaN pixels: {nan_count//3} ({100*nan_count/(3*total_pixels):.1f}%)")
        
        # Save result
        output_file = Path("sphere_matrix.npy")
        np.save(str(output_file), sphere_matrix)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        
        print(f"\n✓ Sphere matrix saved: {output_file}")
        print(f"    File size: {file_size_mb:.2f} MB")
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print("\nYou can now use 'sphere_matrix.npy' to map 2D pixels to 3D sphere coordinates.")
        print("Example usage:")
        print("  sphere_matrix = np.load('sphere_matrix.npy')")
        print("  point_3d = sphere_matrix[v, u]  # For pixel (u, v)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())