"""
Master Reconstruction Script
Analyzes scan + Creates 4 different mesh versions
"""

import numpy as np
import open3d as o3d
import time
from pathlib import Path

print("=" * 70)
print("MASTER POINT CLOUD RECONSTRUCTION")
print("=" * 70)

# ============================================================================
# STEP 1: ANALYZE SCAN QUALITY
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: ANALYZING SCAN QUALITY")
print("=" * 70)

print("\n[1/1] Loading point cloud...")
data = np.load('point_clouds/scan_3d_bosch_glm42.npz')
points = data['points']

print(f"\n✓ Total points: {len(points):,}")

# Calculate bounding box
min_coords = points.min(axis=0)
max_coords = points.max(axis=0)
size = max_coords - min_coords

print(f"\n📐 Bounding box size:")
print(f"   X: {size[0]:.1f} mm (width)")
print(f"   Y: {size[1]:.1f} mm (height)")
print(f"   Z: {size[2]:.1f} mm (depth)")

# Calculate point density
volume = size[0] * size[1] * size[2]
density = len(points) / volume

print(f"\n📈 Point density: {density:.6f} points/mm³")

density_rating = ""
density_rating_plain = ""  # Plain text version for file writing
needs_more = False

if density < 0.01:
    density_rating = "⚠️  VERY SPARSE"
    density_rating_plain = "VERY SPARSE"
    recommendation = "Need 3-5x more points (50,000+)"
    needs_more = True
elif density < 0.1:
    density_rating = "⚠️  SPARSE"
    density_rating_plain = "SPARSE"
    recommendation = "Could use 2-3x more points (40,000+)"
    needs_more = True
elif density < 1.0:
    density_rating = "✓ MODERATE"
    density_rating_plain = "MODERATE"
    recommendation = "Should work for basic reconstruction"
else:
    density_rating = "✓ DENSE"
    density_rating_plain = "DENSE"
    recommendation = "Good quality expected!"

print(f"   {density_rating}")
print(f"   Recommendation: {recommendation}")

# Save analysis report (UTF-8 encoding to handle all characters)
report_path = Path('reports')
report_path.mkdir(exist_ok=True)

with open(report_path / 'scan_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("SCAN QUALITY ANALYSIS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total Points: {len(points):,}\n")
    f.write(f"Bounding Box: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm\n")
    f.write(f"Volume: {volume:.1f} mm³\n")
    f.write(f"Density: {density:.6f} points/mm³\n")
    f.write(f"Rating: {density_rating_plain}\n")
    f.write(f"Recommendation: {recommendation}\n")

print(f"\n✓ Analysis saved to: reports/scan_analysis.txt")

time.sleep(2)

# ============================================================================
# STEP 2: IMPROVED POISSON RECONSTRUCTION (3 QUALITY LEVELS)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: POISSON SURFACE RECONSTRUCTION (3 QUALITY LEVELS)")
print("=" * 70)

# Prepare point cloud
print("\n[1/5] Creating point cloud object...")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Clean outliers
print("[2/5] Removing outlier points...")
pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"   ✓ Removed {len(points) - len(pcd_clean.points):,} outliers")
print(f"   ✓ Clean points: {len(pcd_clean.points):,}")

# Normalize density
print("[3/5] Normalizing point density...")
voxel_size = 2.0  # 2mm voxel size
pcd_down = pcd_clean.voxel_down_sample(voxel_size)
print(f"   ✓ Downsampled to {len(pcd_down.points):,} points")

# Estimate normals
print("[4/5] Estimating surface normals...")
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=50)
)
pcd_down.orient_normals_consistent_tangent_plane(50)
print("   ✓ Normals estimated")

# Reconstruct at 3 quality levels
print("[5/5] Creating meshes at 3 quality levels...")

quality_settings = [
    ('low', 7, 0.10),      # depth=7, remove bottom 10%
    ('medium', 8, 0.15),   # depth=8, remove bottom 15%
    ('high', 9, 0.20)      # depth=9, remove bottom 20%
]

poisson_meshes = {}

for quality_name, depth, density_threshold in quality_settings:
    print(f"\n   Creating {quality_name.upper()} quality mesh (depth={depth})...")
    start_time = time.time()
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=depth, width=0, scale=1.1, linear_fit=False
    )
    
    # Clean mesh
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Save
    output_path = f'point_clouds/scan_poisson_{quality_name}.obj'
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    elapsed = time.time() - start_time
    poisson_meshes[quality_name] = {
        'path': output_path,
        'vertices': len(mesh.vertices),
        'faces': len(mesh.triangles),
        'time': elapsed
    }
    
    print(f"   ✓ {quality_name.upper()}: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} faces ({elapsed:.1f}s)")

print("\n✓ All Poisson meshes created!")

time.sleep(2)

# ============================================================================
# STEP 3: BALL PIVOTING RECONSTRUCTION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: BALL PIVOTING RECONSTRUCTION")
print("=" * 70)

print("\n[1/4] Preparing point cloud for Ball Pivoting...")
# Use the cleaned point cloud from Step 2
print(f"   ✓ Using {len(pcd_clean.points):,} clean points")

# Re-estimate normals for Ball Pivoting
print("[2/4] Estimating normals for Ball Pivoting...")
pcd_clean.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=50)
)
pcd_clean.orient_normals_consistent_tangent_plane(50)
print("   ✓ Normals estimated")

# Calculate optimal ball radii
print("[3/4] Calculating optimal ball radii...")
distances = pcd_clean.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radii = [avg_dist * 1.5, avg_dist * 2, avg_dist * 3, avg_dist * 4]
print(f"   ✓ Ball radii: {', '.join([f'{r:.2f}mm' for r in radii])}")

# Reconstruct
print("[4/4] Creating mesh with Ball Pivoting...")
start_time = time.time()

try:
    mesh_bp = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_clean, o3d.utility.DoubleVector(radii)
    )
    
    # Clean
    mesh_bp.remove_degenerate_triangles()
    mesh_bp.remove_duplicated_triangles()
    mesh_bp.remove_duplicated_vertices()
    mesh_bp.remove_non_manifold_edges()
    
    # Save
    output_path_bp = 'point_clouds/scan_ballpivot.obj'
    o3d.io.write_triangle_mesh(output_path_bp, mesh_bp)
    
    elapsed = time.time() - start_time
    
    ball_pivot_result = {
        'path': output_path_bp,
        'vertices': len(mesh_bp.vertices),
        'faces': len(mesh_bp.triangles),
        'time': elapsed,
        'success': True
    }
    
    print(f"   ✓ BALL PIVOTING: {len(mesh_bp.vertices):,} vertices, {len(mesh_bp.triangles):,} faces ({elapsed:.1f}s)")

except Exception as e:
    print(f"   ⚠️  Ball Pivoting failed: {e}")
    ball_pivot_result = {'success': False}

time.sleep(2)

# ============================================================================
# FINAL SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 70)
print("RECONSTRUCTION COMPLETE!")
print("=" * 70)

print("\n📊 SUMMARY:")
print(f"\n✓ Analyzed {len(points):,} points")
print(f"✓ Point density: {density:.6f} points/mm³ ({density_rating})")
print(f"✓ Created {len(poisson_meshes) + (1 if ball_pivot_result.get('success') else 0)} mesh files")

print("\n📁 FILES CREATED:")
print("\n   Poisson Reconstruction:")
for quality, info in poisson_meshes.items():
    print(f"      • {info['path']}")
    print(f"        {info['vertices']:,} vertices | {info['faces']:,} faces | {info['time']:.1f}s")

if ball_pivot_result.get('success'):
    print("\n   Ball Pivoting:")
    info = ball_pivot_result
    print(f"      • {info['path']}")
    print(f"        {info['vertices']:,} vertices | {info['faces']:,} faces | {info['time']:.1f}s")

print("\n🎯 RECOMMENDATIONS:")
print("\n   1. VIEW ALL MESHES:")
print("      - Double-click each .obj file to open in Windows 3D Viewer")
print("      - Or import into Blender to compare side-by-side")

print("\n   2. WHICH MESH TO USE:")
if density < 0.1:
    print("      ⭐ Try 'scan_ballpivot.obj' FIRST (works better with sparse data)")
    print("      ⭐ Then try 'scan_poisson_medium.obj' as backup")
else:
    print("      ⭐ Try 'scan_poisson_medium.obj' FIRST (balanced quality/speed)")
    print("      ⭐ If too rough, use 'scan_poisson_high.obj'")
    print("      ⭐ If too smooth, try 'scan_ballpivot.obj'")

if needs_more:
    print("\n   3. ⚠️  IMPROVE SCAN QUALITY:")
    print(f"      Your current scan is {density_rating}")
    print(f"      {recommendation}")
    print("\n      TO RE-SCAN WITH MORE POINTS:")
    print("      1. Run: python laser_3d_scanner_advanced.py")
    print("      2. Choose Option 3 (select existing project)")
    print("      3. Choose 'no' to APPEND more points")
    print("      4. Press SPACE 20-30 times from different angles")
    print("      5. Press 's' to save")
    print("      6. Re-run this script to create better meshes!")
else:
    print("\n   3. ✓ SCAN QUALITY IS GOOD:")
    print("      Your point density is sufficient for reconstruction")
    print("      If meshes still don't match perfectly, try:")
    print("      - Adjusting object lighting/position")
    print("      - Scanning specific problem areas more densely")

print("\n" + "=" * 70)
print("✓ All reconstructions complete!")
print("=" * 70)

# Create summary report (UTF-8 encoding)
with open(report_path / 'reconstruction_summary.txt', 'w', encoding='utf-8') as f:
    f.write("RECONSTRUCTION SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Input Points: {len(points):,}\n")
    f.write(f"Point Density: {density:.6f} points/mm³\n")
    f.write(f"Quality Rating: {density_rating_plain}\n\n")
    f.write("MESHES CREATED:\n\n")
    
    for quality, info in poisson_meshes.items():
        f.write(f"  Poisson {quality.upper()}:\n")
        f.write(f"    File: {info['path']}\n")
        f.write(f"    Vertices: {info['vertices']:,}\n")
        f.write(f"    Faces: {info['faces']:,}\n")
        f.write(f"    Time: {info['time']:.1f}s\n\n")
    
    if ball_pivot_result.get('success'):
        info = ball_pivot_result
        f.write(f"  Ball Pivoting:\n")
        f.write(f"    File: {info['path']}\n")
        f.write(f"    Vertices: {info['vertices']:,}\n")
        f.write(f"    Faces: {info['faces']:,}\n")
        f.write(f"    Time: {info['time']:.1f}s\n\n")
    
    f.write(f"\nRecommendation: {recommendation}\n")

print(f"\n✓ Summary saved to: reports/reconstruction_summary.txt")