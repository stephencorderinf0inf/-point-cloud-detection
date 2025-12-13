import numpy as np
import open3d as o3d
from pathlib import Path
import sys

# Path to your existing scan
scan_path = Path(r"D:\Users\Planet UI\world_folder\projects\teleportation-portals\scripts\graphene\camera_tools\scanning\data\point_clouds\scan_3d_bosch_glm42.npz")

print("\n" + "="*80)
print("🔄 PROFESSIONAL POINT CLOUD TO MESH CONVERTER")
print("="*80)

# ============================================================================
# STEP 1: LOAD & ANALYZE POINT CLOUD
# ============================================================================
print(f"\n📂 STEP 1: Loading point cloud...")
data = np.load(scan_path)
points = data['points']
print(f"✓ Loaded {len(points):,} points")

# Check for rotation metadata
if 'angles' in data and 'sessions' in data:
    angles = data['angles']
    sessions = data['sessions']
    unique_sessions = np.unique(sessions)
    
    print(f"\n🔄 ROTATION METADATA DETECTED:")
    print(f"   Total sessions: {len(unique_sessions)}")
    print(f"   Rotation step: {data.get('rotation_step', 'Unknown')}°")
    print(f"   Angular coverage: {angles.min():.1f}° to {angles.max():.1f}°")
    
    # Analyze coverage
    angle_span = angles.max() - angles.min()
    if angle_span >= 270:
        print(f"   ✅ EXCELLENT - Near full 360° coverage ({angle_span:.1f}°)")
    elif angle_span >= 180:
        print(f"   ✅ GOOD - Half+ rotation coverage ({angle_span:.1f}°)")
    elif angle_span >= 90:
        print(f"   ⚠️  LIMITED - Quarter rotation only ({angle_span:.1f}°)")
    else:
        print(f"   ❌ INSUFFICIENT - Very limited angles ({angle_span:.1f}°)")
    
    print(f"\n   📊 Points per angle:")
    for session in unique_sessions:
        mask = sessions == session
        count = np.sum(mask)
        angle = angles[mask][0] if count > 0 else 0
        print(f"      {angle:6.1f}°: {count:,} points")
    
    # Color code points by rotation angle for better visualization
    print(f"\n   → Applying angle-based colors for better reconstruction...")
    import matplotlib.pyplot as plt
    colors = plt.cm.hsv(angles / 360.0)[:, :3]  # HSV colormap (red→green→blue→red)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
else:
    print(f"\n⚠️  NO ROTATION METADATA FOUND")
    print(f"   → Points treated as single viewpoint")
    print(f"   → Rescan with updated scanner for rotation tracking")

# Analyze point cloud quality
print(f"\n📊 Point Cloud Analysis:")
print(f"   Total points: {len(points):,}")
print(f"   Quality assessment:")
if len(points) < 50000:
    print(f"   ❌ VERY SPARSE - Need 100K+ points for good mesh")
    print(f"   → Recommendation: Rescan with ROI + multiple angles")
elif len(points) < 100000:
    print(f"   ⚠️  SPARSE - Minimal quality expected")
    print(f"   → Recommendation: Add more capture angles")
elif len(points) < 300000:
    print(f"   ✅ ADEQUATE - Should produce recognizable mesh")
else:
    print(f"   ✅✅ EXCELLENT - High quality mesh expected")

# ============================================================================
# STEP 2: CLEAN & PREPROCESS (Remove noise, downsample)
# ============================================================================
print(f"\n🔧 STEP 2: Cleaning & Preprocessing...")

# Convert millimeters to meters for proper scale
print("   → Converting units (mm → meters)...")
points = points / 1000.0
print(f"      X: {points[:, 0].min():.3f} to {points[:, 0].max():.3f} m")
print(f"      Y: {points[:, 1].min():.3f} to {points[:, 1].max():.3f} m")
print(f"      Z: {points[:, 2].min():.3f} to {points[:, 2].max():.3f} m")

# Center at origin (improves reconstruction)
print("   → Centering at origin...")
centroid = points.mean(axis=0)
points = points - centroid

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Remove statistical outliers (noise reduction)
print("   → Removing statistical outliers...")
pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
removed = len(pcd.points) - len(pcd_clean.points)
print(f"      Removed {removed:,} noise points ({removed/len(pcd.points)*100:.1f}%)")
print(f"      Kept {len(pcd_clean.points):,} clean points")

# Optional: Downsample if too dense (keeps essential features)
if len(pcd_clean.points) > 500000:
    print("   → Downsampling to improve performance...")
    voxel_size = 0.002  # 2mm voxels
    pcd_clean = pcd_clean.voxel_down_sample(voxel_size)
    print(f"      Downsampled to {len(pcd_clean.points):,} points")

# ============================================================================
# STEP 3: ESTIMATE NORMALS (Required for reconstruction)
# ============================================================================
print(f"\n🔧 STEP 3: Estimating surface normals...")
pcd_clean.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=50)
)
pcd_clean.orient_normals_consistent_tangent_plane(k=15)
print(f"   ✓ Normals computed and oriented")

# ============================================================================
# STEP 4: CHOOSE RECONSTRUCTION METHOD
# ============================================================================
print("\n" + "="*80)
print("🎯 STEP 4: Choose Mesh Reconstruction Method")
print("="*80)
print("\n📐 Available Methods:")
print("  1. Poisson Reconstruction ⭐")
print("     • Smooth, watertight surfaces")
print("     • Fills gaps intelligently")
print("     • Best for: Complete objects, organic shapes")
print("     • Quality: Excellent for sparse data")
print()
print("  2. Ball Pivoting Algorithm")
print("     • Preserves fine details")
print("     • Respects original point positions")
print("     • Best for: Dense scans, detailed features")
print("     • Quality: Good for >100K points")
print()
print("  3. Alpha Shape")
print("     • Exact boundary representation")
print("     • Shows gaps/holes accurately")
print("     • Best for: Complete scans, inspection")
print("     • Quality: Requires dense, uniform data")
print()
print("  4. Delaunay Triangulation (2.5D)")
print("     • Fast, structured mesh")
print("     • Good for terrain/relief surfaces")
print("     • Best for: Single-viewpoint scans")
print("     • Quality: Fast but basic")
print()
print("  5. ALL METHODS (Compare results)")
print("     • Generates all 4 mesh types")
print("     • Choose best one in Blender/MeshLab")

choice = input("\n👉 Select method (1-5) [1 - Poisson]: ").strip() or "1"

# ============================================================================
# STEP 5: CREATE MESH
# ============================================================================
print("\n" + "="*80)
print("🔧 STEP 5: Creating Mesh(es)")
print("="*80)

meshes = []

if choice == "1" or choice == "5":
    # ===== POISSON RECONSTRUCTION =====
    print("\n📐 Method 1: Poisson Reconstruction (Smooth, Watertight)")
    print("   → Building mesh (depth=9, scale=1.1)...")
    
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_clean, depth=9, width=0, scale=1.1, linear_fit=False
    )
    
    # Remove low-density vertices (cleanup)
    print("   → Removing low-density artifacts...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
    
    # Post-process
    mesh_poisson.remove_duplicated_vertices()
    mesh_poisson.remove_duplicated_triangles()
    mesh_poisson.remove_degenerate_triangles()
    mesh_poisson.compute_vertex_normals()
    
    meshes.append(("poisson", mesh_poisson))
    print(f"   ✓ Vertices: {len(mesh_poisson.vertices):,}, Triangles: {len(mesh_poisson.triangles):,}")

if choice == "2" or choice == "5":
    # ===== BALL PIVOTING ALGORITHM =====
    print("\n📐 Method 2: Ball Pivoting (Detail Preservation)")
    print("   → Computing optimal ball radii...")
    
    distances = pcd_clean.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * 1.5, avg_dist * 3, avg_dist * 6]
    print(f"   → Using radii: {radii[0]:.4f}, {radii[1]:.4f}, {radii[2]:.4f} m")
    
    mesh_ball = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd_clean,
        o3d.utility.DoubleVector(radii)
    )
    
    # Post-process
    mesh_ball.remove_duplicated_vertices()
    mesh_ball.remove_duplicated_triangles()
    mesh_ball.remove_degenerate_triangles()
    mesh_ball.compute_vertex_normals()
    
    meshes.append(("ball_pivot", mesh_ball))
    print(f"   ✓ Vertices: {len(mesh_ball.vertices):,}, Triangles: {len(mesh_ball.triangles):,}")

if choice == "3" or choice == "5":
    # ===== ALPHA SHAPE =====
    print("\n📐 Method 3: Alpha Shape (Exact Boundary)")
    print("   → Computing optimal alpha value...")
    
    distances = pcd_clean.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    alpha = avg_dist * 3
    print(f"   → Using alpha: {alpha:.4f} m")
    
    mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd_clean, alpha
    )
    
    # Post-process
    mesh_alpha.remove_duplicated_vertices()
    mesh_alpha.remove_duplicated_triangles()
    mesh_alpha.remove_degenerate_triangles()
    mesh_alpha.compute_vertex_normals()
    
    meshes.append(("alpha_shape", mesh_alpha))
    print(f"   ✓ Vertices: {len(mesh_alpha.vertices):,}, Triangles: {len(mesh_alpha.triangles):,}")

if choice == "4" or choice == "5":
    # ===== DELAUNAY 2.5D =====
    print("\n📐 Method 4: Delaunay Triangulation (2.5D Terrain)")
    print("   → Projecting to XY plane and triangulating...")
    
    # Project to 2D and perform Delaunay
    points_2d = np.asarray(pcd_clean.points)[:, :2]  # Use only X, Y
    from scipy.spatial import Delaunay
    tri = Delaunay(points_2d)
    
    mesh_delaunay = o3d.geometry.TriangleMesh()
    mesh_delaunay.vertices = pcd_clean.points
    mesh_delaunay.triangles = o3d.utility.Vector3iVector(tri.simplices)
    
    # Post-process
    mesh_delaunay.remove_duplicated_vertices()
    mesh_delaunay.remove_duplicated_triangles()
    mesh_delaunay.remove_degenerate_triangles()
    mesh_delaunay.compute_vertex_normals()
    
    meshes.append(("delaunay", mesh_delaunay))
    print(f"   ✓ Vertices: {len(mesh_delaunay.vertices):,}, Triangles: {len(mesh_delaunay.triangles):,}")

# ============================================================================
# STEP 6: FILL GAPS & SMOOTH (Post-processing)
# ============================================================================
print("\n" + "="*80)
print("🔧 STEP 6: Post-Processing (Optional Smoothing)")
print("="*80)

smooth = input("\n👉 Apply Laplacian smoothing? (y/N): ").strip().lower()
if smooth == 'y':
    iterations = int(input("   Smoothing iterations (1-10) [3]: ").strip() or "3")
    for name, mesh in meshes:
        print(f"   → Smoothing {name} mesh ({iterations} iterations)...")
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
        mesh.compute_vertex_normals()

# ============================================================================
# STEP 7: SAVE MESHES
# ============================================================================
print("\n" + "="*80)
print("💾 STEP 7: Saving Mesh(es)")
print("="*80)

for suffix, mesh in meshes:
    mesh_path = scan_path.parent / f"scan_3d_bosch_glm42_{suffix}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    
    print(f"\n✓ Saved: {mesh_path.name}")
    print(f"   Vertices:  {len(mesh.vertices):,}")
    print(f"   Triangles: {len(mesh.triangles):,}")
    print(f"   Surface area: {mesh.get_surface_area():.4f} m²")
    
    if mesh.is_watertight():
        print(f"   Volume: {mesh.get_volume():.6f} m³")
        print(f"   Watertight: ✅ Yes (suitable for 3D printing)")
    else:
        print(f"   Watertight: ⚠️  No (has holes)")

# ============================================================================
# STEP 8: NEXT STEPS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("📂 NEXT STEPS: View Your Mesh")
print("="*80)

print("\n🔷 Blender (Recommended for editing):")
print("   1. File → Import → PLY (.ply)")
print("   2. Select mesh file(s)")
print("   3. Press Numpad . to frame object")
print("   4. Tab → Edit Mode for modifications")

print("\n🔶 MeshLab (Recommended for inspection):")
print("   1. File → Import Mesh")
print("   2. Filters → Remeshing for optimization")
print("   3. Render → Show quality histograms")

print("\n🔸 CloudCompare (For point cloud comparison):")
print("   1. Open original .npz as point cloud")
print("   2. Import mesh for side-by-side comparison")
print("   3. Tools → Distances → Cloud/Mesh distance")

print("\n💡 RECOMMENDATIONS:")
print(f"   Current point count: {len(points):,}")
if len(points) < 100000:
    print("   ⚠️  Your scan is SPARSE - for better results:")
    print("      • Use ROI ('r' key) to focus on object")
    print("      • Enable auto-capture ('a' key) - captures 3x per angle")
    print("      • Rotate object 8-12 times (30-45° each)")
    print("      • Target: 150K-300K points for good mesh")
else:
    print("   ✅ Point count is adequate")
    print("   → If mesh still looks wrong, try Poisson method (fills gaps)")

print("\n" + "="*80)
print("✅ Conversion Complete!")
print("="*80 + "\n")

def convert_npz_to_mesh(input_file, output_name=None, method='poisson', depth=9):
    """Convert NPZ point cloud to 3D mesh."""
    
    # Load NPZ file
    data = np.load(input_file)
    points = data['points']
    
    print(f"\n📂 Loaded point cloud:")
    print(f"   File: {input_file}")
    print(f"   Points: {len(points):,}")
    
    # 🎨 NEW: Check for color data
    has_colors = 'colors' in data
    if has_colors:
        colors = data['colors']
        print(f"   Colors: {len(colors):,} RGB values")
        print(f"   🎨 COLOR MODE: Point cloud has RGB data!")
    else:
        colors = None
        print(f"   Colors: None (will use default/angle-based colors)")
    
    # Check for rotation metadata
    if 'angles' in data and 'sessions' in data:
        angles = data['angles']
        sessions = data['sessions']
        unique_sessions = np.unique(sessions)
        
        print(f"\n🔄 ROTATION METADATA DETECTED:")
        print(f"   Total sessions: {len(unique_sessions)}")
        print(f"   Rotation step: {data.get('rotation_step', 'Unknown')}°")
        print(f"   Angular coverage: {angles.min():.1f}° to {angles.max():.1f}°")
        
        # Analyze coverage
        angle_span = angles.max() - angles.min()
        if angle_span >= 270:
            print(f"   ✅ EXCELLENT - Near full 360° coverage ({angle_span:.1f}°)")
        elif angle_span >= 180:
            print(f"   ✅ GOOD - Half+ rotation coverage ({angle_span:.1f}°)")
        elif angle_span >= 90:
            print(f"   ⚠️  LIMITED - Quarter rotation only ({angle_span:.1f}°)")
        else:
            print(f"   ❌ INSUFFICIENT - Very limited angles ({angle_span:.1f}°)")
        
        print(f"\n   📊 Points per angle:")
        for session in unique_sessions:
            mask = sessions == session
            count = np.sum(mask)
            angle = angles[mask][0] if count > 0 else 0
            print(f"      {angle:6.1f}°: {count:,} points")
        
        # Color code points by rotation angle for better visualization
        print(f"\n   → Applying angle-based colors for better reconstruction...")
        import matplotlib.pyplot as plt
        colors = plt.cm.hsv(angles / 360.0)[:, :3]  # HSV colormap (red→green→blue→red)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
    else:
        print(f"\n⚠️  NO ROTATION METADATA FOUND")
        print(f"   → Points treated as single viewpoint")
        print(f"   → Rescan with updated scanner for rotation tracking")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 🎨 NEW: Apply colors if available
    if has_colors:
        # Normalize RGB values (0-255 → 0.0-1.0)
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        print(f"\n🎨 Applied RGB colors to point cloud")
    elif 'angles' in data:
        # Fallback: Color by rotation angle
        colors_from_angles = plt.cm.hsv(angles / 360.0)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors_from_angles)
        print(f"\n🔄 Colored by rotation angle (no RGB data)")
    else:
        # No colors at all - use default gray
        print(f"\n⚪ No color data - using default gray")
    
    # Analyze point cloud quality
    print(f"\n📊 Point Cloud Analysis:")
    print(f"   Total points: {len(points):,}")
    print(f"   Quality assessment:")
    if len(points) < 50000:
        print(f"   ❌ VERY SPARSE - Need 100K+ points for good mesh")
        print(f"   → Recommendation: Rescan with ROI + multiple angles")
    elif len(points) < 100000:
        print(f"   ⚠️  SPARSE - Minimal quality expected")
        print(f"   → Recommendation: Add more capture angles")
    elif len(points) < 300000:
        print(f"   ✅ ADEQUATE - Should produce recognizable mesh")
    else:
        print(f"   ✅✅ EXCELLENT - High quality mesh expected")

    # ============================================================================
    # STEP 2: CLEAN & PREPROCESS (Remove noise, downsample)
    # ============================================================================
    print(f"\n🔧 STEP 2: Cleaning & Preprocessing...")

    # Convert millimeters to meters for proper scale
    print("   → Converting units (mm → meters)...")
    points = points / 1000.0
    print(f"      X: {points[:, 0].min():.3f} to {points[:, 0].max():.3f} m")
    print(f"      Y: {points[:, 1].min():.3f} to {points[:, 1].max():.3f} m")
    print(f"      Z: {points[:, 2].min():.3f} to {points[:, 2].max():.3f} m")

    # Center at origin (improves reconstruction)
    print("   → Centering at origin...")
    centroid = points.mean(axis=0)
    points = points - centroid

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Remove statistical outliers (noise reduction)
    print("   → Removing statistical outliers...")
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    removed = len(pcd.points) - len(pcd_clean.points)
    print(f"      Removed {removed:,} noise points ({removed/len(pcd.points)*100:.1f}%)")
    print(f"      Kept {len(pcd_clean.points):,} clean points")

    # Optional: Downsample if too dense (keeps essential features)
    if len(pcd_clean.points) > 500000:
        print("   → Downsampling to improve performance...")
        voxel_size = 0.002  # 2mm voxels
        pcd_clean = pcd_clean.voxel_down_sample(voxel_size)
        print(f"      Downsampled to {len(pcd_clean.points):,} points")

    # ============================================================================
    # STEP 3: ESTIMATE NORMALS (Required for reconstruction)
    # ============================================================================
    print(f"\n🔧 STEP 3: Estimating surface normals...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=50)
    )
    pcd_clean.orient_normals_consistent_tangent_plane(k=15)
    print(f"   ✓ Normals computed and oriented")

    # ============================================================================
    # STEP 4: CHOOSE RECONSTRUCTION METHOD
    # ============================================================================
    print("\n" + "="*80)
    print("🎯 STEP 4: Choose Mesh Reconstruction Method")
    print("="*80)
    print("\n📐 Available Methods:")
    print("  1. Poisson Reconstruction ⭐")
    print("     • Smooth, watertight surfaces")
    print("     • Fills gaps intelligently")
    print("     • Best for: Complete objects, organic shapes")
    print("     • Quality: Excellent for sparse data")
    print()
    print("  2. Ball Pivoting Algorithm")
    print("     • Preserves fine details")
    print("     • Respects original point positions")
    print("     • Best for: Dense scans, detailed features")
    print("     • Quality: Good for >100K points")
    print()
    print("  3. Alpha Shape")
    print("     • Exact boundary representation")
    print("     • Shows gaps/holes accurately")
    print("     • Best for: Complete scans, inspection")
    print("     • Quality: Requires dense, uniform data")
    print()
    print("  4. Delaunay Triangulation (2.5D)")
    print("     • Fast, structured mesh")
    print("     • Good for terrain/relief surfaces")
    print("     • Best for: Single-viewpoint scans")
    print("     • Quality: Fast but basic")
    print()
    print("  5. ALL METHODS (Compare results)")
    print("     • Generates all 4 mesh types")
    print("     • Choose best one in Blender/MeshLab")

    choice = input("\n👉 Select method (1-5) [1 - Poisson]: ").strip() or "1"

    # ============================================================================
    # STEP 5: CREATE MESH
    # ============================================================================
    print("\n" + "="*80)
    print("🔧 STEP 5: Creating Mesh(es)")
    print("="*80)

    meshes = []

    if choice == "1" or choice == "5":
        # ===== POISSON RECONSTRUCTION =====
        print("\n📐 Method 1: Poisson Reconstruction (Smooth, Watertight)")
        print("   → Building mesh (depth=9, scale=1.1)...")
        
        mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=9, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low-density vertices (cleanup)
        print("   → Removing low-density artifacts...")
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh_poisson.remove_vertices_by_mask(vertices_to_remove)
        
        # Post-process
        mesh_poisson.remove_duplicated_vertices()
        mesh_poisson.remove_duplicated_triangles()
        mesh_poisson.remove_degenerate_triangles()
        mesh_poisson.compute_vertex_normals()
        
        meshes.append(("poisson", mesh_poisson))
        print(f"   ✓ Vertices: {len(mesh_poisson.vertices):,}, Triangles: {len(mesh_poisson.triangles):,}")

    if choice == "2" or choice == "5":
        # ===== BALL PIVOTING ALGORITHM =====
        print("\n📐 Method 2: Ball Pivoting (Detail Preservation)")
        print("   → Computing optimal ball radii...")
        
        distances = pcd_clean.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 1.5, avg_dist * 3, avg_dist * 6]
        print(f"   → Using radii: {radii[0]:.4f}, {radii[1]:.4f}, {radii[2]:.4f} m")
        
        mesh_ball = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_clean,
            o3d.utility.DoubleVector(radii)
        )
        
        # Post-process
        mesh_ball.remove_duplicated_vertices()
        mesh_ball.remove_duplicated_triangles()
        mesh_ball.remove_degenerate_triangles()
        mesh_ball.compute_vertex_normals()
        
        meshes.append(("ball_pivot", mesh_ball))
        print(f"   ✓ Vertices: {len(mesh_ball.vertices):,}, Triangles: {len(mesh_ball.triangles):,}")

    if choice == "3" or choice == "5":
        # ===== ALPHA SHAPE =====
        print("\n📐 Method 3: Alpha Shape (Exact Boundary)")
        print("   → Computing optimal alpha value...")
        
        distances = pcd_clean.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        alpha = avg_dist * 3
        print(f"   → Using alpha: {alpha:.4f} m")
        
        mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd_clean, alpha
        )
        
        # Post-process
        mesh_alpha.remove_duplicated_vertices()
        mesh_alpha.remove_duplicated_triangles()
        mesh_alpha.remove_degenerate_triangles()
        mesh_alpha.compute_vertex_normals()
        
        meshes.append(("alpha_shape", mesh_alpha))
        print(f"   ✓ Vertices: {len(mesh_alpha.vertices):,}, Triangles: {len(mesh_alpha.triangles):,}")

    if choice == "4" or choice == "5":
        # ===== DELAUNAY 2.5D =====
        print("\n📐 Method 4: Delaunay Triangulation (2.5D Terrain)")
        print("   → Projecting to XY plane and triangulating...")
        
        # Project to 2D and perform Delaunay
        points_2d = np.asarray(pcd_clean.points)[:, :2]  # Use only X, Y
        from scipy.spatial import Delaunay
        tri = Delaunay(points_2d)
        
        mesh_delaunay = o3d.geometry.TriangleMesh()
        mesh_delaunay.vertices = pcd_clean.points
        mesh_delaunay.triangles = o3d.utility.Vector3iVector(tri.simplices)
        
        # Post-process
        mesh_delaunay.remove_duplicated_vertices()
        mesh_delaunay.remove_duplicated_triangles()
        mesh_delaunay.remove_degenerate_triangles()
        mesh_delaunay.compute_vertex_normals()
        
        meshes.append(("delaunay", mesh_delaunay))
        print(f"   ✓ Vertices: {len(mesh_delaunay.vertices):,}, Triangles: {len(mesh_delaunay.triangles):,}")

    # ============================================================================
    # STEP 6: FILL GAPS & SMOOTH (Post-processing)
    # ============================================================================
    print("\n" + "="*80)
    print("🔧 STEP 6: Post-Processing (Optional Smoothing)")
    print("="*80)

    smooth = input("\n👉 Apply Laplacian smoothing? (y/N): ").strip().lower()
    if smooth == 'y':
        iterations = int(input("   Smoothing iterations (1-10) [3]: ").strip() or "3")
        for name, mesh in meshes:
            print(f"   → Smoothing {name} mesh ({iterations} iterations)...")
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
            mesh.compute_vertex_normals()

    # ============================================================================
    # STEP 7: SAVE MESHES
    # ============================================================================
    print("\n" + "="*80)
    print("💾 STEP 7: Saving Mesh(es)")
    print("="*80)

    for suffix, mesh in meshes:
        mesh_path = scan_path.parent / f"scan_3d_bosch_glm42_{suffix}.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        
        print(f"\n✓ Saved: {mesh_path.name}")
        print(f"   Vertices:  {len(mesh.vertices):,}")
        print(f"   Triangles: {len(mesh.triangles):,}")
        print(f"   Surface area: {mesh.get_surface_area():.4f} m²")
        
        if mesh.is_watertight():
            print(f"   Volume: {mesh.get_volume():.6f} m³")
            print(f"   Watertight: ✅ Yes (suitable for 3D printing)")
        else:
            print(f"   Watertight: ⚠️  No (has holes)")

    # ============================================================================
    # STEP 8: NEXT STEPS & RECOMMENDATIONS
    # ============================================================================
    print("\n" + "="*80)
    print("📂 NEXT STEPS: View Your Mesh")
    print("="*80)

    print("\n🔷 Blender (Recommended for editing):")
    print("   1. File → Import → PLY (.ply)")
    print("   2. Select mesh file(s)")
    print("   3. Press Numpad . to frame object")
    print("   4. Tab → Edit Mode for modifications")

    print("\n🔶 MeshLab (Recommended for inspection):")
    print("   1. File → Import Mesh")
    print("   2. Filters → Remeshing for optimization")
    print("   3. Render → Show quality histograms")

    print("\n🔸 CloudCompare (For point cloud comparison):")
    print("   1. Open original .npz as point cloud")
    print("   2. Import mesh for side-by-side comparison")
    print("   3. Tools → Distances → Cloud/Mesh distance")

    print("\n💡 RECOMMENDATIONS:")
    print(f"   Current point count: {len(points):,}")
    if len(points) < 100000:
        print("   ⚠️  Your scan is SPARSE - for better results:")
        print("      • Use ROI ('r' key) to focus on object")
        print("      • Enable auto-capture ('a' key) - captures 3x per angle")
        print("      • Rotate object 8-12 times (30-45° each)")
        print("      • Target: 150K-300K points for good mesh")
    else:
        print("   ✅ Point count is adequate")
        print("   → If mesh still looks wrong, try Poisson method (fills gaps)")

    print("\n" + "="*80)
    print("✅ Conversion Complete!")
    print("="*80 + "\n")