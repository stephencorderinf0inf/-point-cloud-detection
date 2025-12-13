"""
Convert Point Cloud to Surface Mesh using Open3D
"""

import numpy as np
import open3d as o3d

# Load your scan
print("Loading point cloud...")
data = np.load('point_clouds/scan_3d_bosch_glm42.npz')
points = data['points']

print(f"✓ Loaded {len(points)} points")

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Estimate normals (needed for surface reconstruction)
print("Estimating normals...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(30)

# Poisson Surface Reconstruction
print("Creating surface mesh (this may take a minute)...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Remove low-density vertices (noise)
print("Cleaning mesh...")
vertices_to_remove = densities < np.quantile(densities, 0.1)
mesh.remove_vertices_by_mask(vertices_to_remove)

# Save mesh
output_path = 'point_clouds/scan_3d_mesh.obj'
o3d.io.write_triangle_mesh(output_path, mesh)

print(f"\n✓ Mesh saved to: {output_path}")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Triangles: {len(mesh.triangles)}")
print("\n✓ You can now import this mesh into Blender!")

# Optional: Visualize
print("\nPress 'Q' to close visualization...")
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)