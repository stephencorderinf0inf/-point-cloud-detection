"""
3D Point Cloud Viewer
View .npz scans from laser_3d_scanner_advanced.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

def view_scan(filename=None):
    """View a 3D scan point cloud."""
    
    # Default to most recent scan
    if filename is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        filename = os.path.join(data_dir, "scan_3d_bosch_glm42.npz")
    
    if not os.path.exists(filename):
        print(f"[X] File not found: {filename}")
        print("\nAvailable scans:")
        data_dir = os.path.dirname(filename)
        for f in os.listdir(data_dir):
            if f.endswith('.npz'):
                print(f"   - {f}")
        return
    
    # Load the scan
    print(f"\n[LOADING] {os.path.basename(filename)}")
    data = np.load(filename)
    points = data['points']
    
    print(f"[CHECK] Loaded {len(points)} 3D points")
    print(f"        X range: {points[:,0].min():.1f} to {points[:,0].max():.1f} mm")
    print(f"        Y range: {points[:,1].min():.1f} to {points[:,1].max():.1f} mm")
    print(f"        Z range: {points[:,2].min():.1f} to {points[:,2].max():.1f} mm")
    
    # Create figure with better size
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by depth (Z-axis)
    scatter = ax.scatter(points[:,0], points[:,1], points[:,2], 
                        c=points[:,2], cmap='viridis', s=1, alpha=0.6)
    
    # Labels and title
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Z (mm) - Distance', fontsize=10)
    ax.set_title(f'3D Scan: {len(points)} points', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Depth (mm)', rotation=270, labelpad=15)
    
    # Equal aspect ratio
    max_range = np.array([
        points[:,0].max()-points[:,0].min(),
        points[:,1].max()-points[:,1].min(),
        points[:,2].max()-points[:,2].min()
    ]).max() / 2.0
    
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Better viewing angle
    ax.view_init(elev=20, azim=45)
    
    print("\n[TIP] Controls:")
    print("   - LEFT MOUSE: Rotate view")
    print("   - RIGHT MOUSE: Zoom")
    print("   - MIDDLE MOUSE: Pan")
    print("   - Close window to exit\n")
    
    plt.tight_layout()
    plt.show()

def view_multiple_views(filename=None):
    """Show multiple viewing angles side-by-side."""
    
    # Default to most recent scan
    if filename is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        filename = os.path.join(data_dir, "scan_3d_bosch_glm42.npz")
    
    if not os.path.exists(filename):
        print(f"[X] File not found: {filename}")
        return
    
    # Load the scan
    print(f"\n[LOADING] {os.path.basename(filename)}")
    data = np.load(filename)
    points = data['points']
    
    print(f"[CHECK] Loaded {len(points)} 3D points")
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    angles = [
        (20, 45, "Perspective View"),
        (0, 0, "Front View (XZ)"),
        (0, 90, "Side View (YZ)"),
        (90, 0, "Top View (XY)")
    ]
    
    for i, (elev, azim, title) in enumerate(angles, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        
        scatter = ax.scatter(points[:,0], points[:,1], points[:,2], 
                           c=points[:,2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        
        # Equal aspect ratio
        max_range = np.array([
            points[:,0].max()-points[:,0].min(),
            points[:,1].max()-points[:,1].min(),
            points[:,2].max()-points[:,2].min()
        ]).max() / 2.0
        
        mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
        mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
        mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(f'3D Scan: {len(points)} points - Multiple Views', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("3D POINT CLOUD VIEWER")
    print("=" * 60)
    print("\nOptions:")
    print("   1 - Single interactive view")
    print("   2 - Multiple views (front/side/top)")
    print("   3 - Load custom file")
    
    choice = input("\nSelect option (1-3, default=1): ").strip()
    
    if choice == '2':
        view_multiple_views()
    elif choice == '3':
        filepath = input("Enter .npz file path: ").strip()
        view_scan(filepath)
    else:
        view_scan()