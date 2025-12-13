import numpy as np

def backproject_to_plane(K, R, t, n, X0, x_pix):
    """
    Back-project pixel coordinates to 3D points on a plane.
    K: camera matrix (3x3)
    R: rotation matrix (3x3)
    t: translation vector (3x1)
    n: plane normal (3,)
    X0: point on plane (3,)
    x_pix: Nx2 array of pixel coordinates
    Returns: Nx3 array of 3D points on the plane
    """
    N = x_pix.shape[0]
    x_h = np.hstack([x_pix, np.ones((N,1))])  # homogeneous
    rays_cam = np.linalg.inv(K) @ x_h.T       # rays in camera coords
    rays_world = R.T @ rays_cam               # rotate into world
    C = -R.T @ t.reshape(-1, 1)               # camera center
    n = n / np.linalg.norm(n)
    # Solve lambda per ray
    num = (X0.reshape(3,1) - C).T @ n.reshape(3,1)
    den = rays_world.T @ n.reshape(3,1)
    lam = num / den
    X = C.T + (lam * rays_world.T)
    return X  # Nx3 world points