import numpy as np

def project_points(K, R, t, X):
    """
    Project 3D points X (Nx3) to image using camera intrinsics and pose.
    K: camera matrix (3x3)
    R: rotation matrix (3x3)
    t: translation vector (3x1)
    X: Nx3 array of 3D points
    Returns: Nx2 array of pixel coordinates
    """
    X = X.reshape(-1, 3)
    X_h = np.hstack([X, np.ones((X.shape[0],1))])
    # Transform points to camera frame
    X_cam = (R @ X.T + t.reshape(3,1)).T
    # Project to image
    x = (K @ X_cam.T).T
    x_img = x[:,:2] / x[:,2:3]
    return x_img