import cv2
import numpy as np

# --- Parameters ---
square_size = 20.0  # mm, adjust to your checkerboard
pattern_size = (7, 7)  # inner corners (rows, cols)
theta = np.deg2rad(45)  # rotation angle for second board

# --- Helper: rotation about hinge axis ---
def rotation_about_axis(axis, angle):
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])
    return R

# --- Generate object points for board 1 ---
def generate_board1_points(pattern_size, square_size):
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp

# --- Generate board 2 points from hinge line ---
def generate_board2_points(board1_points, hinge_indices, square_size, theta):
    # hinge line: pick indices along one edge (e.g., last column)
    hinge_points = board1_points[hinge_indices]
    hinge_origin = hinge_points[0]  # anchor corner
    hinge_axis = hinge_points[-1] - hinge_points[0]  # vector along hinge line

    R = rotation_about_axis(hinge_axis, theta)

    # build second board grid in its local plane
    objp2_local = np.zeros_like(board1_points)
    objp2_local[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp2_local *= square_size

    # rotate into world frame about hinge
    objp2 = (R @ objp2_local.T).T + hinge_origin
    return objp2

# --- Example usage ---
# Board 1 object points
objp1 = generate_board1_points(pattern_size, square_size)

# Suppose hinge is last column of corners
hinge_indices = [i*pattern_size[1] + (pattern_size[1]-1) for i in range(pattern_size[0])]

# Board 2 object points auto-generated
objp2 = generate_board2_points(objp1, hinge_indices, square_size, theta)

# --- Calibration pipeline ---
# Detect corners for board 1
img = cv2.imread("checkerboard_image.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, pattern_size)

if ret:
    # refine corners
    corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # PnP for board 1
    retval, rvec1, tvec1 = cv2.solvePnP(objp1, corners_subpix, K, distCoeffs)

    # If board 2 visible, detect corners and run PnP
    # Else: auto-generate pose using objp2 and hinge transform
