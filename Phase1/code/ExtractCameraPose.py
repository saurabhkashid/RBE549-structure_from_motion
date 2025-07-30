import numpy as np

"""
E = [T]*R 
E = K.T @ F @ K
Essential is in camera frame and fundamental is in image frame
F = K-T @ E @ k-
"""
def get_camera_pose(essential_mat):
    # take the svd of the essential
    UE, SE, VE = np.linalg.svd(essential_mat)

    # the left null space of E is the epipole of the other camera
    t = UE[:,2]

    # intermediate matrix
    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    R2 = UE @ W.T @ VE # W.T is W inverse
    R1 = UE @ W @ VE

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # The four pairs of (R, t)
    # The convention typically uses the third column of U for t, and -t
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]