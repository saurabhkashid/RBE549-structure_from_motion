import numpy as np


def normalize_points(points):
    """
    Normalizes 2D points to improve numerical stability in the 8-point algorithm.
    Normalization involves translating the centroid to the origin and scaling
    so that the average distance from the origin is sqrt(2).

    Args:
        points (numpy.ndarray): N x 2 array of 2D points.

    Returns:
        tuple:
            - normalized_points (numpy.ndarray): N x 2 array of normalized points.
            - T (numpy.ndarray): 3x3 transformation matrix used for normalization.
    """
    if points.shape[0] < 1:
        return points, np.eye(3)

    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Translate points
    translated_points = points - centroid

    # Calculate average distance from origin
    avg_dist = np.mean(np.sqrt(np.sum(translated_points**2, axis=1)))

    # Calculate scaling factor
    scale = np.sqrt(2) / avg_dist if avg_dist > 1e-9 else 1.0 # Avoid division by zero

    # Create normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    # Apply transformation to points (convert to homogeneous, then back to 2D)
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_homogeneous_points = (T @ homogeneous_points.T).T
    normalized_points = normalized_homogeneous_points[:, :2]

    return normalized_points, T

def build_A(x1,x2):
    # x1 [n*2] x2 [n*2] same size
    x1, y1 = x1[:,0], x1[:,1] # N,
    x2, y2 = x2[:,0], x2[:,1] # N,
    ones = np.ones(x1.shape[0])

    A = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, ones] # N x 9
    A = np.asarray(A).T # N x 9
    return A

def fundamental_matrix(feature1, feature2):
    # TODO: Need to normalize the features before calculating.
    feat1_norn, t1 = normalize_points(feature1)
    feat2_norn, t2 = normalize_points(feature2)
    
    A = build_A(feat1_norn, feat2_norn)

    u, s, v = np.linalg.svd(A)
    # choose the vector with the smaller eigen values
    F_flat = v[-1,:]
    F_estimate = F_flat.reshape(3,3)

    # 4. Enforce Rank 2 constraint on F
    U_F, S_F, V_F = np.linalg.svd(F_estimate)

    # Set the smallest singular value to 0
    S_F[2] = 0 # Enforcing rank 2
    
    # Reconstruct the singular value matrix
    Sigma_F_diag = np.diag(S_F)
    
    # Reconstruct F from the modified SVD
    F_rank2 = U_F @ Sigma_F_diag @ V_F

    # 5. Denormalize F
    # F = T2_transpose @ F_normalized @ T1
    F_denormalized = t2.T @ F_rank2 @ t1

    return F_denormalized