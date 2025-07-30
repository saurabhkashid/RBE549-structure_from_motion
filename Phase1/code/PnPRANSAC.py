import numpy as np
from util import reprojection_error_pnp

def build_A_og(x_pt,X_pt):
    X, Y, Z = X_pt
    x, y = x_pt
    A = np.array([[X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x],
                 [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]])
    return A

def build_A(x_pt, X_pt):
    """
    Builds the 2x12 sub-matrix A_pt for a single 2D-3D correspondence (x_pt, X_pt)
    for DLT PnP.

    Args:
        x_pt (numpy.ndarray): 2-element array (u, v) of the 2D image point.
        X_pt (numpy.ndarray): 3-element array (X, Y, Z) of the 3D world point.

    Returns:
        numpy.ndarray: 2x12 matrix A_pt for this correspondence.
    """
    X_world, Y_world, Z_world = X_pt # Renamed for clarity with 3D world coords
    u_image, v_image = x_pt        # Renamed for clarity with 2D image coords

    # Homogeneous 3D world point [X, Y, Z, 1]
    X_homog_row = np.array([X_world, Y_world, Z_world, 1])

    A_pt = np.zeros((2, 12))

    # First row of A_pt: (u * P3 - P1) * X_homog = 0
    # Coefficients for P1 (p_11 to p_14)
    A_pt[0, 0:4] = -X_homog_row
    # Coefficients for P2 (p_21 to p_24) - all zeros in this row
    # A_pt[0, 4:8] = np.zeros(4)
    # Coefficients for P3 (p_31 to p_34)
    A_pt[0, 8:12] = u_image * X_homog_row

    # Second row of A_pt: (v * P3 - P2) * X_homog = 0
    # Coefficients for P1 (p_11 to p_14) - all zeros in this row
    # A_pt[1, 0:4] = np.zeros(4)
    # Coefficients for P2 (p_21 to p_24)
    A_pt[1, 4:8] = -X_homog_row
    # Coefficients for P3 (p_31 to p_34)
    A_pt[1, 8:12] = v_image * X_homog_row
    
    return A_pt

def decompose_P_to_K_R_t(P, K):
    """
    Decomposes a 3x4 camera projection matrix P into camera intrinsic matrix K,
    rotation matrix R, and translation vector t.
    This function handles the SVD-based cleanup for R and normalizes t.

    Args:
        P (numpy.ndarray): The 3x4 camera projection matrix obtained from DLT PnP (or similar linear method).
        K (numpy.ndarray): The 3x3 camera intrinsic matrix (must be invertible).

    Returns:
        tuple:
            - R (numpy.ndarray): 3x3 Rotation matrix from world to camera.
            - t (numpy.ndarray): 3x1 Translation vector from world to camera.
            - success (bool): True if decomposition was successful (K is invertible).
    """
    # 1. Check for K invertibility
    if np.linalg.det(K) == 0:
        print("Error: Camera intrinsic matrix K is singular, cannot decompose P.")
        return None, None, False

    # 2. Extract M and T_col from P
    # P is structured as P = K @ [R | t]
    # So, P = [ K@R | K@t ]
    # We can define M = K@R (the left 3x3 part of P)
    # And T_col = K@t (the rightmost 3x1 column of P)
    M = P[:, :3]    # Left 3x3 part of P
    T_col = P[:, 3] # Rightmost 3x1 column of P

    # 3. Compute the candidate Rotation Matrix: R_candidate = K_inv @ M
    K_inv = np.linalg.inv(K)
    R_candidate = K_inv @ M

    # 4. SVD Cleanup for R_candidate
    # A rotation matrix must be orthonormal (R @ R.T = I) and have det(R) = 1.
    # Due to noise and the linear nature of DLT, R_candidate might not perfectly satisfy these.
    # We use SVD to find the closest proper rotation matrix.
    # If R_candidate = U @ S @ Vh, then the closest rotation matrix is R = U @ Vh.
    U_R, S_R, Vh_R = np.linalg.svd(R_candidate)
    
    # The largest singular value (S_R[0]) represents a scale factor (often denoted gamma or d1)
    # introduced by the linear estimation of P. This scale applies to both R and t.
    scale_factor = S_R[0]
    
    # Reconstruct R, making it orthonormal.
    R = U_R @ Vh_R

    # 5. Ensure det(R) = 1 (Right-handed coordinate system)
    # If det(R) is -1, it means the SVD resulted in a reflection, not a pure rotation.
    # We correct this by flipping the sign of R.
    if np.linalg.det(R) < 0:
        R = -R # This corrects the reflection.

    # 6. Compute Translation Vector t
    # From T_col = K@t, we get t = K_inv @ T_col.
    # This translation also needs to be normalized by the same scale_factor
    # to be consistent with the scale of the rotation and the 3D world.
    t = (K_inv @ T_col) / scale_factor

    return R, t.reshape(3, 1), True # Reshape t to be a 3x1 column vector

def linear_PnP(x, X, K):
    """
    Estimates the 3x4 camera projection matrix (P) using Direct Linear Transformation (DLT)
    given 2D-3D correspondences. Then decomposes P into R and t using the camera intrinsics K.

    Args:
        x (numpy.ndarray): N x 2 array of 2D image points (u, v).
        X (numpy.ndarray): N x 3 array of 3D world points (X, Y, Z).
        K (numpy.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        tuple:
            - R (numpy.ndarray): 3x3 Rotation matrix from world to camera.
            - t (numpy.ndarray): 3x1 Translation vector from world to camera.
    """

    # create empty A 
    A = np.zeros((2*len(X),12))

    for i in range(len(x)):
        A_pt = build_A(x[i], X[i])
        # Place A_pt into the larger A matrix
        # Each A_pt (2x12) occupies two rows in the global A matrix
        A[i * 2 : (i * 2) + 2, :] = A_pt
    U, S_A, Vh_A = np.linalg.svd(A)
    P_vec = Vh_A[-1, :] # This is the vectorized P
    P = P_vec.reshape(3, 4) # This is the 3x4 projection matrix P

    R, t, decomp_result = decompose_P_to_K_R_t(P, K)

    if decomp_result:
        return R, t, P
    
    else:
        raise ValueError("can't decompose the provided P")
    

# --- RANSAC PnP Function ---
def ransac_pnp(x_2d, X_3d, K, iterations=1000, threshold=5.0):
    """
    Estimates the camera pose (R, t) using RANSAC with the linear_PnP algorithm
    to robustly handle outliers.

    Args:
        x_2d (numpy.ndarray): N x 2 array of 2D image points (u, v).
        X_3d (numpy.ndarray): N x 3 array of 3D world points (X, Y, Z).
        K (numpy.ndarray): 3x3 camera intrinsic matrix.
        iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.
        threshold (float, optional): Maximum allowed reprojection error (in pixels)
                                     for a point to be considered an inlier. Defaults to 5.0.
        confidence (float, optional): Desired probability of success. Not directly used for
                                      early exit in this basic implementation, but good to keep.

    Returns:
        tuple:
            - R_best (numpy.ndarray): 3x3 best estimated rotation matrix.
            - t_best (numpy.ndarray): 3x1 best estimated translation vector.
            - inliers_mask (numpy.ndarray): Boolean mask of inlier points.
    """
    num_points = len(x_2d)
    # NOTE: Reason of taking the 6 points is probability of having those 6 point degenrate is hard 
    if num_points < 6:
        print("Warning: Not enough points for RANSAC PnP (minimum 6 required).")
        return np.eye(3), np.zeros((3,1)), np.zeros(num_points, dtype=bool)

    # Minimal set size for DLT PnP
    MIN_POINTS = 6

    best_inliers_count = 0
    best_R = np.eye(3)
    best_t = np.zeros((3,1))
    best_inliers_mask = np.zeros(num_points, dtype=bool)

    print(f"Running RANSAC PnP for {num_points} points with {iterations} iterations and threshold {threshold} pixels...")

    for i in range(iterations):
        # 1. Randomly select a minimal set of points
        sample_indices = np.random.choice(num_points, MIN_POINTS, replace=False)
        x_sample = x_2d[sample_indices]
        X_sample = X_3d[sample_indices]

        try:
            # 2. Estimate pose using linear_PnP on the sample
            R_sample, t_sample, P_sample = linear_PnP(x_sample, X_sample, K)
        except ValueError:
            # linear_PnP might fail if the sample points are degenerate (e.g., collinear)
            continue

        # 3. Calculate reprojection error for ALL points using the estimated pose
        reprojection_errors = reprojection_error_pnp(x_2d, X_3d, P_sample)

        # 4. Count inliers
        current_inliers_mask = reprojection_errors < threshold
        current_inliers_count = np.sum(current_inliers_mask)

        # 5. Update best model if current model is better
        if current_inliers_count > best_inliers_count:
            best_inliers_count = current_inliers_count
            best_R = R_sample
            best_t = t_sample
            best_inliers_mask = current_inliers_mask

    print(f"RANSAC PnP finished. Best model found with {best_inliers_count} inliers ({best_inliers_count/num_points*100:.2f}%).")
    
    # Refine the best pose using ALL inliers (optional but recommended for final accuracy)
    if best_inliers_count >= MIN_POINTS:
        x_inliers = x_2d[best_inliers_mask]
        X_inliers = X_3d[best_inliers_mask]
        try:
            # Use linear_PnP again on all inliers
            R_final, t_final, _ = linear_PnP(x_inliers, X_inliers, K)
            best_R, best_t = R_final, t_final
            print("  Refined pose using all inliers.")
        except ValueError:
            print("  Warning: Final linear_PnP on all inliers failed. Using best sample pose.")
    else:
        print("  Warning: No valid pose found with enough inliers. Returning initial identity pose.")
        best_R = np.eye(3)
        best_t = np.zeros((3,1))
        best_inliers_mask = np.zeros(num_points, dtype=bool) # No inliers if no valid pose

    return best_R, best_t, best_inliers_mask


if __name__ == "__main__":
    from util import project_3d_to_2d
    # --- Setup Test Data ---
    # --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing linear_PnP function ---")

    # 1. Define Camera Intrinsics (K) 📸
    K_test = np.array([
        [800.0, 0.0, 320.0],  # fx, 0, cx
        [0.0, 800.0, 240.0],  # 0, fy, cy
        [0.0, 0.0, 1.0]       # 0, 0, 1
    ], dtype=np.float64)

    # 2. Define Ground Truth Camera Pose (R_true, t_true) 🌎➡️📷
    # Simple rotation: 10 degrees around Y-axis
    theta_rad_true = np.deg2rad(10)
    R_true = np.array([
        [np.cos(theta_rad_true), 0, np.sin(theta_rad_true)],
        [0, 1, 0],
        [-np.sin(theta_rad_true), 0, np.cos(theta_rad_true)]
    ], dtype=np.float64)
    t_true = np.array([0.5, 0.1, 1.5], dtype=np.float64).reshape(3, 1) # Translation (e.g., in meters)

    # Ground truth Projection Matrix
    P_true = K_test @ np.hstack((R_true, t_true))
    print("\n--- Ground Truth ---")
    print(f"Ground Truth R:\n{R_true}")
    print(f"Ground Truth t:\n{t_true.flatten()}")
    print(f"Ground Truth P:\n{P_true}")

    # 3. Generate 3D World Points (X) 🌟
    num_points = 10 # DLT PnP requires at least 6 points
    np.random.seed(42) # For reproducibility
    # Generate points within a reasonable spatial range
    world_points_3d = np.random.rand(num_points, 3) * 10
    world_points_3d[:, 2] += 5 # Ensure points are generally in front of the camera (positive Z)

    print(f"\nNumber of 3D points generated: {num_points}")

    # 4. Project 3D Points to 2D Image Points (x) using Ground Truth Pose
    image_points_2d_ideal = np.array([project_3d_to_2d(P_true, p)[0] for p in world_points_3d])

    # 5. Add Noise to 2D Points (Simulate sensor noise/measurement error) 📏
    noise_std_dev = 0.8 # Standard deviation of noise in pixels
    image_points_2d_noisy = image_points_2d_ideal + np.random.randn(*image_points_2d_ideal.shape) * noise_std_dev
    
    print(f"Simulated noise (std dev): {noise_std_dev} pixels")

    # 6. Run linear_PnP
    try:
        estimated_R, estimated_t, estimated_P = linear_PnP(image_points_2d_noisy, world_points_3d, K_test)

        print("\n--- Linear PnP Results ---")
        print(f"Estimated R:\n{estimated_R}")
        print(f"Estimated t:\n{estimated_t.flatten()}")
        print(f"Estimated P:\n{estimated_P}")

        # 7. Evaluate Results 📊
        print("\n--- Evaluation ---")
        # Compare R
        R_diff_norm = np.linalg.norm(estimated_R - R_true, 'fro') # Frobenius norm
        print(f"Difference in R (Frobenius norm): {R_diff_norm:.6f}")

        # Compare t (Note: DLT PnP recovers t up to scale, but our decomposition normalizes it.
        # Still, expect some error due to noise.)
        t_diff_norm = np.linalg.norm(estimated_t - t_true) # Euclidean norm
        print(f"Difference in t (Euclidean norm): {t_diff_norm:.6f}")

        # Calculate Reprojection Error using the estimated P
        total_reproj_error_sq = 0.0
        for i in range(num_points):
            reprojected_pt = project_3d_to_2d(estimated_P, world_points_3d[i])[0]
            error_vec = image_points_2d_noisy[i] - reprojected_pt
            total_reproj_error_sq += np.sum(error_vec**2)
        
        rmse_reproj_error = np.sqrt(total_reproj_error_sq / num_points)
        print(f"Root Mean Square Reprojection Error: {rmse_reproj_error:.4f} pixels")

        # A successful test would show small differences in R and t,
        # and a reprojection error that's commensurate with the noise level.
        if R_diff_norm < 0.1 and t_diff_norm < 0.2 and rmse_reproj_error < (noise_std_dev * 2): # Example thresholds
            print("\n✅ Test Passed: Linear PnP seems to work correctly!")
        else:
            print("\n❌ Test Failed: Differences in R/t or reprojection error are too high.")

    except ValueError as e:
        print(f"\nError running linear_PnP: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")