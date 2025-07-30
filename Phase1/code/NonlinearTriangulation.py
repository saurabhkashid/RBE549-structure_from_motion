import numpy as np
from scipy.optimize import least_squares
from util import project_3d_to_2d, reprojection_error


def non_linear_triangulation(K1, t_cam1, R_cam1, K2, t_cam2, R_cam2, pts1, pts2, World_pts_init):
    # Construct projection matrices for both cameras
    P1 = K1 @ np.hstack((R_cam1, t_cam1.reshape(3, 1)))
    P2 = K2 @ np.hstack((R_cam2, t_cam2.reshape(3, 1)))

    num_points = World_pts_init.shape[0]
    refined_world_points = np.zeros_like(World_pts_init, dtype=np.float64)

    for i in range(num_points):
        # The specific 2D observations for the current point
        current_obs_pts1 = pts1[i, :]
        current_obs_pts2 = pts2[i, :]
        
        # The initial 3D estimate for the current point
        current_X_init = World_pts_init[i, :]

        # Define the objective function for least_squares for a single point
        # It takes the 3D point X as the first argument, followed by other fixed parameters
        def objective_function(X, P1_arg, P2_arg, obs_pts1_arg, obs_pts2_arg):
            error1 = reprojection_error(obs_pts1_arg, X, P1_arg)
            error2 = reprojection_error(obs_pts2_arg, X, P2_arg)
            return np.concatenate((error1, error2)) # Returns a 1D array of residuals

        # Perform the optimization for the current point
        result = least_squares(
            objective_function,
            current_X_init,
            args=(P1, P2, current_obs_pts1, current_obs_pts2),
            verbose=0, # Set to 1 for detailed optimization progress per point
            method='lm' # Levenberg-Marquardt is robust for this type of problem
        )
        
        refined_world_points[i, :] = result.x

    return refined_world_points

# --- Example Usage (requires the previously defined helper functions) ---
if __name__ == "__main__":
    # Define camera parameters (from previous examples)
    K1 = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    K2 = K1 # Assuming same camera intrinsics for simplicity

    # Best R and t from essential matrix decomposition (example values)
    theta = np.deg2rad(5) # Rotate 5 degrees around Y-axis
    R_best = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t_best = np.array([0.1, 0.05, 0.02])

    # Assume first camera is at identity pose
    R1_pose = np.eye(3)
    t1_pose = np.zeros(3)

    # Second camera's pose relative to the first
    R2_pose = R_best
    t2_pose = t_best

    # --- Generate Multiple Ground Truth 3D Points and their Projections ---
    num_test_points = 5 # Number of points to test
    np.random.seed(42) # For reproducibility
    X_ground_truth_batch = np.random.rand(num_test_points, 3) * 10 - 5 # Points between -5 and 5
    X_ground_truth_batch[:, 2] = np.abs(X_ground_truth_batch[:, 2]) + 5 # Ensure positive depth

    # Project ground truth 3D points to 2D image coordinates (with noise)
    P1_gt = K1 @ np.hstack((R1_pose, t1_pose.reshape(3, 1)))
    P2_gt = K2 @ np.hstack((R2_pose, t2_pose.reshape(3, 1)))

    # Generate noisy 2D observations for all points
    observed_pts1 = np.zeros((num_test_points, 2))
    observed_pts2 = np.zeros((num_test_points, 2))
    noise_std_dev = 0.5 # Pixels

    for i in range(num_test_points):
        obs_x1_ideal, _ = project_3d_to_2d(P1_gt, X_ground_truth_batch[i, :])
        obs_x2_ideal, _ = project_3d_to_2d(P2_gt, X_ground_truth_batch[i, :])
        
        observed_pts1[i, :] = obs_x1_ideal + np.random.randn(2) * noise_std_dev
        observed_pts2[i, :] = obs_x2_ideal + np.random.randn(2) * noise_std_dev

    # --- Initial 3D point estimates (e.g., from linear triangulation) ---
    World_pts_init_batch = X_ground_truth_batch + np.random.randn(num_test_points, 3) * 0.5 # Add some deviation

    print("--- Batch Nonlinear Triangulation Test ---")
    print(f"Number of points: {num_test_points}")
    print("\nInitial 3D Point Estimates (first 2):\n", World_pts_init_batch[:2])
    
    # Calculate initial average reprojection error
    initial_errors_sum_sq = 0
    for i in range(num_test_points):
        reproj_x1_init, _ = project_3d_to_2d(P1_gt, World_pts_init_batch[i, :])
        reproj_x2_init, _ = project_3d_to_2d(P2_gt, World_pts_init_batch[i, :])
        initial_errors_sum_sq += np.sum((observed_pts1[i, :] - reproj_x1_init)**2)
        initial_errors_sum_sq += np.sum((observed_pts2[i, :] - reproj_x2_init)**2)
    print(f"\nInitial Total Reprojection Error (sum_sq): {initial_errors_sum_sq:.6f}")

    # --- Perform Nonlinear Triangulation ---
    refined_world_points = non_linear_triangulation(
        K1, t1_pose, R1_pose, K2, t2_pose, R2_pose,
        observed_pts1, observed_pts2, World_pts_init_batch
    )

    print("\nRefined 3D World Points (first 2):\n", refined_world_points[:2])

    # --- Compare Results ---
    final_errors_sum_sq = 0
    for i in range(num_test_points):
        reproj_x1_final, _ = project_3d_to_2d(P1_gt, refined_world_points[i, :])
        reproj_x2_final, _ = project_3d_to_2d(P2_gt, refined_world_points[i, :])
        final_errors_sum_sq += np.sum((observed_pts1[i, :] - reproj_x1_final)**2)
        final_errors_sum_sq += np.sum((observed_pts2[i, :] - reproj_x2_final)**2)
    print(f"Final Total Reprojection Error (sum_sq): {final_errors_sum_sq:.6f}")

    avg_error_to_gt = np.mean(np.linalg.norm(refined_world_points - X_ground_truth_batch, axis=1))
    print(f"\nAverage Distance from Refined X to Ground Truth X: {avg_error_to_gt:.6f}")
    
    print("\n--- Individual Point Comparison (First Point) ---")
    print(f"Ground Truth X (1st pt): {X_ground_truth_batch[0]}")
    print(f"Initial X (1st pt): {World_pts_init_batch[0]}")
    print(f"Refined X (1st pt): {refined_world_points[0]}")
    print(f"Error (init vs GT): {np.linalg.norm(World_pts_init_batch[0] - X_ground_truth_batch[0]):.6f}")
    print(f"Error (refined vs GT): {np.linalg.norm(refined_world_points[0] - X_ground_truth_batch[0]):.6f}")

