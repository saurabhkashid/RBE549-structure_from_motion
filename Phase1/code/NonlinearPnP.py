import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from util import reprojection_error, project_3d_to_2d

# Assume project_3d_to_2d and reprojection_error are already defined as before
# (They don't change as they operate on P matrix)

def NonlinearPnP(initial_R, initial_t, x, X, K):
    """
    Refines a camera's pose (R, t) using nonlinear optimization (least_squares)
    with rotation represented by a quaternion, utilizing scipy.spatial.transform.Rotation.

    Args:
        initial_R (numpy.ndarray): 3x3 initial rotation matrix.
        initial_t (numpy.ndarray): 3x1 initial translation vector.
        x (numpy.ndarray): N x 2 array of observed 2D image points.
        X (numpy.ndarray): N x 3 array of corresponding 3D world points.
        K (numpy.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        tuple:
            - R_refined (numpy.ndarray): 3x3 refined rotation matrix.
            - t_refined (numpy.ndarray): 3x1 refined translation vector.
            - success (bool): True if optimization converged successfully.
    """
    # 1. Parameterization of Pose
    # Convert initial R to quaternion [x, y, z, w] using Scipy's Rotation object.
    # Scipy's default quaternion order is [x, y, z, w].
    initial_quat = R_scipy.from_matrix(initial_R).as_quat() # Returns [x, y, z, w]
    t_initial_flat = initial_t.flatten()

    # Combine into a single 7-element parameter vector
    pose_params_initial = np.concatenate((initial_quat, t_initial_flat))

    # 2. Define the Objective Function for Least Squares
    def objective_function_scipy_quat(pose_params, K_fixed, X_fixed, x_observed):
        """
        Calculates the reprojection errors for all 2D-3D correspondences.
        Rotation is parameterized by a quaternion, handled by Scipy.Rotation.
        """
        # Extract quaternion and translation from the pose_params vector
        current_quat = pose_params[:4]
        current_t = pose_params[4:].reshape(3, 1)

        # Create a Scipy Rotation object from the quaternion.
        # Scipy's from_quat automatically normalizes if not already normalized.
        current_R_scipy = R_scipy.from_quat(current_quat)
        
        # Convert the Scipy Rotation object back to a 3x3 rotation matrix.
        current_R = current_R_scipy.as_matrix()

        # Construct the current projection matrix P_current = K @ [R | t]
        P_current = K_fixed @ np.hstack((current_R, current_t))

        all_residuals = []
        for i in range(len(X_fixed)):
            residuals = reprojection_error(x_observed[i], X_fixed[i], P_current)
            all_residuals.extend(residuals)

        return np.array(all_residuals)

    # 3. Perform the Optimization
    result = least_squares(
        objective_function_scipy_quat,
        pose_params_initial,
        args=(K, X, x),
        verbose=0,
        method='lm',
        ftol=1e-6,
        xtol=1e-6
    )

    # 4. Extract Refined Pose
    if result.success:
        refined_pose_params = result.x
        refined_quat = refined_pose_params[:4]
        t_refined = refined_pose_params[4:].reshape(3, 1)
        
        # Create Rotation object from refined quaternion and get the matrix
        # This will automatically normalize the quaternion.
        R_refined = R_scipy.from_quat(refined_quat).as_matrix()
        
        return R_refined, t_refined, True
    else:
        print(f"NonlinearPnP (Quaternion with Scipy) optimization failed: {result.message}")
        return initial_R, initial_t, False
    
if __name__ == "__main__":
    from PnPRANSAC import linear_PnP
    from util import project_3d_to_2d


    # --- Setup Test Data ---
    K_test = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # True Camera Pose
    theta_rad_true = np.deg2rad(15)
    R_true = np.array([
        [np.cos(theta_rad_true), 0, np.sin(theta_rad_true)],
        [0, 1, 0],
        [-np.sin(theta_rad_true), 0, np.cos(theta_rad_true)]
    ], dtype=np.float64)
    t_true = np.array([0.7, -0.3, 2.0], dtype=np.float64).reshape(3, 1)
    P_true = K_test @ np.hstack((R_true, t_true))

    # Generate 3D World Points
    num_points = 20
    np.random.seed(42)
    world_points_3d = np.random.rand(num_points, 3) * 10
    world_points_3d[:, 2] += 5 # Ensure positive depth

    # Project 3D points to 2D using true pose and add noise
    image_points_2d = np.array([project_3d_to_2d(P_true, p)[0] for p in world_points_3d])
    noise_std_dev = 1.5 # Increased noise for demonstration of refinement
    image_points_2d_noisy = image_points_2d + np.random.randn(*image_points_2d.shape) * noise_std_dev

    print("--- Nonlinear PnP (Quaternion) Refinement Test ---")
    print(f"Ground Truth R:\n{R_true}")
    print(f"Ground Truth t:\n{t_true.flatten()}")
    print(f"\nNumber of points: {num_points}")
    print(f"Noise std dev: {noise_std_dev} pixels")

    # --- Step 1: Get initial R, t from Linear PnP ---
    try:
        initial_R_pnp, initial_t_pnp, _ = linear_PnP(image_points_2d_noisy, world_points_3d, K_test)
        print("\n--- Initial Pose from Linear PnP ---")
        print(f"Initial R:\n{initial_R_pnp}")
        print(f"Initial t:\n{initial_t_pnp.flatten()}")
        
        R_diff_linear = np.linalg.norm(initial_R_pnp - R_true)
        t_diff_linear = np.linalg.norm(initial_t_pnp - t_true)
        print(f"Linear PnP R diff from GT: {R_diff_linear:.6f}")
        print(f"Linear PnP t diff from GT: {t_diff_linear:.6f}")

        # Calculate initial reprojection error from Linear PnP
        initial_reproj_error_sum_sq = 0
        P_initial = K_test @ np.hstack((initial_R_pnp, initial_t_pnp))
        for i in range(num_points):
            reproj_pt = project_3d_to_2d(P_initial, world_points_3d[i])[0]
            initial_reproj_error_sum_sq += np.sum((reproj_pt - image_points_2d_noisy[i])**2)
        print(f"Initial Reprojection Error (sum_sq): {initial_reproj_error_sum_sq:.6f}")

        # --- Step 2: Refine with Nonlinear PnP using Quaternions ---
        print("\n--- Starting Nonlinear PnP (Quaternion) Refinement ---")
        R_refined_q, t_refined_q, success_q = NonlinearPnP(initial_R_pnp, initial_t_pnp, image_points_2d_noisy, world_points_3d, K_test)

        if success_q:
            print("\n--- Refined Pose from Nonlinear PnP (Quaternion) ---")
            print(f"Refined R:\n{R_refined_q}")
            print(f"Refined t:\n{t_refined_q.flatten()}")

            R_diff_nonlinear_q = np.linalg.norm(R_refined_q - R_true)
            t_diff_nonlinear_q = np.linalg.norm(t_refined_q - t_true)
            print(f"Nonlinear PnP R diff from GT: {R_diff_nonlinear_q:.6f}")
            print(f"Nonlinear PnP t diff from GT: {t_diff_nonlinear_q:.6f}")

            # Calculate final reprojection error from Nonlinear PnP
            final_reproj_error_sum_sq_q = 0
            P_refined_q = K_test @ np.hstack((R_refined_q, t_refined_q))
            for i in range(num_points):
                reproj_pt = project_3d_to_2d(P_refined_q, world_points_3d[i])[0]
                final_reproj_error_sum_sq_q += np.sum((reproj_pt - image_points_2d_noisy[i])**2)
            print(f"Final Reprojection Error (sum_sq): {final_reproj_error_sum_sq_q:.6f}")

            print(f"\nReprojection Error Reduced (Quaternion): {initial_reproj_error_sum_sq > final_reproj_error_sum_sq_q}")
            print(f"Linear R error: {R_diff_linear:.6f}, Nonlinear R (Quaternion) error: {R_diff_nonlinear_q:.6f}")
            print(f"Linear t error: {t_diff_linear:.6f}, Nonlinear t (Quaternion) error: {t_diff_nonlinear_q:.6f}")
        else:
            print("Nonlinear PnP (Quaternion) did not converge successfully.")

    except ValueError as e:
        print(f"Error during PnP process: {e}")