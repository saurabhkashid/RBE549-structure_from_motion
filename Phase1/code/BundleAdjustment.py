import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy
from util import reprojection_error


def bundle_adjustment(initial_camera_poses, initial_world_points, observations, K_matrices):
    """
    Performs Bundle Adjustment to jointly refine camera poses and 3D world points.

    Args:
        initial_camera_poses (list): List of tuples (R, t) for each camera.
                                     R is 3x3, t is 3x1.
        initial_world_points (numpy.ndarray): N x 3 array of initial 3D world points.
        observations (list): List of tuples (camera_idx, point_idx, observed_u, observed_v).
        K_matrices (list or numpy.ndarray): List of 3x3 intrinsic matrices for each camera,
                                            or a single 3x3 matrix if all cameras are identical.

    Returns:
        tuple:
            - refined_camera_poses (list): List of (R, t) for refined camera poses.
            - refined_world_points (numpy.ndarray): N x 3 array of refined 3D world points.
            - success (bool): True if optimization converged successfully.
    """
    num_cameras = len(initial_camera_poses)
    num_points = initial_world_points.shape[0]

    # Ensure K_matrices is a list for consistent indexing
    if not isinstance(K_matrices, list):
        K_matrices_list = [K_matrices] * num_cameras
    else:
        K_matrices_list = K_matrices

    # 1. Parameterization of all variables into a single 1D vector
    # Order: [rvec1, t1, rvec2, t2, ..., rvecM, tM, X1, Y1, Z1, X2, Y2, Z2, ..., XN, YN, ZN]
    
    # Camera parameters (6 per camera: 3 for rvec, 3 for t)
    camera_params_flat = []
    for R_cam, t_cam in initial_camera_poses:
        rvec, _ = cv2.Rodrigues(R_cam) # Convert R to Rodrigues vector
        camera_params_flat.extend(rvec.flatten())
        camera_params_flat.extend(t_cam.flatten())
    camera_params_flat = np.array(camera_params_flat)

    # World point parameters (3 per point: X, Y, Z)
    world_points_flat = initial_world_points.flatten()

    # Combine all parameters
    all_params_initial = np.concatenate((camera_params_flat, world_points_flat))

    # --- Indices for unpacking parameters in the objective function ---
    # These help efficiently extract camera and point parameters from the flat vector
    camera_param_start_idx = 0
    camera_param_end_idx = num_cameras * 6 # 6 params per camera
    point_param_start_idx = camera_param_end_idx
    point_param_end_idx = point_param_start_idx + num_points * 3 # 3 params per point

    # 2. Define the Objective Function for Least Squares
    def objective_function_ba(all_params, num_cameras, num_points, observations_fixed, K_matrices_fixed):
        """
        Calculates the reprojection errors for all observations across all cameras and points.
        This function is minimized by least_squares.
        """
        # Unpack parameters
        current_camera_params_flat = all_params[camera_param_start_idx : camera_param_end_idx]
        current_world_points_flat = all_params[point_param_start_idx : point_param_end_idx]

        # Reshape world points
        current_world_points = current_world_points_flat.reshape(num_points, 3)

        all_residuals = []
        for cam_idx, point_idx, observed_u, observed_v in observations_fixed:
            # Get current camera pose for this observation
            cam_offset = cam_idx * 6
            rvec = current_camera_params_flat[cam_offset : cam_offset + 3].reshape(3, 1)
            t = current_camera_params_flat[cam_offset + 3 : cam_offset + 6].reshape(3, 1)
            
            # Convert rvec to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Get intrinsic matrix for this camera
            K = K_matrices_fixed[cam_idx]

            # Construct projection matrix
            P = K @ np.hstack((R, t))

            # Get current 3D point for this observation
            X_3d = current_world_points[point_idx]

            # Calculate reprojection error
            residuals = reprojection_error(np.array([observed_u, observed_v]), X_3d, P)
            all_residuals.extend(residuals)

        return np.array(all_residuals)

    # 3. Perform the Optimization
    print(f"Starting Bundle Adjustment with {len(all_params_initial)} parameters and {len(observations) * 2} residuals.")
    result = least_squares(
        objective_function_ba,
        all_params_initial,
        args=(num_cameras, num_points, observations, K_matrices_list),
        verbose=1, # Set to 2 for more detailed progress, 0 for none
        method='lm', # Levenberg-Marquardt is standard for BA
        ftol=1e-6,
        xtol=1e-6,
        # gtol=1e-6, # Gradient tolerance
        # max_nfev=2000 # Max function evaluations
    )

    # 4. Extract Refined Parameters
    refined_camera_poses = []
    refined_world_points = np.zeros_like(initial_world_points)

    if result.success or result.status in [1, 2]: # status 1: x_tol, 2: ftol
        refined_params = result.x

        # Extract refined camera poses
        for cam_idx in range(num_cameras):
            cam_offset = cam_idx * 6
            rvec_refined = refined_params[cam_offset : cam_offset + 3].reshape(3, 1)
            t_refined = refined_params[cam_offset + 3 : cam_offset + 6].reshape(3, 1)
            R_refined, _ = cv2.Rodrigues(rvec_refined)
            refined_camera_poses.append((R_refined, t_refined))

        # Extract refined world points
        refined_world_points_flat = refined_params[point_param_start_idx : point_param_end_idx]
        refined_world_points = refined_world_points_flat.reshape(num_points, 3)
        
        print(f"\nBundle Adjustment converged with status: {result.message}")
        print(f"Final cost (sum of squared residuals): {result.cost:.6f}")
        return refined_camera_poses, refined_world_points, True
    else:
        print(f"\nBundle Adjustment failed to converge: {result.message}")
        print(f"Final cost (sum of squared residuals): {result.cost:.6f}")
        return initial_camera_poses, initial_world_points, False # Return initial on failure
    
if __name__ == "__main__":
    import cv2 # Ensure OpenCV is imported for Rodrigues
    from BuildVisibilityMatrix import create_observations_list

    # --- Setup: Camera Intrinsics ---
    K_common = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    K_matrices_list = [K_common, K_common, K_common] # Example: 3 cameras, all same intrinsics

    # --- Generate Ground Truth Camera Poses ---
    # Camera 1: Identity (reference frame)
    R1_true = np.eye(3)
    t1_true = np.zeros((3, 1))

    # Camera 2: Slight rotation and translation
    R2_true = R_scipy.from_rotvec([0.1, 0.2, 0.05]).as_matrix()
    t2_true = np.array([0.5, 0.1, 0.2]).reshape(3, 1)

    # Camera 3: More rotation and translation
    R3_true = R_scipy.from_rotvec([-0.1, 0.3, -0.15]).as_matrix()
    t3_true = np.array([1.0, -0.2, 0.8]).reshape(3, 1)

    true_camera_poses = [(R1_true, t1_true), (R2_true, t2_true), (R3_true, t3_true)]
    num_cameras = len(true_camera_poses)

    # --- Generate Ground Truth 3D World Points ---
    num_points = 300 # Number of 3D points
    np.random.seed(0)
    true_world_points = np.random.rand(num_points, 3) * 5 - 2.5 # Points between -2.5 and 2.5
    true_world_points[:, 2] = np.abs(true_world_points[:, 2]) + 5 # Ensure positive depth

    print("--- Bundle Adjustment Test ---")
    print(f"Number of cameras: {num_cameras}")
    print(f"Number of 3D points: {num_points}")

    # --- Simulate Observations (Visibility Graph) ---
    observation_noise_std_dev = 1.5 # Pixels
    observations = create_observations_list(
        true_camera_poses, true_world_points, K_matrices_list,
        noise_std_dev=observation_noise_std_dev
    )
    print(f"Total simulated observations (2D-3D correspondences): {len(observations)}")
    print(f"Observation noise std dev: {observation_noise_std_dev} pixels")

    # --- Create Initial Guesses (add some noise to ground truth) ---
    # This simulates the output of PnP and linear triangulation
    initial_camera_poses = []
    pose_noise_std_dev_r = 0.05 # For rotation vector components
    pose_noise_std_dev_t = 0.1 # For translation components (meters)
    
    for R_true_cam, t_true_cam in true_camera_poses:
        rvec_true, _ = cv2.Rodrigues(R_true_cam)
        rvec_noisy = rvec_true + np.random.randn(3, 1) * pose_noise_std_dev_r
        t_noisy = t_true_cam + np.random.randn(3, 1) * pose_noise_std_dev_t
        R_noisy, _ = cv2.Rodrigues(rvec_noisy)
        initial_camera_poses.append((R_noisy, t_noisy))

    initial_world_points = true_world_points + np.random.randn(num_points, 3) * 0.1 # 10cm noise

    print("\n--- Initial State ---")
    # Calculate initial total reprojection error
    initial_total_reproj_error_sq = 0.0
    for cam_idx, point_idx, obs_u, obs_v in observations:
        R_cam, t_cam = initial_camera_poses[cam_idx]
        K_cam = K_matrices_list[cam_idx]
        P_cam = K_cam @ np.hstack((R_cam, t_cam.reshape(3, 1)))
        
        X_3d = initial_world_points[point_idx]
        
        residuals = reprojection_error(np.array([obs_u, obs_v]), X_3d, P_cam)
        initial_total_reproj_error_sq += np.sum(residuals**2)
    initial_rmse = np.sqrt(initial_total_reproj_error_sq / len(observations))
    print(f"Initial RMS Reprojection Error: {initial_rmse:.4f} pixels")

    # --- Perform Bundle Adjustment ---
    refined_camera_poses, refined_world_points, ba_success = bundle_adjustment(
        initial_camera_poses, initial_world_points, observations, K_matrices_list
    )

    # --- Evaluate Results ---
    if ba_success:
        print("\n--- Refined State ---")
        # Calculate final total reprojection error
        final_total_reproj_error_sq = 0.0
        for cam_idx, point_idx, obs_u, obs_v in observations:
            R_cam, t_cam = refined_camera_poses[cam_idx]
            K_cam = K_matrices_list[cam_idx]
            P_cam = K_cam @ np.hstack((R_cam, t_cam.reshape(3, 1)))
            
            X_3d = refined_world_points[point_idx]
            
            residuals = reprojection_error(np.array([obs_u, obs_v]), X_3d, P_cam)
            final_total_reproj_error_sq += np.sum(residuals**2)
        final_rmse = np.sqrt(final_total_reproj_error_sq / len(observations))
        print(f"Final RMS Reprojection Error: {final_rmse:.4f} pixels")

        print(f"\nRMS Reprojection Error Reduced: {initial_rmse > final_rmse}")

        # Evaluate camera pose accuracy
        print("\n--- Camera Pose Accuracy ---")
        for i in range(num_cameras):
            R_est, t_est = refined_camera_poses[i]
            R_true_cam, t_true_cam = true_camera_poses[i]
            
            R_diff_norm = np.linalg.norm(R_est - R_true_cam, 'fro')
            t_diff_norm = np.linalg.norm(t_est - t_true_cam, 'fro')
            print(f"Camera {i}: R diff={R_diff_norm:.6f}, t diff={t_diff_norm:.6f}")

        # Evaluate 3D point accuracy
        print("\n--- 3D Point Accuracy ---")
        avg_point_diff = np.mean(np.linalg.norm(refined_world_points - true_world_points, axis=1))
        max_point_diff = np.max(np.linalg.norm(refined_world_points - true_world_points, axis=1))
        print(f"Average 3D point position difference: {avg_point_diff:.6f} units")
        print(f"Max 3D point position difference: {max_point_diff:.6f} units")

        # A successful BA run should significantly reduce the reprojection error
        # and bring the estimated poses and points closer to their ground truth values.
        if final_rmse < initial_rmse and final_rmse < (observation_noise_std_dev * 0.8): # Expect improvement
            print("\n✅ Bundle Adjustment Test Passed: Reprojection error reduced and within expected range!")
        else:
            print("\n❌ Bundle Adjustment Test Failed: Reprojection error not sufficiently reduced or too high.")
    else:
        print("\nBundle Adjustment did not succeed.")