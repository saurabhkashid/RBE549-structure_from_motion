import numpy as np

def traingulate_points (K1, R1, t1, K2, R2, t2, pts1, pts2):
    # Construct projection matrices P1 = K1 * [R1 | t1] and P2 = K2 * [R2 | t2]
    P1 = K1 @ np.hstack((R1, t1.reshape(3, 1)))
    P2 = K2 @ np.hstack((R2, t2.reshape(3, 1)))

    num_points = pts1.shape[0]
    triangulated_points_4d = np.zeros((num_points, 4))

    for i in range(num_points):
        u1, v1 = pts1[i, 0], pts1[i, 1]
        u2, v2 = pts2[i, 0], pts2[i, 1]
        """
        A = [x]* P
        x point in image frame and P is projection matrix P = k.(R,t)
        """
        # Construct matrix A (4x4) for the linear system AX = 0
        # Rows are (u_i * P_3rd_row - P_1st_row) and (v_i * P_3rd_row - P_2nd_row)
        A = np.zeros((4, 4))

        # Equations from the first camera
        A[0, :] = u1 * P1[2, :] - P1[0, :]
        A[1, :] = v1 * P1[2, :] - P1[1, :]

        # Equations from the second camera
        A[2, :] = u2 * P2[2, :] - P2[0, :]
        A[3, :] = v2 * P2[2, :] - P2[1, :]

        # Solve AX = 0 using SVD
        # The solution X is the right-singular vector corresponding to the smallest singular value.
        # This is the last column of V (or last row of V_transpose in numpy output).
        U_A, S_A, V_A_transpose = np.linalg.svd(A)
        
        # The 3D point in homogeneous coordinates is the last column of V_A_transpose.T
        X_homogeneous = V_A_transpose[-1, :] # V_A_transpose is already V^T from numpy.linalg.svd

        # Store the result
        triangulated_points_4d[i] = X_homogeneous

    return triangulated_points_4d

if __name__ == "__main__":
    # --- Setup: Dummy Camera Parameters and Points ---
    # These are the same parameters used in the previous example
    theta = np.deg2rad(5) # Rotate 5 degrees around Y-axis
    R_ground_truth = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t_ground_truth = np.array([0.1, 0.05, 0.02]) # Small translation

    K1_test = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    K2_test = K1_test # Assuming same camera for simplicity

    # Generate some dummy 3D points
    num_test_points = 10 # Fewer points for easier inspection
    np.random.seed(0) # For reproducibility
    test_points_3d_gt = np.random.rand(num_test_points, 3) * 10 - 5 # Points between -5 and 5
    test_points_3d_gt[:, 2] = np.abs(test_points_3d_gt[:, 2]) + 5 # Ensure positive depth

    # Project 3D points to 2D in camera 1 (reference frame: R=I, t=[0,0,0])
    R_cam1 = np.eye(3)
    t_cam1 = np.zeros(3)
    P1_test = K1_test @ np.hstack((R_cam1, t_cam1.reshape(3,1)))
    test_pts1_homog = (P1_test @ np.hstack((test_points_3d_gt, np.ones((num_test_points, 1)))).T).T
    # Normalize by the homogeneous coordinate (last element)
    test_pts1 = test_pts1_homog[:, :2] / test_pts1_homog[:, 2:]

    # Project 3D points to 2D in camera 2 (using ground truth pose)
    R_cam2_gt = R_ground_truth
    t_cam2_gt = t_ground_truth
    P2_test = K2_test @ np.hstack((R_cam2_gt, t_cam2_gt.reshape(3,1)))
    test_pts2_homog = (P2_test @ np.hstack((test_points_3d_gt, np.ones((num_test_points, 1)))).T).T
    # Normalize by the homogeneous coordinate (last element)
    test_pts2 = test_pts2_homog[:, :2] / test_pts2_homog[:, 2:]

    # Add some noise to simulated 2D points to make it realistic
    test_pts1 += np.random.randn(*test_pts1.shape) * 0.5
    test_pts2 += np.random.randn(*test_pts2.shape) * 0.5

    print("--- Manual Triangulation Test ---")
    print(f"Number of points to triangulate: {num_test_points}")

    # Perform manual triangulation
    triangulated_3d_points_manual_homog = traingulate_points(
        K1_test, R_cam1, t_cam1, K2_test, R_cam2_gt, t_cam2_gt, test_pts1, test_pts2
    )

    if triangulated_3d_points_manual_homog is not None:
        # Convert homogeneous to Euclidean for comparison
        triangulated_3d_points_manual = triangulated_3d_points_manual_homog[:, :3] / triangulated_3d_points_manual_homog[:, 3:]

        print("\nGround Truth 3D Points:\n", test_points_3d_gt[:5]) # Show first 5
        print("\nManually Triangulated 3D Points (Euclidean):\n", triangulated_3d_points_manual[:5]) # Show first 5

        # Calculate difference (error)
        diff = np.linalg.norm(triangulated_3d_points_manual - test_points_3d_gt, axis=1)
        print(f"\nAverage Euclidean distance error: {np.mean(diff):.4f}")
        print(f"Max Euclidean distance error: {np.max(diff):.4f}")

        # --- Compare with OpenCV's triangulatePoints (for verification) ---
        import cv2

        P1_cv = K1_test @ np.hstack((R_cam1, t_cam1.reshape(3, 1)))
        P2_cv = K2_test @ np.hstack((R_cam2_gt, t_cam2_gt.reshape(3, 1)))

        triangulated_3d_points_opencv_homog = cv2.triangulatePoints(
            P1_cv, P2_cv, test_pts1.T.astype(np.float32), test_pts2.T.astype(np.float32)
        )
        triangulated_3d_points_opencv = triangulated_3d_points_opencv_homog[:3, :] / triangulated_3d_points_opencv_homog[3, :]
        triangulated_3d_points_opencv = triangulated_3d_points_opencv.T # Transpose back to N x 3

        print("\nOpenCV Triangulated 3D Points (Euclidean):\n", triangulated_3d_points_opencv[:5]) # Show first 5

        diff_opencv = np.linalg.norm(triangulated_3d_points_opencv - test_points_3d_gt, axis=1)
        print(f"\nOpenCV Average Euclidean distance error: {np.mean(diff_opencv):.4f}")
        print(f"OpenCV Max Euclidean distance error: {np.max(diff_opencv):.4f}")
