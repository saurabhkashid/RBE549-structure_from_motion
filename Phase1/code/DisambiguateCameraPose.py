import numpy as np

def check_chirality(t_cam2, R_cam2, world_pts_homogenous):
    # Convert 3D homogeneous points to non-homogeneous (X, Y, Z)
    points_3d_euclidean = world_pts_homogenous[:, :3] / world_pts_homogenous[:, 3:]

    # Check depth in the first camera's coordinate system (Z > 0)
    # Assuming the first camera is at origin with identity rotation
    depths_cam1 = points_3d_euclidean[:, 2]

    # Check depth in the second camera's coordinate system
    # Transform points from camera 1's frame to camera 2's frame:
    # X_cam2 = R_cam2 @ X_cam1 + t_cam2
    # This can be written as X_cam2 = R_cam2 @ (X_cam1 - (-t_cam2)) for clarity
    # or just: X_cam2 = R_cam2 @ X_cam1 + t_cam2
    points_3d_cam2 = (R_cam2 @ points_3d_euclidean.T + t_cam2.reshape(3, 1)).T
    depths_cam2 = points_3d_cam2[:, 2]

    # Count points that have positive depth in both camera systems
    in_front_count = np.sum((depths_cam1 > 0) & (depths_cam2 > 0))

    return in_front_count > 0, in_front_count

def disambiguate_camera_pose(possible_poses, world_pts):
    """
    Estimates the unique camera pose (R, t) from the 4 possible poses
    by performing chirality checks.

    Args:
        world_pts: points in world frame from cam1
        possible_poses: possible poses from the essential matrix

    Returns:
        tuple: A tuple (R, t) representing the unique rotation and translation
               from camera 1 to camera 2. Returns (None, None) if no valid pose is found.
    """
    max_in_front_count = 0

    for i, (R_cam2, t_cam2) in enumerate(possible_poses):
        print(f"\n--- Solution {i+1} ---")
        print("R:\n", R_cam2)
        print("t:", t_cam2)

        # Check chirality for the triangulated points
        world_pts_homogenous = np.hstack((world_pts[i],np.ones((world_pts[i].shape[0],1))))
        is_valid, in_front_count = check_chirality(t_cam2, R_cam2, world_pts_homogenous)

        print(f"Points in front of both cameras: {in_front_count}/{len(world_pts_homogenous)}")

        if is_valid and in_front_count > max_in_front_count:
            max_in_front_count = in_front_count
            best_R = R_cam2
            best_t = t_cam2
            best_pts = world_pts_homogenous
            print("Status: Candidate for correct pose (more points in front).")
        elif is_valid:
            print("Status: Candidate for correct pose (but fewer points in front than current best).")
        else:
            print("Status: Not a valid physical pose (points not in front of both cameras).")

    if best_R is not None:
        print("\n--- Final Best Pose ---")
        print("Selected R:\n", best_R)
        print("Selected t:\n", best_t)
        return best_R, best_t.reshape(3,1), best_pts[:,:3]
    else:
        print("\nNo valid camera pose found.")
        return None, None, None