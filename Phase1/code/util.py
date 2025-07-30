import numpy as np
import cv2 


def parse_matching_file(filepath, base_image_id):
    """
    Parses a matching file like matching3.txt where each feature is seen in base_image
    and other images (e.g., I3 ↔ I4, I3 ↔ I5).

    Args:
        filepath (str): path to the matching file
        base_image_id (int): ID of the base image (e.g., 3 for matching3.txt)

    Returns:
        matches: list of tuples (base_id, [u_base, v_base], match_id, [u_match, v_match], feature_id)
    """
    matches = []
    feature_id = 0

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        tokens = line.strip().split()
        if len(tokens) < 6:
            continue

        n_matches = int(tokens[0])
        u_base = float(tokens[4])
        v_base = float(tokens[5])

        for i in range(n_matches - 1):
            idx = 6 + i * 3
            img_id = int(tokens[idx])
            u = float(tokens[idx + 1])
            v = float(tokens[idx + 2])

            matches.append((
                base_image_id, [u_base, v_base],
                img_id, [u, v],
                feature_id
            ))

        feature_id += 1

    return matches

def extract_pair_matches(matches, target_image_id):
    """
    From all matches, get correspondences between base image and target image.
    Returns: two arrays of shape (N, 2)
    """
    pts1 = []
    pts2 = []
    for base_id, uv1, match_id, uv2, _ in matches:
        if match_id == target_image_id:
            pts1.append(uv1)
            pts2.append(uv2)
    return np.array(pts1), np.array(pts2)

def draw_inlier_matches(img1, img2, pts1, pts2):
    img = cv2.hconcat([img1, img2])
    offset = img1.shape[1]
    for (u1, v1), (u2, v2) in zip(pts1, pts2):
        pt1 = (int(u1), int(v1))
        pt2 = (int(u2 + offset), int(v2))
        cv2.line(img, pt1, pt2, (0, 255, 0), 1)
    cv2.imshow("RANSAC Inliers", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def project_3d_to_2d(P, X):
    X_homog = np.append(X, 1)
    # X = np.atleast_2d(X)  # Ensure X is at least 2D (N, 3)
    # create the homogeneous matrix for multiple points
    # X_homog = np.hstack((X,np.ones((X.shape[0],1))))
    projected_homog = P @ X_homog
    
    # Handle division by zero for points at infinity or behind camera
    if projected_homog[2] == 0:
        return np.array([np.inf, np.inf]), projected_homog
    
    # Normalize to get pixel coordinates [u, v]
    pixel_coords = projected_homog[:2] / projected_homog[2]
    return pixel_coords, projected_homog

def reprojection_error(obs_pts1, X, P):
    # reproject the X to the image frame and calculat the error 
    reproj_2d, _ = project_3d_to_2d(P, X)
    error = obs_pts1 - reproj_2d
    return error

def reprojection_error_pnp(x_2d_obs, X_3d_world, P_matrix):
    errors = np.zeros(len(x_2d_obs))
    for i in range(len(x_2d_obs)):
        reproj_2d, _ = project_3d_to_2d(P_matrix, X_3d_world[i])
        # Handle cases where reprojection might fail (e.g., behind camera)
        if np.isinf(reproj_2d).any() or np.isnan(reproj_2d).any():
            errors[i] = np.inf # Large error to exclude as inlier
        else:
            errors[i] = np.linalg.norm(x_2d_obs[i] - reproj_2d)
    return errors


def extract_common_and_new_features(feature_map):
    pass