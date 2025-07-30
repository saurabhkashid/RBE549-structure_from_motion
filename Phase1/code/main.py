import numpy as np
import cv2
import argparse
import glob
import matplotlib.pyplot as plt
from util import *
from GetInliersRANSAC import ransac_homography
from EstimateFundamentalMatrix import fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import get_essentail_matrix
from ExtractCameraPose import get_camera_pose
from LinearTriangulation import traingulate_points
from DisambiguateCameraPose import disambiguate_camera_pose
from NonlinearTriangulation import non_linear_triangulation
from PnPRANSAC import ransac_pnp
from NonlinearPnP import NonlinearPnP
from BuildVisibilityMatrix import add_new_observations_to_map, bundle_adjustment_sparsity
from BundleAdjustment import bundle_adjustment


def main ():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--basePath', default=r"D:\\courses\\computer_vision\\Home_Work\\RBE549-structure_from_motion\\Phase1\\Data", help="add the base path for data")
    Parser.add_argument('--outputPath',default=r"D:\\courses\\computer_vision\\Home_Work\\RBE549-structure_from_motion\\Phase1\\Data\\output", help="Path to save the output images")
    Parser.add_argument('--calibrationMatrix', default=r"Phase1\\Data\\calibration.txt",help="path to the calibration matrix")

    Arg = Parser.parse_args()
    base_path = Arg.basePath
    output_path = Arg.outputPath
    calib_mat_file = Arg.calibrationMatrix

    # read the image form the data
    # Assuming features_files contains paths like 'matching01.txt', 'matching02.txt', etc.
    features_files = glob.glob(f"{base_path}\\matching*.txt")
    
    # Infer total number of images based on existing data or a predefined count
    # This is a heuristic; ideally, you'd know the exact number of images.
    num_total_images = len(glob.glob(f"{base_path}\\image*.png")) 
    if num_total_images == 0:
        # If no image files, infer from match files or set a default.
        # Assuming matching files are `matchingX.txt` where X refers to image X.
        # Or, if matching files are pairwise `matching_X_Y.txt`, then parse image indices.
        # For this example, let's assume 5 images as in the mock.
        num_total_images = 5 
    
    # get the calibration matrix
    K = np.loadtxt(calib_mat_file)

    # image_matches_filtered will store: (img_base_idx, img_target_idx) -> (x_base_uv, x_target_uv)
    # As per constraint, no local indices are stored here.
    image_matches_filtered = {}

    # Read all feature matches from files
    # `all_feature_data_from_files[i]` will contain the raw matches from processing `features_files[i]`.
    # Based on the constraint, `parse_matching_file` just returns a list of (uv_base, uv_target) pairs.
    all_feature_data_from_files = [] 
    for i, feature_file in enumerate(features_files):
        all_feature_data_from_files.append(parse_matching_file(feature_file, i + 1))


    # Define all match pairs to process (e.g., (1,2), (1,3), etc. if you have 5 images)
    match_pairs = []
    for i_idx in range(1, num_total_images + 1):
        for j_idx in range(i_idx + 1, num_total_images + 1):
            match_pairs.append((i_idx, j_idx))
    
    # Process each match pair to get filtered inliers for Homography
    for match_pair in match_pairs:
        base_idx, target_idx = match_pair 
        
        # `all_feature_data_from_files[base_idx-1]` should contain raw matches where `base_idx` is the query image.
        # `extract_pair_matches` filters for `target_idx`.
        uvi_raw, uvj_raw = extract_pair_matches(all_feature_data_from_files[base_idx-1], target_idx)
        
        if len(uvi_raw) > 0:
            # RANSAC for Homography (for initial filtering, not for PnP)
            inliers_mask_h = ransac_homography(uvi_raw, uvj_raw) # Returns boolean mask or indices
            
            # Ensure inliers_mask_h is a boolean mask for consistent indexing
            if inliers_mask_h.dtype == bool:
                uvi_inliers = uvi_raw[inliers_mask_h]
                uvj_inliers = uvj_raw[inliers_mask_h]
            else: # Assume it's an array of indices
                uvi_inliers = uvi_raw[inliers_mask_h]
                uvj_inliers = uvj_raw[inliers_mask_h]
            
            # Store filtered matches (only UV coordinates, as per constraint)
            if len(uvi_inliers) >= 8: 
                image_matches_filtered[match_pair] = (uvi_inliers, uvj_inliers)
            else:
                print(f"  Warning: Not enough inliers ({len(uvi_inliers)}) for {match_pair} after Homography RANSAC. Skipping.")

    # --- Initialize data structures for SfM ---
    Cset = [] # List of camera rotation matrices (3x3)
    Rset = [] # List of camera translation vectors (3x1)
    X = np.empty((0, 3), dtype=np.float64) # Global set of 3D points
    
    # Store observations mapping global point_id to 2D observation
    # Format: (camera_idx_0_based, global_point_id, u, v)
    global_observations = []
    global_point_id_counter = [0] # Mutable counter for unique ID for each triangulated 3D point

    # A map from (image_idx_1_based, (u_coord_rounded, v_coord_rounded)) to global_point_id
    # This uses (u,v) tuples as keys due to the constraint of no local indices.
    mapped_2d_to_3d_points = [{} for _ in range(num_total_images + 1)] # Using 1-indexing for cameras


    # --- For first two images (Bootstrapping Phase) ---
    print("\n--- Processing the first two images (Image 1 and Image 2) ---")
    img1_idx_1_based, img2_idx_1_based = 1, 2 
    
    try:
        x1_uv, x2_uv = image_matches_filtered[(img1_idx_1_based, img2_idx_1_based)]
        if len(x1_uv) < 8: 
            raise ValueError(f"Not enough inliers ({len(x1_uv)}) for initial pair ({img1_idx_1_based},{img2_idx_1_based}) after RANSAC. Exiting.")
    except KeyError:
        print(f"Error: Initial match pair ({img1_idx_1_based},{img2_idx_1_based}) not found or had too few inliers after RANSAC. Exiting.")
        return # Exit main function


    # Estimate Fundamental Matrix
    F = fundamental_matrix(x1_uv, x2_uv)
    # Estimate Essential Matrix
    E = get_essentail_matrix(F, K)

    # Extract Camera Pose Candidates
    possible_poses = get_camera_pose(E) 

    # Perform Linear Triangulation for each candidate pose
    Xset_linear_candidates = [] 
    R_cam1_fixed, t_cam1_fixed = np.eye(3), np.zeros((3,1)) # Camera 1 is world origin (0-indexed cam)
    for pose in possible_poses:
        R_cand, t_cand = pose[0], pose[1]
        current_X_lin_homogeneous = traingulate_points(K, R_cam1_fixed, t_cam1_fixed, K, R_cand, t_cand, x1_uv, x2_uv)
        
        valid_points_mask = current_X_lin_homogeneous[:, 3] != 0
        current_X_lin_3d = np.full((current_X_lin_homogeneous.shape[0], 3), np.nan) 
        current_X_lin_3d[valid_points_mask] = current_X_lin_homogeneous[valid_points_mask, :3] / current_X_lin_homogeneous[valid_points_mask, 3:]
        Xset_linear_candidates.append(current_X_lin_3d)

    # Disambiguate Camera Pose
    R_cam2, t_cam2, X_initial_guess_raw = disambiguate_camera_pose(possible_poses, Xset_linear_candidates) # this takes homohgenous coords
    
    valid_initial_guess_mask = ~np.isnan(X_initial_guess_raw).any(axis=1) & ~np.isinf(X_initial_guess_raw).any(axis=1)
    X_initial_guess_filtered = X_initial_guess_raw[valid_initial_guess_mask]
    
    # Filter corresponding 2D points (no local indices here)
    x1_uv_filtered = x1_uv[valid_initial_guess_mask]
    x2_uv_filtered = x2_uv[valid_initial_guess_mask]

    if X_initial_guess_filtered.shape[0] < 2: 
        print("  Disambiguation yielded too few valid initial 3D points. Exiting.")
        return 

    # Add initial cameras to Cset, Rset
    Cset.append(t_cam1_fixed) # Camera 1 (index 0)
    Rset.append(R_cam1_fixed)
    Cset.append(t_cam2)       # Camera 2 (index 1)
    Rset.append(R_cam2)
    print(f"  Initialized {len(Cset)} cameras (Image 1 and Image 2).")

    # Perform Non-linear triangulation for bootstrapping points
    X_bootstrapped_refined = non_linear_triangulation(K, t_cam1_fixed, R_cam1_fixed, K, t_cam2, R_cam2,
                                                      x1_uv_filtered, x2_uv_filtered, X_initial_guess_filtered)
    
    # Check for valid refined points (e.g., positive depth in both cameras)
    final_bootstrapped_points = []
    final_x1_uv_for_obs = []
    final_x2_uv_for_obs = []

    P_cam1_final = K @ np.hstack((R_cam1_fixed, t_cam1_fixed))
    P_cam2_final = K @ np.hstack((R_cam2, t_cam2))

    for idx, X_pt in enumerate(X_bootstrapped_refined):
        _, proj_homog1 = project_3d_to_2d(P_cam1_final, X_pt)
        _, proj_homog2 = project_3d_to_2d(P_cam2_final, X_pt)
        
        # Check cheirality and if point is finite
        if proj_homog1[2] > 0 and proj_homog2[2] > 0 and not np.isinf(X_pt).any() and not np.isnan(X_pt).any():
            final_bootstrapped_points.append(X_pt)
            final_x1_uv_for_obs.append(x1_uv_filtered[idx])
            final_x2_uv_for_obs.append(x2_uv_filtered[idx])

    # Append bootstrapped points to the global X
    X = np.array(final_bootstrapped_points)
    print(f"  Bootstrapped {X.shape[0]} initial 3D points.")

    # Populate global_observations and mapped_2d_to_3d_points for initial two cameras
    for i in range(X.shape[0]):
        current_global_id = global_point_id_counter[0]
        
        global_observations.append((0, current_global_id, final_x1_uv_for_obs[i][0], final_x1_uv_for_obs[i][1]))
        global_observations.append((1, current_global_id, final_x2_uv_for_obs[i][0], final_x2_uv_for_obs[i][1]))
        
        # Map (u,v) tuple to global 3D point ID (using rounded values for robustness)
        mapped_2d_to_3d_points[img1_idx_1_based][tuple(np.round(final_x1_uv_for_obs[i], 2))] = current_global_id
        mapped_2d_to_3d_points[img2_idx_1_based][tuple(np.round(final_x2_uv_for_obs[i], 2))] = current_global_id
        
        global_point_id_counter[0] += 1
    
    print(f"  Initialized {len(global_observations)} observations for bootstrapping.")
    print(f"  Next available global point ID: {global_point_id_counter[0]}")


    # --- Incremental Reconstruction Loop (for images 3 onwards) ---
    # `i` represents the 1-indexed image number for the current image being registered
    # Loop from 3rd image up to the last image
    print("\n--- Starting Incremental Reconstruction Loop ---")
    for i in range(3, num_total_images + 1): 
        print(f"\nProcessing new image: Image {i} (1-indexed)")
        current_cam_idx_0_based = len(Cset) # This will be the 0-indexed ID for the new camera 'i'

        # --- Step 1: Register new camera 'i' using PnP ---
        # Find 2D points in current image 'i' that match existing 3D points in `X`.
        
        x_i_for_pnp = [] # 2D points in current image 'i'
        X_for_pnp = []   # Corresponding 3D points in world coordinates
        
        # Iterate over already registered cameras (0-indexed) to find matches with image `i`
        registered_cam_indices_0_based = list(range(len(Cset))) 

        # We'll try to find PnP correspondences from ANY registered camera to the current `i`
        pnp_data_found = False
        for registered_cam_idx_0_based in registered_cam_indices_0_based:
            ref_cam_idx_1_based = registered_cam_idx_0_based + 1 # Convert to 1-based for image_matches_filtered key

            # Try to get matches (ref_cam -> current_i) or (current_i -> ref_cam)
            x_ref_uv_raw, x_i_uv_raw = (None, None) # Initialize
            if (ref_cam_idx_1_based, i) in image_matches_filtered:
                x_ref_uv_raw, x_i_uv_raw = image_matches_filtered[(ref_cam_idx_1_based, i)]
            elif (i, ref_cam_idx_1_based) in image_matches_filtered: # Reversed pair
                x_i_uv_raw, x_ref_uv_raw = image_matches_filtered[(i, ref_cam_idx_1_based)]
            else:
                continue # No matches found for this pair

            if x_ref_uv_raw is None or len(x_ref_uv_raw) == 0:
                continue

            # Filter matches: only use those where the reference feature (in ref_cam_idx_1_based)
            # is already mapped to a 3D point (using its rounded UV coordinates as key)
            for k in range(len(x_ref_uv_raw)):
                ref_uv_rounded = tuple(np.round(x_ref_uv_raw[k], 2))
                if ref_uv_rounded in mapped_2d_to_3d_points[ref_cam_idx_1_based]:
                    global_id = mapped_2d_to_3d_points[ref_cam_idx_1_based][ref_uv_rounded]
                    
                    if global_id < X.shape[0]: # Ensure global_id is valid for current X
                        x_i_for_pnp.append(x_i_uv_raw[k]) 
                        X_for_pnp.append(X[global_id]) 
                        pnp_data_found = True
        
        X_for_pnp = np.array(X_for_pnp)
        x_i_for_pnp = np.array(x_i_for_pnp)

        if len(x_i_for_pnp) < 6:
            print(f"  Not enough common 3D points ({len(x_i_for_pnp)}) to register image {i} via PnP. Skipping this image.")
            continue

        # Perform RANSAC PnP
        try:
            R_linear, t_linear, inliers_mask_pnp = ransac_pnp(x_i_for_pnp, X_for_pnp, K)
        except ValueError as e:
            print(f"  RANSAC PnP for Image {i} failed: {e}. Skipping this image.")
            continue
        
        if np.sum(inliers_mask_pnp) < 6:
            print(f"  RANSAC PnP for Image {i} found too few inliers ({np.sum(inliers_mask_pnp)}). Skipping this image.")
            continue

        x_i_inliers_for_nonlinear = x_i_for_pnp[inliers_mask_pnp]
        X_pnp_inliers_for_nonlinear = X_for_pnp[inliers_mask_pnp]

        # Perform Non-linear PnP refinement
        R_refine, t_refine, pnp_success = NonlinearPnP(R_linear, t_linear, x_i_inliers_for_nonlinear, X_pnp_inliers_for_nonlinear, K)

        if not pnp_success:
            print(f"  NonlinearPnP for Image {i} failed. Using RANSAC's linear result.")
            R_refine, t_refine = R_linear, t_linear
        
        # Add the newly registered camera's pose to Cset and Rset
        Cset.append(t_refine)
        Rset.append(R_refine)
        print(f"  Registered Image {i}. Total cameras now: {len(Cset)}.")

        # Update global_observations for EXISTING 3D points seen by the NEW camera
        # Loop through the PnP inliers to add observations for the new camera `i`
        for k in range(len(x_i_inliers_for_nonlinear)):
            # Find the global ID of this 3D point (by matching its 3D coordinates in global X)
            current_X_3d_point = X_pnp_inliers_for_nonlinear[k]
            
            # This is a potentially slow operation (linear search for 3D point in X)
            # A more efficient way would be if PnP could return the original global_id
            # However, given the constraint of only (u,v) being passed around, this lookup is needed.
            global_pt_id_matches = np.where(np.all(np.isclose(X, current_X_3d_point, atol=1e-6), axis=1))[0]
            if len(global_pt_id_matches) > 0:
                global_pt_id = global_pt_id_matches[0] # Get the first match
                global_observations.append((current_cam_idx_0_based, global_pt_id,
                                            x_i_inliers_for_nonlinear[k][0], x_i_inliers_for_nonlinear[k][1]))
                
                # Map the (rounded) UV feature in image 'i' to its global 3D point ID
                mapped_2d_to_3d_points[i][tuple(np.round(x_i_inliers_for_nonlinear[k], 2))] = global_pt_id
            else:
                print(f"  Warning: Inlier 3D point not found in global X during observation update.")


        # --- Step 2: Triangulate new points ---
        # Find matches between Image 1 (reference) and current Image `i`
        # that correspond to *new* 3D points (not already in `X`).
        
        triangulation_match_pair_key = (img1_idx_1_based, i)

        x_ref_for_tri_raw, x_i_for_tri_raw = (None, None)
        if triangulation_match_pair_key in image_matches_filtered:
            x_ref_for_tri_raw, x_i_for_tri_raw = image_matches_filtered[triangulation_match_pair_key]
        elif (i, img1_idx_1_based) in image_matches_filtered: # Check reverse pair
             x_i_for_tri_raw, x_ref_for_tri_raw = image_matches_filtered[(i, img1_idx_1_based)]

        if x_ref_for_tri_raw is None or x_ref_for_tri_raw.shape[0] == 0:
            print(f"  No filtered matches for new point triangulation for (Image {img1_idx_1_based}, Image {i}). Skipping new points.")
            continue # Continue to next image in the loop

        x_ref_new_pts_for_tri = []
        x_i_new_pts_for_tri = []
        
        # Iterate through matches from the reference camera (Image 1) to the current image (i)
        for j in range(len(x_ref_for_tri_raw)):
            # If the feature from Image 1 at (rounded) UV `x_ref_for_tri_raw[j]`
            # is *not* already mapped to a global 3D point
            ref_uv_rounded = tuple(np.round(x_ref_for_tri_raw[j], 2))
            if ref_uv_rounded not in mapped_2d_to_3d_points[img1_idx_1_based]:
                x_ref_new_pts_for_tri.append(x_ref_for_tri_raw[j])
                x_i_new_pts_for_tri.append(x_i_for_tri_raw[j])
        
        x_ref_new_pts_for_tri = np.array(x_ref_new_pts_for_tri)
        x_i_new_pts_for_tri = np.array(x_i_new_pts_for_tri)

        if len(x_ref_new_pts_for_tri) < 2: 
            print(f"  Not enough new matches ({len(x_ref_new_pts_for_tri)}) to triangulate for Image {i}. Skipping adding new points.")
            continue
        
        # Get the pose of the reference camera (Image 1) and the newly registered camera (Image i)
        R1_tri = Rset[img1_idx_1_based - 1] # R_cam1_fixed is at 0-indexed position 0
        t1_tri = Cset[img1_idx_1_based - 1] # t_cam1_fixed is at 0-indexed position 0
        Ri_tri = R_refine # The refined pose of current image `i` (just registered)
        ti_tri = t_refine

        # Linear Triangulation
        X_new_linear_homogeneous = traingulate_points(K, R1_tri, t1_tri, K, Ri_tri, ti_tri,
                                                      x_ref_new_pts_for_tri, x_i_new_pts_for_tri)
        
        valid_new_points_mask = X_new_linear_homogeneous[:, 3] != 0
        X_new_linear_3d_filtered = np.full((X_new_linear_homogeneous.shape[0], 3), np.nan)
        X_new_linear_3d_filtered[valid_new_points_mask] = X_new_linear_homogeneous[valid_new_points_mask, :3] / X_new_linear_homogeneous[valid_new_points_mask, 3:]

        # Filter corresponding 2D points (no local indices needed here)
        x_ref_new_pts_filtered = x_ref_new_pts_for_tri[valid_new_points_mask]
        x_i_new_pts_filtered = x_i_new_pts_for_tri[valid_new_points_mask]

        if X_new_linear_3d_filtered.shape[0] == 0:
            print("  No valid points after linear triangulation filtering. Skipping adding new points.")
            continue

        # Non-linear Triangulation for refinement
        X_new_refined_all = non_linear_triangulation(K, t1_tri, R1_tri, K, ti_tri, Ri_tri,
                                                     x_ref_new_pts_filtered, x_i_new_pts_filtered, X_new_linear_3d_filtered)
        
        # --- Step 3: Add new points to global map and update visibility ---
        points_added_count = 0
        P_cam1_ref = K @ np.hstack((R1_tri, t1_tri))
        P_cami_new = K @ np.hstack((Ri_tri, ti_tri))

        for p_idx in range(X_new_refined_all.shape[0]):
            X_new_single_point = X_new_refined_all[p_idx]
            
            # Check cheirality (positive Z in both cameras) after refinement and if point is finite
            _, projected_homog_cam1_ref = project_3d_to_2d(P_cam1_ref, X_new_single_point)
            _, projected_homog_cami_new = project_3d_to_2d(P_cami_new, X_new_single_point)
            
            if (projected_homog_cam1_ref[2] > 0 and projected_homog_cami_new[2] > 0 and
                not np.isinf(X_new_single_point).any() and not np.isnan(X_new_single_point).any()):
                
                # Add to global 3D point set
                X = np.vstack((X, X_new_single_point))
                
                # Add observations and update UV-to-GlobalID mapping
                # `create_observations_list` is adjusted to work with UV coords as keys.
                # It will increment global_point_id_counter internally
                add_new_observations_to_map(
                    global_observations, 
                    current_cam_idx_0_based, # 0-based index of new camera
                    global_point_id_counter, 
                    x_i_new_pts_filtered[p_idx], # New 2D point UV in current camera
                    x_ref_new_pts_filtered[p_idx], # New 2D point UV in reference camera
                    img1_idx_1_based - 1, # 0-based index of ref camera (Image 1)
                    mapped_2d_to_3d_points, # The mapping dictionary
                    )
                points_added_count += 1
        
        print(f"  Added {points_added_count} new 3D points. Total 3D points: {X.shape[0]}")
    
    print("\n--- Incremental Reconstruction Loop Finished ---")
    print(f"Final total cameras: {len(Cset)}")
    print(f"Final total 3D points: {X.shape[0]}")
    print(f"Final total observations: {len(global_observations)}")

    # --- Step 4: Final Bundle Adjustment ---
    print("\n--- Initiating Final Bundle Adjustment ---")
    initial_camera_poses_for_ba = [(Cset[j], Rset[j]) for j in range(len(Cset))]
    
    # create the sparcity matrix 
    sparsity_mat = bundle_adjustment_sparsity(len(Cset),X.shape[0],np.array([obs[0] for obs in global_observations], dtype=int),np.array([obs[1] for obs in global_observations], dtype=int))
    if X.shape[0] > 0 and len(global_observations) > 0:
        refined_camera_poses, refined_world_points, ba_success = bundle_adjustment(
            initial_camera_poses_for_ba, 
            X, # Use the accumulated global X
            global_observations, # Use the accumulated global_observations
            K
        )

        if ba_success:
            print("Bundle Adjustment completed and refined the reconstruction.")
            # Update the global Cset, Rset, X with refined values
            Cset[:] = [pose[0] for pose in refined_camera_poses] 
            Rset[:] = [pose[1] for pose in refined_camera_poses] 
            X[:] = refined_world_points 
            # plot the 3d points
            ax = plt.subplot(projection="3d")
            ax.scatter(X[:,0],X[:,1],X[:,2])
            plt.plot()
            print("DOne")
        else:
            print("Bundle Adjustment failed or skipped. Reconstruction remains unrefined.")
    else:
        print("Skipping Bundle Adjustment: No 3D points or observations to optimize.")


if __name__ == "__main__":
    main()