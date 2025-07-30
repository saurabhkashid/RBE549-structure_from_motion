import numpy as np
from scipy.sparse import lil_matrix

def add_new_observations_to_map(global_observations_list, current_image_idx_0_based, global_point_id_counter,
                                x_2d_new_in_current_cam_uv_single, 
                                x_2d_new_in_ref_cam_uv_single, ref_cam_idx_0_based,
                                mapped_2d_to_3d_points):
    """
    Adds observations for a single newly triangulated 3D point to the global map.
    This function effectively extends the 'visibility list' (global_observations_list)
    and updates the 2D-to-3D point mapping.

    Args:
        global_observations_list (list): The list of (camera_idx_0_based, global_point_id, u, v) tuples.
                                         This is the core structure for the Bundle Adjustment's
                                         "sparsity matrix" or observation list.
        current_image_idx_0_based (int): 0-based index of the newly registered camera (the 'current' image).
        global_point_id_counter (list): A mutable list/array holding the next available global point ID.
                                         This is incremented within the function.
        x_2d_new_in_current_cam_uv_single (np.ndarray): 1x2 array of the new 2D point UV in the current camera.
        x_2d_new_in_ref_cam_uv_single (np.ndarray): 1x2 array of the new 2D point UV in the reference camera.
        ref_cam_idx_0_based (int): 0-based index of the reference camera (e.g., Image 1, which has existing 3D points).
        mapped_2d_to_3d_points (list of dicts): A list where each element is a dictionary
                                                for a 1-indexed camera. The dictionary maps
                                                (rounded_u, rounded_v) tuple to a global_point_id.
    """
    ROUND_PRECISION = 2 # Precision for rounding UV coordinates for dictionary keys

    # Get the next unique global 3D point ID
    new_global_id = global_point_id_counter[0]

    # Add observation for the reference camera
    global_observations_list.append((ref_cam_idx_0_based, new_global_id,
                                     x_2d_new_in_ref_cam_uv_single[0], x_2d_new_in_ref_cam_uv_single[1]))

    # Add observation for the current (newly registered) camera
    global_observations_list.append((current_image_idx_0_based, new_global_id,
                                     x_2d_new_in_current_cam_uv_single[0], x_2d_new_in_current_cam_uv_single[1]))

    # Update the 2D to 3D mapping for both cameras
    # Ensure dictionaries exist for the 1-indexed camera IDs
    ref_cam_idx_1_based = ref_cam_idx_0_based + 1
    current_cam_idx_1_based = current_image_idx_0_based + 1

    if ref_cam_idx_1_based >= len(mapped_2d_to_3d_points):
        # Extend the list if needed (should ideally be pre-allocated based on num_total_images)
        for _ in range(ref_cam_idx_1_based - len(mapped_2d_to_3d_points) + 1):
            mapped_2d_to_3d_points.append({})
            
    if current_cam_idx_1_based >= len(mapped_2d_to_3d_points):
        for _ in range(current_cam_idx_1_based - len(mapped_2d_to_3d_points) + 1):
            mapped_2d_to_3d_points.append({})

    print("debug: ",tuple(np.round(x_2d_new_in_ref_cam_uv_single, ROUND_PRECISION)))
    mapped_2d_to_3d_points[ref_cam_idx_1_based][tuple(np.round(x_2d_new_in_ref_cam_uv_single, ROUND_PRECISION))] = new_global_id
    mapped_2d_to_3d_points[current_cam_idx_1_based][tuple(np.round(x_2d_new_in_current_cam_uv_single, ROUND_PRECISION))] = new_global_id

    # Increment the global point ID counter for the next new point
    global_point_id_counter[0] += 1

    print(f"  Added observations for new point (Global ID: {new_global_id}) from Image {ref_cam_idx_1_based} and Image {current_cam_idx_1_based}.")


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Constructs the sparsity pattern of the Jacobian matrix for Bundle Adjustment.
    
    The Jacobian has `m` rows (2 * num_observations) and `n` columns
    (num_cameras * 6 + num_points * 3).
    
    Each observation (a 2D point) contributes to 2 residuals (u and v).
    These 2 residuals depend on:
    1. The 6 parameters of the camera that observed the point.
    2. The 3 parameters of the 3D point itself.
    
    Args:
        n_cameras (int): Total number of cameras.
        n_points (int): Total number of 3D world points.
        camera_indices (np.ndarray): (M,) array, 0-based index of the camera for each observation.
        point_indices (np.ndarray): (M,) array, 0-based global point ID for each observation.
    Returns:
        scipy.sparse.lil_matrix: The sparsity pattern matrix.
    """
    m = camera_indices.size * 2 # 2 residuals (u,v) per observation
    n = n_cameras * 6 + n_points * 3 # 6 params per camera (R_vec, t), 3 params per point (X,Y,Z)
    A = lil_matrix((m, n), dtype=int)

    # `i` is the row index in the residuals array (0 to 2*num_observations - 1)
    # `k` is the index of the observation in the camera_indices/point_indices arrays (0 to num_observations - 1)
    # So for observation `k`, its residuals are at rows `2*k` and `2*k+1`.
    i = np.arange(camera_indices.size) # This represents `k` in my explanation above

    # Mark non-zero entries for camera parameters
    # The parameters for camera `c` are at columns `c*6` to `c*6 + 5`
    for s in range(6): # For each of the 6 camera parameters (3 R_vec, 3 t)
        A[2 * i, camera_indices * 6 + s] = 1   # For the u-residual
        A[2 * i + 1, camera_indices * 6 + s] = 1 # For the v-residual

    # Mark non-zero entries for 3D point parameters
    # The parameters for point `p` are at columns `n_cameras*6 + p*3` to `n_cameras*6 + p*3 + 2`
    # `n_cameras * 6` is the offset to start of point parameters
    for s in range(3): # For each of the 3 point parameters (X, Y, Z)
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

# Example Usage (assuming some dummy data and structures)
if __name__ == '__main__':
    # Mock initial data, similar to main_test's state after bootstrapping
    num_total_images = 3 # For this example
    Cset = [np.zeros((3,1)), np.array([[0],[0],[1]])] # 2 cameras
    Rset = [np.eye(3), np.eye(3)]
    X = np.array([[1.0, 1.0, 5.0], [2.0, 2.0, 6.0], [3.0, 3.0, 7.0]]) # 3 initial 3D points
    
    # Initialize global_observations and global_point_id_counter
    global_observations = []
    global_point_id_counter = [len(X)] # Counter starts after existing points
    
    mapped_2d_to_3d_points = [{} for _ in range(num_total_images + 1)] # 1-indexed

    # Simulate some initial observations for the bootstrapped points
    # Point 0 seen by Cam 0 and Cam 1
    global_observations.append((0, 0, 100, 100)) # Cam 0 sees Pt 0 at (100,100)
    global_observations.append((1, 0, 110, 110)) # Cam 1 sees Pt 0 at (110,110)
    mapped_2d_to_3d_points[1][tuple(np.round(np.array([100,100]),2))] = 0
    mapped_2d_to_3d_points[2][tuple(np.round(np.array([110,110]),2))] = 0
    
    # Point 1 seen by Cam 0 and Cam 1
    global_observations.append((0, 1, 200, 200)) # Cam 0 sees Pt 1 at (200,200)
    global_observations.append((1, 1, 210, 210)) # Cam 1 sees Pt 1 at (210,210)
    mapped_2d_to_3d_points[1][tuple(np.round(np.array([200,200]),2))] = 1
    mapped_2d_to_3d_points[2][tuple(np.round(np.array([210,210]),2))] = 1

    # Point 2 seen by Cam 0 and Cam 1
    global_observations.append((0, 2, 300, 300)) # Cam 0 sees Pt 2 at (300,300)
    global_observations.append((1, 2, 310, 310)) # Cam 1 sees Pt 2 at (310,310)
    mapped_2d_to_3d_points[1][tuple(np.round(np.array([300,300]),2))] = 2
    mapped_2d_to_3d_points[2][tuple(np.round(np.array([310,310]),2))] = 2

    print(f"Initial global_observations (length {len(global_observations)}):")
    for obs in global_observations:
        print(f"  Cam: {obs[0]}, Pt ID: {obs[1]}, UV: ({obs[2]:.2f}, {obs[3]:.2f})")
    print(f"Next global point ID: {global_point_id_counter[0]}")
    print("\nInitial mapped_2d_to_3d_points:")
    for cam_idx, mapping in enumerate(mapped_2d_to_3d_points):
        if mapping:
            print(f"  Image {cam_idx}: {mapping}")


    # Simulate triangulating a new point from Image 1 (ref_cam_idx_0_based=0) and a new Image 3 (current_image_idx_0_based=2)
    new_3d_point_coords = np.array([5.0, 5.0, 10.0]) # A new 3D point
    new_2d_ref = np.array([400.0, 400.0]) # Its projection in Image 1
    new_2d_current = np.array([450.0, 450.0]) # Its projection in Image 3

    # Add the new 3D point to the global set (as done in main_test)
    X = np.vstack((X, new_3d_point_coords))

    # Add observations for this new point
    add_new_observations_to_map(
        global_observations,
        current_image_idx_0_based=2, # Image 3 is the third camera (0-indexed: 2)
        global_point_id_counter=global_point_id_counter,
        x_2d_new_in_current_cam_uv_single=new_2d_current,
        x_2d_new_in_ref_cam_uv_single=new_2d_ref,
        ref_cam_idx_0_based=0, # Image 1 is the first camera (0-indexed: 0)
        mapped_2d_to_3d_points=mapped_2d_to_3d_points
    )

    print(f"\nAfter adding new observations (length {len(global_observations)}):")
    for obs in global_observations:
        print(f"  Cam: {obs[0]}, Pt ID: {obs[1]}, UV: ({obs[2]:.2f}, {obs[3]:.2f})")
    print(f"Next global point ID: {global_point_id_counter[0]}")
    print("\nUpdated mapped_2d_to_3d_points:")
    for cam_idx, mapping in enumerate(mapped_2d_to_3d_points):
        if mapping:
            print(f"  Image {cam_idx}: {mapping}")

    # Conceptual sparsity matrix (for visualization/understanding, not typically passed to BA)
    num_cameras_in_map = len(Cset) + 1 # Include the newly added camera
    num_points_in_map = X.shape[0]

    conceptual_sparsity_matrix = build_sparsity_matrix_for_ba(num_cameras_in_map, num_points_in_map, global_observations)
    print("\nConceptual Sparsity Matrix (Cameras x 3D Points):")
    print(conceptual_sparsity_matrix)
    print(" (Row = Camera Index, Column = 3D Point ID. 1 = Observed, 0 = Not Observed)")