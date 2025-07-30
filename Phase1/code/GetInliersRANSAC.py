import numpy as np
import random
import cv2

def ransac_homography(kps_src, kps_dst, max_iters=1000, threshold= 5):
    
    # add 1 in the end to make it 3,1
    one_mat = np.ones((kps_src.shape[0],1))
    kpt_src_homo = np.hstack((kps_src,one_mat)).T # shape 3*n
    max_inliers = []
    for iter in range(max_iters):
        if len(max_inliers)>len(kps_dst):
            break

        # pick random 4 points 
        rand_pts_idx = random.sample(range(len(kps_src)-1),4)
        selected_kpt_src = kps_src[rand_pts_idx]
        selected_kpt_dst = kps_dst[rand_pts_idx]

        H = cv2.findHomography(selected_kpt_src,selected_kpt_dst)
        if H[0] is None:
            continue
        x_transformed = H[0]@kpt_src_homo
        try:
            x_transformed = np.array([x_transformed[0]/x_transformed[2],x_transformed[1]/x_transformed[2]]).T # normalize the x,y with z
        except RuntimeError:
            continue
        error = np.sum((kps_dst - x_transformed)**2, axis=1)

        curr_inliers_no = np.sum(error < threshold)
        if curr_inliers_no > np.sum(max_inliers):
            max_inliers = (error < threshold)
    
    return max_inliers