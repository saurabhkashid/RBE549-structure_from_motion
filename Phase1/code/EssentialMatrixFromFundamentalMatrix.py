import numpy as np

def get_essentail_matrix(fundamental_matrix, K):
    E = K.T @ fundamental_matrix @ K
    u, s, v = np.linalg.svd(E)
    
    # enforcing s to 1,1 0 
    s = np.diag([1,1,0])
    E_enforced = u @ s @ v
    
    return E_enforced