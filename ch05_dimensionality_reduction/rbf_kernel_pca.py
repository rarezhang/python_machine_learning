"""
kernel principle component analysis (kernel PCA)
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np 


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation 
    :para X: numpy array, shape=(n, n)
    :para gamma: float, tuning parameter of the RBF kernel
    :para n_components: int, number of principal components to return
    
    :return X_pc: numpy array, shape=(n, k_feature), projected dataset
    :return lamb: list, eigenvalues 
    """
    # calculate pairwise squared Euclidean distances in the M*N dimensional dataset  
    sq_dists = pdist(X, 'sqeuclidean')
    
    # convert pairwise distances into a square matrix 
    mat_sq_dists = squareform(sq_dists)
    
    # compute the symmetric kernel matrix 
    K = exp(-gamma * mat_sq_dists)
    
    # center the kernel matrix 
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # obtaining eigen-pairs from the centered kernel matrix 
    # numpy.eigh returns them in sorted order 
    eigvals, eigvecs = eigh(K)
    
    # collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))
    
    # collect the corresponding eigenvalues 
    lamb = [eigvals[-i] for i in range(1, n_components+1)]
    
    return X_pc, lamb
    