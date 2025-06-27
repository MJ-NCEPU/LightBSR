# -*- coding: utf-8 -*-
import numpy as np

from scipy import ndimage
import scipy
import scipy.stats as ss
import scipy.io as io

from scipy.ndimage import filters, measurements, interpolation
from scipy.interpolate import interp2d


"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
"""


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


"""
# --------------------------------------------
# calculate PCA projection matrix
# --------------------------------------------
"""


def get_pca_matrix(x, dim_pca=15):
    """
    Args:
        x: 225x10000 matrix
        dim_pca: 15

    Returns:
        pca_matrix: 225x15
    """
    C = np.dot(x, x.T)
    w, v = scipy.linalg.eigh(C)
    pca_matrix = v[:, -dim_pca:]

    return pca_matrix


def cal_pca_matrix(path='PCA_matrix.mat', ksize=15, l_max=12.0, dim_pca=15, num_samples=500):
    kernels = np.zeros([ksize*ksize, num_samples], dtype=np.float32)
    for i in range(num_samples):

        theta = np.pi*np.random.rand(1)
        l1    = 0.1+l_max*np.random.rand(1)
        l2    = 0.1+(l1-0.1)*np.random.rand(1)

        k = anisotropic_Gaussian(ksize=ksize, theta=theta[0], l1=l1[0], l2=l2[0])

        # util.imshow(k)

        kernels[:, i] = np.reshape(k, (-1), order="F")  # k.flatten(order='F')

    # io.savemat('k.mat', {'k': kernels})

    pca_matrix = get_pca_matrix(kernels, dim_pca=dim_pca)

    io.savemat(path, {'p': pca_matrix})

    return pca_matrix
