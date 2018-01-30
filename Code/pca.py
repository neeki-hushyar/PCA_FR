import numpy as np
import sys


class PCA:
    def __init__(self, components, matrix):
        self.components = components # number of principal components
        self.matrix = matrix # grey-scale value matrix of image
        self.mean = None # mean of data - row-wise
        self.covariance = None # covariance matrix
        self.evals = None # evals of covariance matrix
        self.evecs = None # evecs of covariance matrix
        self.princ_comps = None # principal components of cov mat
        self.reduced = None # reduced data based on top x princ comps
        self.evec_matrix = None # evecs with top x evals
        self.recon = None # final reconstruction

    def covariance_matrix(self):
        """
        self.covariance = matrix.transpose*matrix where n is the
        # of rows/columns - doesn't matter which because it's a square image.
        """
        self.covariance = np.dot(self.matrix, self.matrix.transpose())

    def evecs_and_evals(self):
        """
        Computes eigenvalues/vectors of the symmetric covariance matrix.
        """
        self.evals, self.evecs = np.linalg.eigh(self.covariance)
    
    def reduce_dimensions(self):
        """
        Project centered data onto principal axis to yield principal components
        aka Z = XY where X is centered data and Y is principal components
        """
        self.reduced = np.dot(self.matrix, self.evec_matrix)
    
    def principal_components(self):
        """
        Find x largest eigenvalues. The corresponding eigenvectors have the 
        highest variance.
        """
        top_evecs = np.empty((self.components, len(self.matrix)))
        indices = (self.evals.argsort()[::-1]) # descending
        prnc_cmp_locs = indices[:self.components]
        ctr = 0
        for i in prnc_cmp_locs:
            top_evecs[ctr] = self.evecs[i] # evec per row - highest eval first
            ctr += 1
        self.evec_matrix = top_evecs.transpose()

    def get_evec_matrix(self):
        return self.evec_matrix

    def reconstruct(self):
        """
        Reconstructs image. Never get's called.
        """
        self.recon = np.dot(self.reduced, self.evec_matrix.transpose())
        return self.recon

