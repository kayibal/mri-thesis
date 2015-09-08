# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:24:59 2015

@author: Alan
"""
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import os
import glob

class Pca:
    '''Principal Component Analysis Class
    
        Computes the principal componentes of a given matrix. Using fast SVD technique.
        
        Parameters
            ----------
            raw_data : ndarray
                a 2-dimensional matrix of size n x k, where n is the number of
                experiments/observations and k the number of features
            energy_treshold: float[0,1]
                the relative treshold how much of the energy the pca should preserve
                determines the amount of eigenvectors the pca matrix will contain

        Attributes
        -------
        X (ndarray)     :  mean centered data matrix
        pcs (ndarray)   :  principal components to reach treshold
    
    '''
    def __init__(self, raw_data, energy_treshold=0.9):        
        #center data arround center
        self.energy_treshold = energy_treshold
        self.mean = np.mean(raw_data, axis=0)
        self.X = raw_data - self.mean
        #compute svd
        self.us, self.en, pcs = svd(self.X, full_matrices=0)
        self.dim = self.cutoff_index()
        self.pcs = pcs[:self.dim]
        
    def cutoff_index(self):
        k = 0
        accumulated = 0
        absolute_treshold = self.energy_treshold* sum(self.en)
        for e in self.en:
            accumulated += e
            k += 1
            if (accumulated >= absolute_treshold):
                return k
            
    def project_data(self, data):
        '''
        accepts single row vector or matrix data with data along rows
        returns reduced dimensionality data
        '''
        centered = data-self.mean[:,np.newaxis]
        return np.dot(self.pcs,centered)
        
    def get_pca_rep(self, vector):
        coeffs = self.project_data(vector)
        return np.dot(self.pcs.transpose(), coeffs) + self.mean

        
data_dir = '/home/kayibal/thesis/dataset/spectral_data'
extension = '*.fluc'

os.chdir(data_dir)
data = []
for(directory, _,files) in os.walk("."):
    for fl in glob.glob(data_dir+os.path.join(directory[1:],extension)):
        print fl
        data.append(np.fromfile(fl))
data = np.vstack(data)
p = Pca(data, energy_treshold=0.5)
print p.dim
reduced = p.project_data(data.transpose())
np.save("reduced_data",reduced)
np.save("mean",p.mean)
np.save("pcs",p.pcs)
