# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:24:59 2015

@author: Alan
"""
import numpy as np
from scipy.linalg import svd
from math import log
#import matplotlib.pyplot as plt
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
        self.us, self.en, self.pcs = svd(self.X, full_matrices=0)
        
    def cutoff_index(self,energy_treshold):
        k = 0
        accumulated = 0
        absolute_treshold = energy_treshold* sum(self.en)
        for e in self.en:
            accumulated += e
            k += 1
            if (accumulated >= absolute_treshold):
                return k
            
    def project_data(self, data, energy):
        '''
        accepts single row vector or matrix data with data along rows
        returns reduced dimensionality data
        '''
        self.dim = self.cutoff_index(energy)
        centered = data-self.mean[:,np.newaxis]
        return np.dot(self.pcs[:self.dim],centered)
        
    def get_pca_rep(self, vector):
        coeffs = self.project_data(vector)
        return np.dot(self.pcs.transpose(), coeffs) + self.mean

class MultivariateGaussian(object):
    
    @classmethod
    def from_data_matrix(cls, feature_matrix):
        mean = np.mean(feature_matrix, axis=1)
        cov = np.cov(feature_matrix)
        return cls(cov,mean)
    
    def __init__(self, cov,mean):
        self.mean = mean
        self.cov = cov
        self.icov = np.linalg.inv(self.cov)
        logdet = 2* sum(np.log(np.diag(np.linalg.cholesky(cov))))

    @staticmethod
    def kld(x1, x2):
        if x1 == x2:
            return 0
        try:
            return 0.5 * (x1.logdet-x2.logdet) + np.trace(np.dot(x2.icov,x1.cov)) + np.dot(np.dot((x2.mean - x1.mean)[np.newaxis,:],x1.icov),(x2.mean-x1.mean))-len(x1.mean)
        except ValueError, e:
            print np.linalg.det(x1.cov)/np.linalg.det(x2.cov)
            print x1.cov
            print np.linalg.det(x1.cov)
            raise e
    
    @staticmethod
    def skld(x1, x2):
        return (MultivariateGaussian.kld(x1,x2) + MultivariateGaussian.kld(x2,x1))/2.
         
    def kld_left_centroid(self,mvn, weight= 0.5):
        if self == mvn:
            return self
        
        mean = (1-weight) * (np.dot(self.icov, self.mean)) + weight * (np.dot(mvn.icov,mvn.mean))
        cov = (1-weight) * (0.5 *self.icov) + (weight) + (0.5 * mvn.icov)
        cov = np.linalg.inv(2*cov)
        mean = np.dot(cov,mean)
        #cov = np.linalg.inv((1.-weight)*self.icov + weight*mvn.icov)
        #mean = np.dot(cov, ((1.-weight)*np.dot(self.icov, self.mean) + weight * np.dot(mvn.icov, mvn.mean) ) )
        
        if(np.linalg.det(cov) < 0):
            print "left problem"
        return MultivariateGaussian(cov, mean)
        
    def kld_right_centroid(self,mvn, weight= 0.5):
        if self == mvn:
            return self
        mean = ((1.-weight)*self.mean + weight * mvn.mean)
        cov =   -1. * np.dot(mean[:,np.newaxis],mean[:,np.newaxis].transpose())
        cov += ((1.-weight)* -(np.dot(self.mean[:,np.newaxis],self.mean[:,np.newaxis].transpose())+ self.cov)
                     + weight * -(np.dot(mvn.mean[:,np.newaxis],mvn.mean[:,np.newaxis].transpose())+ mvn.cov))
        cov = -(cov + np.dot(mean[:,np.newaxis],mean[:,np.newaxis].transpose()))
        if(np.linalg.det(cov) < 0):
            print "right problem"
        return MultivariateGaussian(cov, mean)
        
    def skld_centroid(self,mvn, weight = 0.5):
        if self == mvn:
            return self
        cr = self.kld_right_centroid(mvn, weight)
        cl = self.kld_left_centroid(mvn, weight)
        mean = 0.5 * (cr.mean + cl.mean)
        mean = np.asmatrix(mean)
        mean1 = np.asmatrix(self.mean)
        mean2 = np.asmatrix(mvn.mean)
        cov1 = np.asmatrix(self.cov)
        cov2 = np.asmatrix(mvn.cov)
        cov = 0.5*((cov1 + mean1.transpose()*mean1) + (cov2 + mean2.transpose()*mean2)) - mean.transpose()*mean
        if(np.linalg.det(cov) < 0):
            print "center problem"
        else:
            print "no problem"
        return MultivariateGaussian(np.asarray(cov), np.asarray(mean))

    def __eq__(self, other):
        delta = np.linalg.norm(self.mean-other.mean) + np.linalg.norm(self.cov.flatten()-other.cov.flatten())
        if isinstance(other, self.__class__):
            if ( delta > 0.0000001):
                return False
            else:
                return True
        else:
            return False

'''
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
'''