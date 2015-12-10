# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 23:07:30 2015

@author: Alan
"""
import os
import numpy as np
import random
from math import exp, log
from scipy.spatial.distance import cdist, euclidean
from pca import MultivariateGaussian

class SOM(object):
    def __init__(self,dim, inp_size, spacing=1):
        #init variables
        self.som_dim = dim
        #position nodes in gutter
        np.arange(dim, step=spacing)
        self.pos = np.zeros((dim,dim,2))
        #y-dimension
        self.pos[:,:,1] = np.tile(np.asarray([np.arange(dim, step=spacing)]).transpose(), (1,dim))
        #x-dimensions
        self.pos[:,:,0] = np.tile(np.asarray([np.arange(dim, step=spacing)]), (dim,1))
        #rand init weights
        self.weights = np.random.rand(dim,dim,inp_size)
        
        #time dependent variables
        self.winning_neuron = (0,0)
        self.last_input = None
        self.time_n = 0
        self.learn_n = 0
        
        #time parameters       
        self.n = 1
        self.t_initial=dim*spacing*2
        self.time_const = 0
        
        #learning parameters
        self.l_initial = 0.1
        self.learning_rate = 1000
        
        self.trained = False

        
    def compute_bmu(self, inp):
        #computes distances using euklid norm
        dist = np.linalg.norm(self.weights-inp, axis=2)
        self.winning_neuron = np.unravel_index(np.argmin(dist), dist.shape)
        
    def compute_2nd_bmu(self, inp):
        #computes distances using euklid norm
        dist = np.linalg.norm(self.weights-inp, axis=2)
        return np.unravel_index(np.partition(dist,2)[2], dist.shape)
        
    def calc_time(self):
        self.time_n = self.t_initial * exp(-float(self.n)/self.time_const)
    
    '''
    def top_neighborhood_func(self,x1=self.winning_neuron,x2):
        x1 = tuple(x1)
        x2 = tuple(x2)
        
        d = np.linal.norm(self.pos[x1],self.pos[x2])**2
        return exp(-d/2*self.time_n)
    '''
    
    def calc_learn_rate(self):
        self.learn_n = self.l_initial * exp(-float(self.n)/self.learning_rate)
    
    def synaptic_adaptation(self):
        distance = cdist(self.pos.reshape((-1,2)), np.asarray([self.pos[self.winning_neuron]])).reshape(self.som_dim,self.som_dim,1)
        neighborhood_matrix = np.exp(-distance**2/(2*self.time_n))
        self.weights += self.learn_n*neighborhood_matrix*(self.last_input - self.weights)
        
    def start_learning(self,training_data):
        self.training_data = training_data
        self.time_const=len(training_data)/log(self.t_initial)
        np.random.shuffle(training_data)
        for inp in training_data:
            self.last_input = inp  #remove?
            self.calc_time()
            self.calc_learn_rate()
            self.compute_bmu((inp/np.linalg.norm(inp)))
            self.synaptic_adaptation()
            self.n += 1
        self.trained = True
            
    def compute_topological_error(self, data = False):
        if self.trained:
            if not data:
                #use trained data (default)
                data = self.training_data
            err = 0.
            for inp in data:
                self.compute_bmu(inp)
                scnd = self.compute_2nd_bmu(inp)
                if (euclidean(self.winning_neuron, scnd) > 1.42):
                    err += 1
            return err *1. / len(data)
            
    
    def compute_quantization_error(self, data = False):
        if self.trained:
            if not data:
                #use trained data (default)
                data = self.training_data
            err = 0.
            for inp in data:
                inp = inp/np.linalg.norm(inp)
                self.compute_bmu(inp)
                err += euclidean(self.weights[self.winning_neuron], inp)
            return err / len(data)
    
    
    def fit_predict(self, data):
        if not self.trained:
            self.start_learning(data)
        clusters = []
        for inp in data:
            self.compute_bmu(inp)
            cluster = np.ravel_multi_index(self.winning_neuron,(self.som_dim,self.som_dim))
            distance = euclidean(self.weights[self.winning_neuron], inp)
            clusters.append(cluster)
        return clusters
            

    def save(self, path):
        np.save(path, self.weights)
            
    def visualize(self):
        import matplotlib.pyplot as plt
        u_matrix = np.zeros((self.som_dim-1,self.som_dim-1))
        for i in range(u_matrix.shape[0]):
            for j in range(u_matrix.shape[0]):
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i,j-1])
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i,j+1])
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i-1,j])
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i+1,j])
                
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i - 1,j + 1])
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i + 1,j + 1])
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i - 1,j - 1])
                u_matrix[i,j] += euclidean(self.weights[i,j],self.weights[i + 1,j - 1])
        plt.figure()
        #plt.subplot(1,2,1)
        plt.imshow(np.log(u_matrix), cmap="gray")
        #plt.subplot(1,2,2)
        #plt.imshow(np.floor(self.weights*255).astype('uint8'), origin='upper', aspect='auto', interpolation='nearest')


#data_dir = '/Users/Alan/Documents/thesis/mri-thesis/code/spectral_data'
#3243

class SomMvg(SOM):
    def __init__(self, dim, inp_size, data):
        super(SomMvg, self).__init__(dim, inp_size)
        self.mvgs = np.empty(shape=(dim,dim), dtype=object)
        self.training_data = np.asarray(data, dtype=object)
        for i in range(dim):
            for j in range(dim):
                self.mvgs[i,j] = random.choice(data).mvg

    def compute_bmu(self, inp, mvg_weight = 1.):
        ''' input must be a tuple of a vector and mvg element'''
        dist = np.linalg.norm(self.weights-inp[0], axis=2)
        mvg_dist = np.zeros((self.som_dim,self.som_dim))
        for i in range(self.som_dim):
            for j in range(self.som_dim):
                mvg_dist[i,j] = MultivariateGaussian.skld(self.mvgs[i,j], inp[1])
        self.winning_neuron = np.unravel_index(np.argmin(dist+mvg_weight*mvg_dist), dist.shape)

    def synaptic_adaptation(self):
        distance = cdist(self.pos.reshape((-1,2)), np.asarray([self.pos[self.winning_neuron]])).reshape(self.som_dim,self.som_dim,1)
        neighborhood_matrix = np.exp(-distance**2/(2*self.time_n))
        self.weights += self.learn_n*neighborhood_matrix*(self.last_input[0] - self.weights)
        for i in range(self.som_dim):
            for j in range(self.som_dim):
                self.mvgs[i,j] = self.mvgs[i,j].skld_centroid(self.last_input[1], self.learn_n*neighborhood_matrix[i,j])

    def start_learning(self):
        self.time_const=len(self.training_data)/log(self.t_initial)
        np.random.shuffle(self.training_data)
        for inp in self.training_data:
            inp = (inp.fluc_comp[:,0]/np.linalg.norm(inp.fluc_comp), inp.mvg,)
            self.last_input = inp  #remove?
            self.calc_time()
            self.calc_learn_rate()
            self.compute_bmu(inp)
            self.synaptic_adaptation()
            self.n += 1
            print self.n
        self.trained = True

    def save(self, path):
        means = np.zeros(self.som_dim**2)
        covs = np.zeros(self.som_dim**2)
        for idx, mvg in enumerate(self.mvgs.flatten()):
            means[idx] = mvg.mean
            covs[idx] = mvg.covs
        np.save(path+"_means", means)
        np.save(path+"_covs", covs)



'''
som = SOM(40,3)
reds = np.random.rand(200,3)*np.asarray([1.,10./255,10./255])
blue = np.random.rand(200,3)*np.asarray([10./255,10./255,1])
green = np.random.rand(200,3)*np.asarray([10./255,1,10./255])
yell = np.random.rand(200,3)*np.asarray([1,0.5,0])/2. + 0.5
training = np.round(np.concatenate((reds,blue,green,yell)))
som.start_learning(training)
som.visualize()
'''