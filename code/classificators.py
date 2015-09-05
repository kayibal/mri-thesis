# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 23:07:30 2015

@author: Alan
"""
import numpy as np
from math import exp, log
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt
class SOM:
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
        
        
        
    def compute_bmu(self, inp):
        #computes distances using euklid norm
        dist = np.linalg.norm(self.weights-inp, axis=2)
        self.winning_neuron = np.unravel_index(np.argmin(dist), dist.shape)
        
    
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
        self.time_const=len(training_data)/log(self.t_initial)
        np.random.shuffle(training_data)
        for inp in training_data:
            self.last_input = inp
            self.calc_time()
            self.calc_learn_rate()
            self.compute_bmu(inp)
            self.synaptic_adaptation()
            self.n += 1
            
    def visualize(self):
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
        plt.imshow(u_matrix, cmap='gray')
        #plt.subplot(1,2,2)
        #plt.imshow(np.floor(self.weights*255).astype('uint8'), origin='upper', aspect='auto', interpolation='nearest')

som = SOM(40,3)
reds = np.random.rand(200,3)*np.asarray([1.,10./255,10./255])
blue = np.random.rand(200,3)*np.asarray([10./255,10./255,1])
green = np.random.rand(200,3)*np.asarray([10./255,1,10./255])
yell = np.random.rand(200,3)*np.asarray([1,0.5,0])/2. + 0.5
training = np.round(np.concatenate((reds,blue,green,yell)))
som.start_learning(training)
som.visualize()
