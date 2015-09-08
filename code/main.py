# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:34:24 2015

@author: Alan
"""
import os
import glob
import time
import numpy as np
from pca import Pca
from classificators import SOM
from audioanalytics import FluctuationPattern, SmallFileError

audio_dir = '/Users/Alan/Documents/thesis/mri-thesis/code/music'
data_dir = '/home/kayibal/thesis/dataset/spectral_data'
audio_extension = '*.mp3'
data_extension = '*.npy'

def create_fluctuation_data():
    for(directory, _,files) in os.walk(audio_dir):
        for audio in glob.glob(os.path.join(directory,audio_extension)):
            print audio
            start = time()
            filename = audio[1:].split(".")[0]
            if not os.path.isfile(data_dir+filename):
                try:
                    a = FluctuationPattern(audio,chunk=16,Hz=11025,)
                    fm = a.get_feature_matrix()
                    #a.plot_frame(fm)
                    if not os.path.exists(data_dir+os.path.dirname(filename)):
                        os.makedirs(data_dir+os.path.dirname(filename))
                    np.save(os.path.join(data_dir,"fluc_"+filename),fm)
                    print "last file took: " + str(round(time()-start,3))
                except SmallFileError, e:
                    print e.message
    
        	else:
                 print "skipped"
                 
def reduce_data():    
    os.chdir(data_dir)
    data = []
    for(directory, _,files) in os.walk("."):
        for fl in glob.glob(data_dir+os.path.join(directory[1:],data_extension)):
            print fl
            data.append(np.fromfile(fl))
    data = np.vstack(data)
    p = Pca(data, energy_treshold=0.5)
    print p.dim
    reduced = p.project_data(data.transpose())
    np.save("reduced_data",reduced)
    np.save("energies",p.en)
    np.save("mean",p.mean)
    np.save("pcs",p.pcs)
    
def train_som():
    r = int(np.random.rand(1)[0]*10000)
    np.random.seed(9697)
    print r
    os.chdir(data_dir)
    pcs = np.load("pcs.npy")
    reduced = np.load("reduced_data.npy").transpose()
    som = SOM(100,46)
    som.start_learning(reduced)
    som.visualize()