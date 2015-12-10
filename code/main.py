# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:34:24 2015

@author: Alan
"""
import os
import glob
from time import time
import numpy as np
from pca import Pca
from classificator import SOM
import re
from audioanalytics import *
from pca import MultivariateGaussian

audio_dir = '/home/kayibal/sc-recom/code/data_aq/data'
data_dir = '/home/kayibal/sc-recom/code/data_aq/spectral_data'
audio_extension = '*.mp3'
data_extension = '*.npy'

class DataElement:
    def __init__(self,path):
        os.chdir(path)
        mean = 0
        cov = 0
        self.name = os.path.dirname.split('-')[0]
        self.id = os.path.dirname.split('-')[1]
        for fl in glob.glob(data_extension):
            if "mfcc" in fl:
                self.mfcc = np.load(fl)
            else if "fluc" in fl:
                self.fluc = np.load(fl)
            else if "mean" in fl:
                mean = np.load(fl)
            else if "cov" in fl:
                cov = np.load(fl)
        self.mvg = MultivariateGaussian(cov, mean)

def create_fluctuation_data():
    for(directory, _,files) in os.walk(audio_dir):
        for audio in glob.glob(os.path.join(directory,audio_extension)):
            print audio
            start = time()
            filename = re.search(r".*\/(\d)\.mp3$", audio).group(1) +'_fluc.npy'
            rel_path = re.search(r".*\/([^\/]*\/)\d\.mp3$", audio).group(1)
            save_path = os.path.join(data_dir,rel_path,"fluc_"+filename)
            if not os.path.isfile(save_path):
                try:
                    a = FluctuationPattern(audio,chunk=16,Hz=11025,)
                    fm = a.get_feature_matrix()
                    #a.plot_frame(fm)
                    if not os.path.exists(os.path.join(data_dir,rel_path)):
                        os.makedirs(os.path.join(data_dir,rel_path))
                    np.save(save_path,fm)

                    print "last file took: " + str(round(time()-start,3))
                    print "saved_to" + save_path
                except SmallFileError, e:
                    print "ERROR: " + e.message
            else:
                 print "skipped file exists" + save_path

def create_mfcc_data():
    for(directory, _,files) in os.walk(audio_dir):
        for audio in glob.glob(os.path.join(directory,audio_extension)):
            print audio
            start = time()
            filename = re.search(r".*\/(\d)\.mp3$", audio).group(1) +'.npy'
            rel_path = re.search(r".*\/([^\/]*\/)\d\.mp3$", audio).group(1)
            save_path = os.path.join(data_dir,rel_path,"mfcc_"+filename)
            if not os.path.isfile(save_path):
                try:
                    mfcc = MFCC(audio,mel_bands=24)
                    mvg = MultivariateGaussian.from_data_matrix(mfcc.mfcc)
                    #a.plot_frame(fm)
                    if not os.path.exists(os.path.join(data_dir,rel_path)):
                        os.makedirs(os.path.join(data_dir,rel_path))
                    np.save(save_path, mfcc.mfcc)
                    np.save(os.path.join(data_dir,rel_path,"mean_"+filename, mvg.mean)
                    np.save(os.path.join(data_dir,rel_path,"cov_"+filename, arr)
                    print "last file took: " + str(round(time()-start,3))
                    print "saved_to" + save_path
                except SmallFileError, e:
                    print "ERROR: " + e.message
            else:
                 print "skipped file exists" + save_path

def collect_data():
    data = []
    for(directory, _,files) in os.walk(audio_dir):
        d = DataElement(directory)
        data.append(d)
   
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
    som = SOM(10,10)
    som.start_learning(reduced)
    som.visualize()