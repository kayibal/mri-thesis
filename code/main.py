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

#TODO can have various files
class DataElement:
    def __init__(self,path,name):
        cov = []
        mean = []

        os.chdir(path)
        self.path = path
        self.name = name
        self.artist_name = os.path.split(path)[1].split('-')[0]
        self.id = os.path.split(path)[1].split('-')[1]
        self.valid = False
        for fl in glob.glob("*"+name+data_extension):
            if "mfcc" in fl:
                self.mfcc = np.load(fl)
            elif "fluc_comp" in fl: 
                self.fluc_comp = np.load(fl)
            elif "fluc_" in fl:
                self.fluc = np.load(fl)
            elif "mean" in fl:
                mean = np.load(fl)
            elif "cov" in fl:
                cov = np.load(fl)
            self.valid = True
        if len(mean) > 0 and len(cov) > 0:
            self.mvg = MultivariateGaussian(cov, mean)

    def __repr___(self):
        return "" + self.artist_name + " | " + self.name

    def __str___(self):
        return "" + self.artist_name + " | " + self.name

def create_fluctuation_data(audio_dir):
    for(directory, _,files) in os.walk(audio_dir):
        for audio in glob.glob(os.path.join(directory,audio_extension)):
            print audio
            start = time()
            filename = re.search(r"\/([^\/]*)\.mp3$", audio).group(1)
            rel_path = audio.replace(filename, '')
            save_path = os.path.join(rel_path,"fluc_"+filename+'.npy')
            if not os.path.isfile(save_path):
                try:
                    a = FluctuationPattern(audio,chunk=16,Hz=11025,)
                    fm = a.get_feature_matrix()
                    #a.plot_frame(fm)

                    if not os.path.exists(rel_path):
                        os.makedirs(rel_path)
                    np.save(save_path,fm)
                    print "last file took: " + str(round(time()-start,3))
                    print "saved_to" + save_path
                except SmallFileError, e:
                    print "ERROR: " + e.message
            else:
                 print "skipped file exists" + save_path

def create_mfcc_data(audio_dir):
    for(directory, _,files) in os.walk(audio_dir):
        for audio in glob.glob(os.path.join(directory,audio_extension)):
            print audio
            start = time()
            filename = re.search(r"\/([^\/]*)\.mp3$", audio).group(1)
            rel_path = audio.replace(filename, '')
            save_path = os.path.join(rel_path,"mfcc_"+filename+'.npy')
            if not os.path.isfile(save_path):
                try:
                    mfcc = MFCC(audio,mel_bands=24)
                    mvg = MultivariateGaussian.from_data_matrix(mfcc.mfcc)
                    #a.plot_frame(fm)
                    if not os.path.exists(rel_path):
                        os.makedirs(rel_path)
                    np.save(save_path, mfcc.mfcc)
                    np.save(os.path.join(rel_path,"mean_"+filename), mvg.mean)
                    np.save(os.path.join(rel_path,"cov_"+filename), mvg.cov)
                    print "last file took: " + str(round(time()-start,3))
                    print "saved_to" + save_path
                except SmallFileError, e:
                    print "ERROR: " + e.message
            else:
                 print "skipped file exists" + save_path

def collect_data():
    data = []
    for(directory, _,files) in os.walk(data_dir):
        if(directory != data_dir):
            print directory
            for i in range(1,4): 
                d = DataElement(directory,str(i))
                if d.valid:
                    data.append(d)
    return data

   
def reduce_data():
    flucs = []
    data = collect_data()
    for element in data:
        flucs.append(element.fluc.flatten())
    flucs = np.vstack(flucs)
    p = Pca(flucs, energy_treshold=0.5)
    #save compressed vector
    for element in data:
        if True:#not os.path.isfile(element.path+"fluc_comp_1.npy"):
            fluc_comp = p.project_data(element.fluc.flatten()[:,np.newaxis])
            np.save(os.path.join(element.path,"fluc_comp_"+element.name), fluc_comp)

    """reduced = p.project_data(flucs.transpose())
                np.save("reduced_data",reduced)"""
    np.save("/home/kayibal/sc-recom/code/data_aq/pca/energies",p.en)
    np.save("/home/kayibal/sc-recom/code/data_aq/pca/mean",p.mean)
    np.save("/home/kayibal/sc-recom/code/data_aq/pca/pcs",p.pcs)
    
def train_som():
    r = int(np.random.rand(1)[0]*10000)
    np.random.seed(9697)
    data = collect_data()
    t_data = []
    for el in data:
        t_data.append(el.fluc_comp)
    t_data = np.asarray(t_data)
    som = SOM(10,t_data[0].shape[0])
    som.start_learning(t_data)
    som.save('/home/kayibal/sc-recom/code/data_aq/brain/som')
    #som.visualize()

def train_som_mvg():
    t_data = collect_data()
    som = SomMvg(8,t_data[0].fluc_comp.shape[0],t_data)
    som.start_learning()
    som.save('/home/kayibal/sc-recom/code/data_aq/brain/som_mvg')


#reduce_data()
#train_som_mvg()
