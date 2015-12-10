# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:28:02 2015

@author: Alan
"""
import os
import numpy as np
from numpy.fft import rfftn
from math import floor, log, exp, ceil
from pydub import AudioSegment
from scipy import interpolate
from scipy.fftpack import fft, dct
#import matplotlib.pyplot as plt
from pylab import get_current_fig_manager
plt = {}
class AudioAnalytics(object):
    """AudioAnalytics base class.

        This class stores all important constants and prepares the raw audio data.
        It contains some often needed utility functions
        
        Parameters
        ----------
        audiofile : string
            The path to an audiofile
        freq : int
            Target frequency in Hz. Analytic methods will use this value and audio will be processed to match it.
        mono : bool
            If true seperate channels will be joined
        chunk : int
            framesize the audio will be split into - logarithmic to base 2
        window_size: float
            in seconds will be floored to the prev power of 2
        terhardt : bool
            if True the Terhardt Outer Ear Model will be applied before mapping to bark scale
        max_length:
            maximal length of clip to be analyzed if the file is bigger the middle part is taken, if it is smaller
            then the whole file is taken. Intro and outro are always removed
    
        Attributes
        -------
        loudn_bark (ndarray) :  a 2 dimensional array containg the bark filterbanks. Mapping from phon curves to bark spectrum to fit the given bark frequencies.
        eq_loudness (ndarray) : 6 phon curves with 22 samples
        loudn_freq (ndarray) :  is the x (frequency) value for eq_values object
        phons (ndarray) :       available phon levels
        bark :                  border definitions of the 24 critical bands of hearing (bark scale)
        fs_model :              fluctuation strength model fluctuations arround 4KHz get highest weights
        
        n_frames :              number of frames the audio has been split into
        raw :                   the raw audio file data        
        window_size :           number of fft window size to use in samples
        """
    # zwicker & fastl: psychoacoustics 1999, page 159
    bark = np.array([100,   200,  300,  400,  510,  630,   770,   920, 
            1080, 1270, 1480, 1720, 2000, 2320,  2700,  3150,
            3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500])
    
    #the 6 curves which define the phon scale
    eq_loudness = np.array(
        [[ 55,   40,  32,  24,  19,  14, 10,  6,  4,  3,  2,  
            2,    0,  -2,  -5,  -4,   0,  5, 10, 14, 25, 35], 
         [ 66,   52,  43,  37,  32,  27, 23, 21, 20, 20, 20,  
           20,   19,  16,  13,  13,  18, 22, 25, 30, 40, 50], 
         [ 76,   64,  57,  51,  47,  43, 41, 41, 40, 40, 40,
         39.5, 38,  35,  33,  33,  35, 41, 46, 50, 60, 70], 
         [ 89,   79,  74,  70,  66,  63, 61, 60, 60, 60, 60,  
           59,   56,  53,  52,  53,  56, 61, 65, 70, 80, 90], 
         [103,   96,  92,  88,  85,  83, 81, 80, 80, 80, 80,  
           79,   76,  72,  70,  70,  75, 79, 83, 87, 95,105], 
         [118,  110, 107, 105, 103, 102,101,100,100,100,100,  
           99,   97,  94,  90,  90,  95,100,103,105,108,115]])
           
    #defines mapping of tone frequency to the 22 discrete values of the phone curves    
    #or is the x value for eq_values object
    loudn_freq = np.array(
        [31.62,   50,  70.7,   100, 141.4,   200, 316.2,  500, 
         707.1, 1000,  1414,  1682,  2000,  2515,  3162, 3976,
         5000,  7071, 10000, 11890, 14140, 15500])
    
    #phone levels we have data for
    phons = np.array([0,3,20,40,60,80,100,101])
    
    def __init__(self, audiofile, Hz=11025, mono=True, chunk=16, window_size=0.023, terhardt=False, max_length = 120):
        self.freq = Hz
        self.max_length =  max_length / round( 2.**chunk/self.freq)
        self.mono = mono
        self.chunk = chunk
        self.audiofile = audiofile
        self.bark_filter_bank()
        self.terhardt = terhardt
        self.window_size = 2**int(ceil(log(window_size*Hz)/log(2)))
        self.frame_size = 2**chunk / self.window_size
        self.load_raw_audio()
        self.prepare_raw_data()
        
    def compute_sone(self, plot=True, separate=True, frames=1):
        if plot:
            self.show_mono_waveform(self.raw)
            if not separate:
                plt.figure(figsize=(9,12)).subplots_adjust(hspace=0.3)
                plt.suptitle('Sone Computation Steps', fontsize=16)
        self.powerspectrum()
        if plot:
            if separate:
                plt.figure()
            else:
                plt.subplot(3,2,1).set_title("Powerspectrum")
            CS = plt.imshow(self.processed,origin='lower', aspect='auto', interpolation='nearest') 
            plt.colorbar(CS)
        self.map_to_bark()
        if plot:
            if separate:
                plt.figure()
            else:
                plt.subplot(3,2,2).set_title("Signal in bark scale")
            CS = plt.imshow(self.processed,origin='lower', aspect='auto', interpolation='nearest')
            plt.colorbar(CS)
        self.spectral_masking()
        if plot:
            if separate:
                plt.figure()
            else:
                plt.subplot(323).set_title("Aplied spectral masking")
            CS = plt.imshow(self.processed,origin='lower', aspect='auto', interpolation='nearest')
            plt.colorbar(CS)
        self.map_to_decibel()
        
        if plot:
            if separate:
                plt.figure()
            else:
                plt.subplot(324).set_title("Converted to Db unit")
            CS = plt.imshow(self.processed,origin='lower', aspect='auto', interpolation='nearest')
            plt.colorbar(CS)
        self.map_to_phon()
        if plot:
            if separate:
                plt.figure()
            else:
                plt.subplot(325)
            CS = plt.imshow(self.processed,origin='lower', aspect='auto', interpolation='nearest')
            plt.colorbar(CS)
        self.map_to_sone()
        if plot:
            if separate:
                plt.figure()
            else:
                plt.subplot(326)
            CS = plt.imshow(self.processed,origin='lower', aspect='auto', interpolation='nearest')
            plt.colorbar(CS)

    def load_raw_audio(self):
        '''Loads raw pcm data'''
        self.raw = AudioSegment.from_file(self.audiofile)
        if(self.mono):
            self.raw = self.raw.set_channels(1)
        self.raw = self.raw.set_frame_rate(self.freq)
        dt = np.dtype("i"+str(self.raw.sample_width))
        self.raw = np.fromstring(self.raw._data,dtype=dt)
    
    def prepare_raw_data(self, bounds=[]):
        '''
        Removes intro and outro and reduces data load by dividing into chunks and only keeping every third of it
        '''
        if len(bounds) == 2:
            self.processed= self.raw[bounds[0] : bounds[1]]
            self.raw= self.raw[bounds[0] : bounds[1]]
        else:
            #chopping into chunks
            chunk_size = 2**self.chunk
            s = self.raw
            self.n_frames = floor(len(s)/float(chunk_size))
            if(self.n_frames < 7):
                raise SmallFileError("this file is to short for analysis")
            elif(self.n_frames <= self.max_length + 4): #usable audio is less than 2 min
                print "small"
                s = self.raw[chunk_size*2:chunk_size*(self.n_frames-2)] #get what we have remove intro
                self.n_frames -= 4
            elif (self.n_frames > self.max_length + 4):
                s = self.raw[chunk_size*2:chunk_size*(self.n_frames-2)] #
                idx = np.asarray([-chunk_size*10,chunk_size*10])+len(s)/2
                s = s[idx[0]:idx[1]]
                self.n_frames = 20
            #keep only every third chunk
            print len(s),self.n_frames
            self.processed = (s.reshape((self.n_frames,chunk_size))[::3]).flatten()
    
    def powerspectrum(self):
        '''Computes the power spectrum over our processed data
        
        SFTT over the whole time domain is computed using a hanning window.
        The fft output is used to create a spectrum over discrete timepoints
        '''
        w = np.hanning(self.window_size)
        n_iter = len(self.processed)/self.window_size*2-1
    spectrum = np.zeros((self.window_size/2+1,n_iter))
        idx = np.arange(self.window_size)
        #TODO vectorize with numpy rfftn... problem overlapping
        for i in range(n_iter):
            #X = np.fft.fft(self.processed[idx]*w,self.window_size)
            spectrum[:,i] = self.periodogram(self.processed[idx])# X[:self.window_size/2+1]
            idx += self.window_size/2
        self.processed = spectrum
        #self.processed = np.abs(spectrum/sum(w)*2)**2
    
    def periodogram(self,x):
        win = np.hanning(self.window_size)
        nfft = self.window_size
        
        U  = np.dot(win.conj().transpose(), win) # compensates for the power of the window.
        Xx = fft((x * win),nfft) # verified
        P  = Xx*np.conjugate(Xx)/U
        
        # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
        # Also, compute the corresponding freq vector & freq units.
        
        # Generate the one-sided spectrum [Power] if so wanted
        if nfft % 2 != 0:
            select = np.arange((nfft+1)/2)  # ODD
            P = P[select] # Take only [0,pi] or [0,pi)
            P[1:-1] = P[1:-1] * 2 # Only DC is a unique point and doesn't get doubled
        else:
            select = np.arange(nfft/2+1);    # EVEN
            P = P[select]         # Take only [0,pi] or [0,pi) # todo remove?
            P[1:-2] = P[1:-2] * 2
        
        P = P / (2* np.pi)
    
        return P
    
    def map_to_bark(self):
        p = self.processed
        freq_axis = float(self.freq)/((p.shape[0]-1)*2)*np.arange(0,p.shape[0])
        cb= len(AudioAnalytics.bark)
        '''
        if((np.where(bark>f/2))[0].shape[0]):
            cb = np.where(bark > f/2)[0].min() - 1
        else:
            cb = len(bark)
        '''
        #bark_center = bark_center[:cb]    
        if(self.terhardt):
            #apply outer ear model
            p = p*np.tile(np.expand_dims(self.terhardtOuterEar(freq_axis),axis=0).transpose(),(1,p.shape[1]))
        matrix = np.zeros((cb,p.shape[1]))
        barks = AudioAnalytics.bark[:]
        barks = np.insert(barks,0,0)
        for i in range(cb-1):
            matrix[i] = np.sum(p[((freq_axis >=barks[i]) & (freq_axis < barks[i+1]))], axis=0)
        self.processed = matrix
        
    def spectral_masking(self):
        n_bark_bands = self.processed.shape[0]
        CONST_spread = np.zeros((n_bark_bands,n_bark_bands))
    
        for i in range(n_bark_bands):
            CONST_spread[i,:] = 10**((15.81+7.5*((i-np.arange(n_bark_bands))+0.474)-17.5*(1+((i-np.arange(n_bark_bands))+0.474)**2)**0.5)/10)
        spread = CONST_spread[0:self.processed.shape[0],:]
        self.processed = np.dot(spread, self.processed)
    
    def map_to_decibel(self):
        '''Logarithmic decibel scale is applied'''
        self.processed[np.where(self.processed<1)] = 1.
        self.processed = 10*np.log10(self.processed)
        
    def map_to_phon(self):
        '''decibel are converted into phon which are a logarithmic unit to human loudness sensation'''
        # number of bark bands, matrix length in time dim
        n_bands = self.processed.shape[0]
        t       = self.processed.shape[1]
    
        
        # DB-TO-PHON BARK-SCALE-LIMIT TABLE
        # introducing 1 level more with level(1) being infinite
        # to avoid (levels - 1) producing errors like division by 0
        table_dim = n_bands; 
        cbv       = np.concatenate((np.tile(np.inf,(table_dim,1)), 
                                    self.loudn_bark[:,0:n_bands].transpose()),1)
        
        # the matrix 'levels' stores the correct Phon level for each datapoint
        # init lowest level = 2
        levels = np.tile(2,(n_bands,t)) 
        
        for lev in range(1,6): 
            db_thislev = np.tile(np.asarray([cbv[:,lev]]).transpose(),(1,t))
            levels[np.where(self.processed > db_thislev)] = lev + 2
        #ok here we compute indexes that match the cbv array such that when indexing the cbv array we get back an matrix with the proper dimensions
        #this indices match the upper or lower phon level according to the current value of spl in our matrix
        cbv_ind_hi = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-1]), order='F') 
        cbv_ind_lo = np.ravel_multi_index(dims=(table_dim,7), multi_index=np.array([np.tile(np.array([range(0,table_dim)]).transpose(),(1,t)), levels-2]), order='F') 
    
        #for each datapoint in our matrix a interpolation factor 0 < f < 1 is calculated
        ifac = (self.processed[:,0:t] - cbv.transpose().ravel()[cbv_ind_lo]) / (cbv.transpose().ravel()[cbv_ind_hi] - cbv.transpose().ravel()[cbv_ind_lo])
        
        ifac[np.where(levels==2)] = 1 # keeps the upper phon value;
        ifac[np.where(levels==8)] = 1 # keeps the upper phon value;
    
        #finally the interpolation is computed here
        self.processed[:,0:t] = AudioAnalytics.phons.transpose().ravel()[levels - 2] + (ifac * (AudioAnalytics.phons.transpose().ravel()[levels - 1] - AudioAnalytics.phons.transpose().ravel()[levels - 2])) # OPT: pre-calc diff    
    
    def map_to_sone(self):
        '''phon are converted to linear sone scale'''
        idx     = np.where(self.processed >= 40)
        not_idx = np.where(self.processed < 40)
        
        self.processed[idx]     =  2**((self.processed[idx]-40)/10)
        self.processed[not_idx] =  (self.processed[not_idx]/40)**2.642
    
    def terhardtOuterEar(f_bins):
        temp = -6.5 * np.array(map(lambda x:exp(x),(-0.6*(f_bins[1:]/1000. -3.3)**2)))
        w = 10.**((-3.64*(f_bins[1:]/1000.)**-0.8 + temp - 0.001*(f_bins[1:]/1000)**4)/20.)
        w = np.insert(w,0,0)
        return w
        
    def phone_curve(self, frequency):
        '''interpolates the 6 phon curves'''
        result = np.zeros(6)
        for i in range(6):
            tck = interpolate.splrep(AudioAnalytics.loudn_freq,AudioAnalytics.eq_loudness[i],s=0)
            result[i] = interpolate.splev(frequency, tck, der=0)
        return result
        
    def bark_filter_bank(self, mode='classic'):
        '''interpolate the bark filter banks'''
        self.loudn_bark = np.zeros((self.eq_loudness.shape[0],len(AudioAnalytics.bark)))
        
        i = 0
        j = 0
        
        for bsi in AudioAnalytics.bark:
            if mode == 'classic':
                #for each bark center frequency get the according column index for eq_loudness
                while j < len(AudioAnalytics.loudn_freq) and bsi > AudioAnalytics.loudn_freq[j]:
                    j +=1
                j -= 1
                
                #check if this frequency has already some matching index in our data
                if np.where(AudioAnalytics.loudn_freq == bsi)[0].size != 0:
                    self.loudn_bark[:,i] = AudioAnalytics.eq_loudness[:,np.where(AudioAnalytics.loudn_freq == bsi)][:,0,0]
                else:
                    #no data available so we do linear interpolation between two points
                    w1 = 1. / np.abs(AudioAnalytics.loudn_freq[j] - bsi)
                    w2 = 1. / np.abs(AudioAnalytics.loudn_freq[j + 1] - bsi)
                    self.loudn_bark[:,i] = (AudioAnalytics.eq_loudness[:,j]*w1 + AudioAnalytics.eq_loudness[:,j+1]*w2) / (w1 + w2) 
            elif mode == 'cubic':
                self.loudn_bark[:,i] = self.phone_curve(bsi)
                
            i += 1  

    def show_mono_waveform(self, samples):
        
        plt.figure(num=1, figsize=(15, 3.5), dpi=72, facecolor='w', edgecolor='k')
        plt.subplot(111)
        plt.ylabel('Channel 1')
                #channel_1.set_xlim(0,song_length) # todo
        plt.ylim(-32768,32768)
        plt.plot(samples)
        plt.show()
     
     
class FluctuationPattern(AudioAnalytics):
     
    def compute_fp(self):
        #n_seg = matrix.shape[1] / 2**16
        #idx = np.arange(2**16)
        n_bands = self.processed.shape[0]
        #result = np.zeros((n_seg,n_bands,61))
        #take 61 instead 60 frequency bins as next step will discard last bin
        fs_model_weights = self.fs_model_curve((11000./2**16) * np.arange(60))
        #use rfftn
        self.processed = np.c_[self.processed,np.zeros(n_bands)]                #calculate fluctuation patter for each frame
        self.processed = fs_model_weights * np.abs(rfftn(self.processed.reshape(n_bands,self.frame_size*2,-1).transpose(2,0,1), axes=(2,))[:,:,:60])
        
    def compute_modified_fp(self):
        '''
        Input is matrix of dimension : (bands(24), number of segements, frequencie_bins(61))
        '''
        #apply gradient 
        #plt.figure()
        #plt.imshow(self.processed[10],origin='lower', aspect='auto', interpolation='nearest')
        self.processed -= np.pad(self.processed[:,:,1:],((0,0),(0,0),(0,1)),mode='constant')
        self.processed += np.pad(self.processed,((0,0),(0,0),(1,0)),mode='constant')[:,:,:-1]
        self.processed = np.abs(self.processed)
        #sobel_x = np.asarray([[-1.,0.,1.], [-1.,0.,1.], [-1.,0.,1.]])
        #self.processed = np.abs(convolve2d(self.processed,sobel_x,boundary="symm",mode="same"))
        #apply gaussian filters
        spread_bands = self.spread_matrix(24,[0.5,0.25,0.1,0.25])
        spread_freq = self.spread_matrix(60,[0.5,0.25,0.1,0.25])
        self.processed = np.transpose(np.dot(self.processed.transpose(0,2,1),spread_bands.transpose()),(0,2,1))
        self.processed = np.dot(self.processed,spread_freq)
    
    def spread_matrix(self,n,w):
        S = np.identity(n)
        for i in range(len(w)-1):
            S[:-(i+1),i+1:] += w[i] * np.identity(n-i-1)
            S[i+1:,:-(i+1)] += w[i] * np.identity(n-i-1)
        S /= np.sum(S,1)
        return S
    
    def get_feature_matrix(self):
        self.compute_sone(plot=False)
        self.compute_fp()
        self.compute_modified_fp()
        return np.median(self.processed, axis=0)
        
    @staticmethod
    def plot_frame(x):
        '''
            Plots a 2-dimensional frame in modulation spectrum
        '''
        plt.figure()
        CS = plt.imshow(x,origin='lower', aspect='auto', interpolation='nearest') 
        ticks_lbl = np.arange(0,600,50)        
        ticks_loc = (np.arange(float(len(ticks_lbl))))/len(ticks_lbl)*x.shape[1]
        plt.xticks(ticks_loc,ticks_lbl)
        ticks_lbl = AudioAnalytics.bark[::4]
        ticks_loc = (np.arange(float(len(ticks_lbl))))/len(ticks_lbl)*x.shape[0]
        plt.yticks(ticks_loc,ticks_lbl)
        plt.xlabel("bpm")
        plt.ylabel("frequency scale [bark]")
        plt.colorbar(CS)
        
    #model of fluctuation strength
    fs_model = np.array([0,42,80,95,100,95,89,82,77,74,68,65])/100.
        
    def fs_model_curve(self, frequency):
        '''interpolates fs model data'''
        tck = interpolate.splrep(np.arange(len(FluctuationPattern.fs_model)),FluctuationPattern.fs_model,s=0)
        return interpolate.splev(frequency,tck,der=0)

class MFCC(AudioAnalytics):
    
    def __init__(self, audiofile, mel_bands = 40, weight=2):
        super(MFCC, self).__init__(audiofile);
        self.weight = float(weight)
        self.mel_bands = mel_bands
        self.powerspectrum()
        self.map_to_mel()
        self.calc_gradient()
        plt.figure()
        self.mfcc = dct(self.processed, type=2, norm="ortho", axis=0)[:13]
        plt.imshow(mfcc, origin="lower", aspect="auto",  interpolation="nearest")
        self.delta = dct(self.delta, type=2, norm="ortho", axis=0)[:13]
        self.processed = np.zeros(26)
        self.processed[:13] = np.mean(mfcc, axis = 1)
        self.processed[13:] = np.mean(self.delta, axis = 1)
        
    def mel_function(self, freq):
        return 1125 * np.log(1. + (freq/700.))
    
    def inv_mel_function(self, mel):
        return 700*(np.exp(mel/1125.) - 1.)
    
    def map_to_mel(self, max_freq=-1, min_freq = 0):
        if max_freq == -1:
            max_freq = self.freq/2
            
        filterbank = np.zeros((self.processed.shape[0],self.mel_bands))
        mel_min = self.mel_function(min_freq)
        mel_max = self.mel_function(max_freq)
        #do linear spacing in logarithmic mel domain and transform back to frequency domain to get logarithmic spacing
        mel_center = np.arange(mel_min,mel_max+1, (mel_max-mel_min)/(self.mel_bands+1))
        freq_center = self.inv_mel_function(mel_center)
        
        freq_bin = float(self.freq)/self.window_size*np.arange(0,self.processed.shape[0])
        for i in range(self.mel_bands):
            low = freq_center[i]
            cen = freq_center[i+1]
            hi = freq_center[i+2]
            
            lid = np.where((freq_bin >= low) &(freq_bin < cen))
            rid = np.where((freq_bin >= cen) &(freq_bin < hi))
            
            lslope = self.weight / (cen-low)
            rslope = self.weight / (hi-cen)
            
            filterbank[:,i][lid] = lslope * (freq_bin[lid]-low)
            filterbank[:,i][rid] = rslope * (hi - self.freq_bin[rid])
        self.processed = np.log10(np.dot(self.processed.transpose(),filterbank).transpose())
        
    def calc_gradient(self):
        self.delta = np.zeros(self.processed.shape)
        for i in range(self.processed.shape[0]):
            self.delta[i] = np.gradient(self.processed[i])
    
class SmallFileError(Exception):
    def __init__(self,message):
        super(SmallFileError,self).__init__(message)

#os.chdir('/Users/Alan/Documents/thesis/mri-thesis/code/music')
#m = MFCC("house.mp3")

    