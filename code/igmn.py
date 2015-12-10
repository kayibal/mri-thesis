from numpy import * 
import numpy as np
import scipy as sci
import scipy.linalg as linalg
from scipy.stats import chi2
import time 

class IGMN():
    def __init__(self, dataRange=None, delta=0.1, tau=0.1, vmin=None, spmin=None, \
                uniform=False, fullcovs=True, regularize=0):
        # configuration params
        self.dimension = dataRange.size
        self.vmin = vmin if vmin is not None else 2 * self.dimension
        self.spmin = spmin if spmin is not None else self.dimension + 1
        self.delta = delta
        self.tau = tau
        self.SIGMA = (self.delta * dataRange)**2
        self.maxDist = chi2.ppf(1 - self.tau, self.dimension)
        self.uniform = uniform
        self.fullcovs = fullcovs 
        self.regVal = regularize
        # components params
        self.priors = []
        self.means = []
        self.covs = []
        self.sps = []
        self.vs = []
        self.nc = 0
        # components outputs
        self.loglikes = []
        self.posts = []

        # Mahalanobis distance
        self.mahalaD = []
        
        # model likelihood
        self.dataLikelihood = 0

    def logmvnpdf(self, x, i=None, mean=None, cov=None):
        '''Logarithmic Multivariate Normal Probability Density Function
        Calculates the logarithmic PDF of a given Gaussian component
        
        Parameters
        ----------
        x: array_like
            Datapoint for which the density will be calculated
        i: integer, optional
            index of the component to use, default is None
        mean: array_like, optional
            if no index is specified the mean to be used, default is to None
        cov: array_like, optional
            if no index was specified the covariance matrix to be used, default is None
        
        Returns
        -------
        mahalaD: float
            Mahalonis Distance
        loglike: float
            Loglikelihood
        '''
        n_dim = x.size
        mean = self.means[i] if i is not None else mean
        cov = self.covs[i] if i is not None else cov

        if self.fullcovs:
            L = linalg.cholesky(cov, lower=False)
            logDetCov = 2 * np.sum(np.log(np.diagonal(L)))
        else:
            L = np.sqrt(cov)
            logDetCov = np.sum(np.log(cov))

        xcentered = x - mean
        if self.fullcovs:
            xRinv = linalg.solve(L.T, xcentered)
        else:
            xRinv = np.dot(xcentered, (1. / L))
        
       # print 'xRinv: ', xRinv

        mahalaD = np.sum(xRinv**2) 
        loglike = -0.5 * (mahalaD + logDetCov + n_dim * np.log(2 * np.pi))

        return loglike, mahalaD
    
    def computeLikelihood(self, x):
        '''Compute log likelihood
        computes the likelihood for a single datapoint over all components and updates corresponding attributes
        
        Parameters
        ----------
        x: array_like
            a multidimenional vector representing a datapoint
        '''
        self.loglikes = []
        self.mahalaD = []
        for i in range(self.nc):
            loglike, mahalaD = self.logmvnpdf(x, i)
            self.loglikes.append(loglike)
            self.mahalaD.append(mahalaD)

    def computePosterior(self):
        post = self.loglikes + np.log(self.priors)
        maxll = np.max(post)
        post = np.exp(post - maxll)
        density = np.sum(post)
        self.posts = post / density
        self.dataLikelihood += np.log(density) + maxll
          
    def hasAcceptableDistribution(self):
        for i in range(self.nc):
            if self.mahalaD[i] <= self.maxDist:
                return True
        return False

    def updatePriors(self):
        if not self.uniform:
            spSum = float(np.sum(self.sps))
            self.priors = [self.sps[i] / spSum for i in range(self.nc)]
        else:
            self.priors = [1./self.nc] * self.nc

    def updateComponents(self, x):
        for i in range(self.nc):
            self.vs[i] += 1
            self.sps[i] += self.posts[i]
            
            w = self.posts[i] / self.sps[i]

            xcentered = x - self.means[i]
            deltaMU = w * xcentered
            self.means[i] = self.means[i] + deltaMU 

            xcenteredNEW = x - self.means[i]    
            if self.fullcovs:
                self.covs[i] = self.covs[i] - np.outer(deltaMU, deltaMU) + \
                    w * (np.outer(xcenteredNEW, xcenteredNEW) - self.covs[i])
                #floor to small values to avoid overfitting
                diag = np.diag(self.covs[i]).copy()
                probl_idxs = np.where(diag < self.regVal)
                diag[probl_idxs] = self.regVal
                np.fill_diagonal(self.covs[i],diag)
            else: 
                self.covs[i] = self.covs[i] - deltaMU**2 + w * (xcenteredNEW**2 - self.covs[i]) + self.regVal

        # normalize the priors
        self.updatePriors()

    def createComponent(self, x):
        self.nc += 1 
        self.priors.append(1.)
        self.sps.append(1.)
        self.vs.append(0)
        self.means.append(x) 
        if self.fullcovs == True:
            self.covs.append(np.diagflat(self.SIGMA))
        else:
            self.covs.append(self.SIGMA)
        self.updatePriors()
        loglike, mahalaD = self.logmvnpdf(x, self.nc-1)
        self.loglikes.append(loglike)
        self.mahalaD.append(mahalaD) 
    
    def removeSpurious(self):
        for i in reversed(range(self.nc)):
            if self.vs[i] > self.vmin and self.sps[i] < self.spmin:
                self.nc -= 1
                del self.vs[i]
                del self.sps[i]
                del self.priors[i]
                del self.means[i]
                del self.covs[i]
               
    def train(self, X):
        N = X.shape[0]
        for i in range(N):
            if (i > 0 and i % 100 == 0): print 'processing data %d/%d -- num comps: %d' %(i,N,self.nc)

            x = X[i,:]

            self.computeLikelihood(x)
            if not self.hasAcceptableDistribution():
                self.createComponent(x)
            self.computePosterior()
            self.updateComponents(x)
            self.removeSpurious()

    def recall(self, X):
        N, alpha = X.shape
        beta = self.dimension - alpha

        B = sci.zeros((N, self.dimension - alpha))
        for j in range(N):
            if (j > 0 and j % 100 == 0): print 'processing data %d/%d' %(j,N)
            
            x = X[j,:]
            
            posts = sci.zeros((self.nc,1))
            xm = sci.zeros((self.nc, beta))

            for i in range(self.nc):
                meanA = self.means[i][0:alpha]
                meanB = self.means[i][alpha:alpha+beta]
                
                if self.fullcovs:
                    covA = self.covs[i][0:alpha,0:alpha]
                    covBA = self.covs[i][alpha:alpha+beta,0:alpha]
                   
                    xm[i,:] = meanB + np.dot(covBA, linalg.solve(covA,  x - meanA))
                else:
                    covA = self.covs[0:alpha] 
                    xm[i,:] = meanB
                
                loglike = self.logmvnpdf(x, mean=meanA, cov=covA)[0]
                posts[i] = loglike + np.log(self.priors[i])
            
            maxll = np.max(posts)
            posts = np.exp(posts - maxll)
            posts = posts / np.sum(posts)
            
            B[j,:] = np.dot(posts.T, xm)
        return B   
    
    def get_clusters(self, data, labels = None):
        clusters = []
        for inp in data:
            self.computeLikelihood(inp)
            self.computePosterior()
            idx = np.argmax(self.posts)
            if labels:
                clusters.append(labels[idx])
            else:
                clusters.append(idx)
        return clusters

    def get_song_rep(self, data):
        posts = []
        for inp in data:
            self.computeLikelihood(inp)
            self.computePosterior()
            posts.append(self.posts)
        posts = np.array(posts)
        return posts

    def reset(self):
        self.nc = 0
        self.priors = []
        self.means = []
        self.covs = []
        self.sps = []
        self.vs = []

def leaveoneout(model, data, labels):
    from sklearn.cross_validation import LeaveOneOut
    import copy
    n,d = data.shape
 
    loo = LeaveOneOut(n)
    confTable = np.zeros((3,3))
    for trainIdx, testIdx in loo:
       # train model
        modelc = copy.copy(model)
        modelc.train(data[trainIdx,:])

       # test model
        out = modelc.recall(data[testIdx,:4])

        predictedClass = np.argmax(out,axis=1)
 
        confTable[labels[testIdx], predictedClass] += 1
    print confTable



if __name__ == '__main__':
    data = np.loadtxt(open("iris.data","rb"),delimiter=" ")
    dataRange = np.max(data,axis=0) - np.min(data,axis=0)
   
    m = IGMN(dataRange=dataRange, tau=0.1, delta=0.45, spmin=2, vmin=3)
    
    labels = np.argmax(data[:,4:], axis=1)
    startTime = time.time()
    leaveoneout(m, data, labels)   
    elapsedTime = time.time() - startTime
    print 'Elapsed Time:', elapsedTime