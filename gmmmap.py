#!/usr/bin/python
# coding: utf-8

import numpy as np
from numpy import linalg
from sklearn.mixture import GMM
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

class GMMMap(object):
    """GMM-based frame-by-frame speech parameter mapping. 

    GMMMap represents a class to transform spectral features of a source
    speaker to that of a target speaker based on Gaussian Mixture Models
    of source and target joint spectral features.
    
    Notation
    --------
    Source speaker's feature: X = {x_t}, 0 <= t < T
    Target speaker's feature: Y = {y_t}, 0 <= t < T
    where T is the number of time frames.

    Parameters
    ----------
    gmm : scipy.mixture.GMM
        Gaussian Mixture Models of source and target joint features
    
    swap : bool
        True: source -> target
        False target -> source
    
    Attributes
    ----------
    num_mixtures : int
        the number of Gaussian mixtures

    weights : array, shape (`num_mixtures`)
        weights for each gaussian

    src_means : array, shape (`num_mixtures`, `order of spectral feature`)
        means of GMM for a source speaker

    tgt_means : array, shape (`num_mixtures`, `order of spectral feature`)
        means of GMM for a target speaker

    covarXX : array, shape (`num_mixtures`, `order of spectral feature`, 
        `order of spectral feature`)
        variance matrix of source speaker's spectral feature

    covarXY : array, shape (`num_mixtures`, `order of spectral feature`, 
        `order of spectral feature`)
        covariance matrix of source and target speaker's spectral feature

    covarYX : array, shape (`num_mixtures`, `order of spectral feature`, 
        `order of spectral feature`)
        covariance matrix of target and source speaker's spectral feature

    covarYY : array, shape (`num_mixtures`, `order of spectral feature`, 
        `order of spectral feature`)
        variance matrix of target speaker's spectral feature
    
    D : array, shape (`num_mixtures`, `order of spectral feature`, 
        `order of spectral feature`)
        covariance matrices of target static spectral features

    px : scipy.mixture.GMM
        Gaussian Mixture Models of source speaker's features

    Reference
    ---------
      - [Toda 2007] Voice Conversion Based on Maximum Likelihood Estimation
        of Spectral Parameter Trajectory.
        http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf

    """
    def __init__(self, gmm, swap=False):
        # D is the order of spectral feature for a speaker
        self.num_mixtures, D = gmm.means_.shape[0], gmm.means_.shape[1]/2
        self.weights = gmm.weights_

        # Split source and target parameters from joint GMM
        self.src_means = gmm.means_[:, 0:D]
        self.tgt_means = gmm.means_[:, D:]
        self.covarXX = gmm.covars_[:, :D, :D]
        self.covarXY = gmm.covars_[:, :D, D:]
        self.covarYX = gmm.covars_[:, D:, :D]
        self.covarYY = gmm.covars_[:, D:, D:]

        # swap src and target parameters
        if swap:
            self.tgt_means, self.src_means = self.src_means, self.tgt_means
            self.covarYY, self.covarXX = self.covarXX, self.covarYY
            self.covarYX, self.covarXY = self.covarXY, self.covarYX

        # Compute D eq.(12) in [Toda 2007]
        self.D = np.zeros(self.num_mixtures * D * D).reshape(self.num_mixtures, D, D)
        for m in range(self.num_mixtures):
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            self.D[m] = self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy)

        # p(x), which is used to compute posterior prob. for a given source
        # spectral feature in mapping stage.
        self.px = GMM(n_components = self.num_mixtures, covariance_type = "full")
        self.px.means_ = self.src_means
        self.px.covars_ = self.covarXX
        self.px.weights_ = self.weights

    def convert(self, src):
        """
        Mapping source spectral feature x to target spectral feature y 
        so that minimize the mean least squared error.
        More specifically, it returns the value E(p(y|x)].

        Parameters
        ----------
        src : array, shape (`order of spectral feature`)
            source speaker's spectral feature that will be transformed

        Return
        ------
        converted spectral feature
        """
        D = len(src)

        # Eq.(11)
        E = np.zeros((self.num_mixtures, D))
        for m in range(self.num_mixtures):
            xx = np.linalg.solve(self.covarXX[m], src - self.src_means[m])
            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx.transpose())
                
        # Eq.(9) p(m|x)
        posterior = self.px.predict_proba(np.atleast_2d(src))

        # Eq.(13) conditinal mean E[p(y|x)]
        return posterior.dot(E)
            
class TrajectoryGMMMap(GMMMap):
    """
    Trajectory-based speech parameter mapping for voice conversion
    based on the maximum likelihood criterion.

    Parameters
    ----------
    gmm : scipy.mixture.GMM
        Gaussian Mixture Models of source and target speaker joint features

    gv : scipy.mixture.GMM (default=None)
        Gaussian Mixture Models of target speaker's global variance of spectral
        feature
    
    swap : bool (default=False)
        True: source -> target
        False target -> source

    Attributes
    ----------
    TODO 

    Reference
    ---------
      - [Toda 2007] Voice Conversion Based on Maximum Likelihood Estimation
        of Spectral Parameter Trajectory.
        http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf
    """
    def __init__(self, gmm, T, gv = None, epoch = 100, step = 1e-5, swap = False):
        super(TrajectoryGMMMap, self).__init__(gmm, swap)

        # shape[1] = d(src) + d(src_delta) + d(tgt) + d(tgt_delta)
        D = gmm.means_.shape[1] / 4

        ## Setup for GV post-filtering
        # It is assumed that GV is modeled as a single mixture GMM
        self.gv = gv
        if gv != None:
            self.gv_mean = gv.means_[0]
            self.gv_covar = gv.covars_[0]
            self.Pv = np.linalg.inv(self.gv_covar)

            self.epoch = epoch
            self.step = step

    def __construct_weight_matrix(self, T, D):
        # Construct Weight matrix W
        # Eq.(25) ~ (28)
        for t in range(T):
            w0 = scipy.sparse.lil_matrix((D, D * T))
            w1 = scipy.sparse.lil_matrix((D, D * T))
            w0[0:, t * D: (t + 1) * D] = scipy.sparse.diags(np.ones(D), 0)

            if t - 1 >= 0:
                tmp = np.zeros(D)
                tmp.fill(-0.5)
                w1[0:, (t - 1) * D: t * D] = scipy.sparse.diags(tmp, 0)
            if t + 1 < T:
                tmp = np.zeros(D)
                tmp.fill(0.5)
                w1[0:, (t + 1) * D: (t + 2) * D] = scipy.sparse.diags(tmp, 0)

            W_t = scipy.sparse.vstack([w0, w1])

            # Slower
            # self.W[2 * D * t: 2 * D * (t + 1), :] = W_t

            if t == 0:
                self.W = W_t
            else:
                self.W = scipy.sparse.vstack([self.W, W_t])

        self.W = scipy.sparse.csr_matrix(self.W)

        assert self.W.shape == (D * T * 2, D * T)

    # gvgrad computes gradient of the likelihood with regard to GV.
    def __gvgrad(self, y):
        D, T = y.shape

        gv = y.var(1)
        mean = y.mean(1)

        v = np.zeros((T, D))
        for t in range(T):
            # Eq.(55)
            v[t] = -2.0 / T * self.Pv.dot(gv - self.gv_mean).dot(y[:, t] - mean)

        return v.reshape(D, T)
        
    def convert(self, src):
        """
        Mapping source spectral feature x to target spectral feature y 
        so that maximize the likelihood of y given x.

        Parameters
        ----------
        src : array, shape (`the number of frames`, `the order of spectral feature`)
            a sequence of source speaker's spectral feature that will be
            transformed

        Return
        ------
        a sequence of transformed spectral features
        """
        T, D = src.shape[0], src.shape[1] / 2
        self.__construct_weight_matrix(T, D)

        # A suboptimum mixture sequence  (eq.37)
        optimum_mix = self.px.predict(src)

        # Compute E eq.(40)
        self.E = np.zeros((T, D * 2))
        for t in range(T):
            m = optimum_mix[t] # estimated mixture index at time t
            xx = np.linalg.solve(self.covarXX[m], src[t] - self.src_means[m])
            # Eq. (22)
            self.E[t] = self.tgt_means[m] + np.dot(self.covarYX[m], xx)
        self.E = self.E.flatten()

        # Compute D eq.(41). Note that self.D represents D^-1.
        self.D = np.zeros((T, D * 2, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            # Eq. (23)
            self.D[t] = self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy)
            self.D[t] = np.linalg.inv(self.D[t])
        self.D = scipy.sparse.block_diag(self.D, format = 'csr')

        # Compute target static features
        # eq.(39)
        mutual = self.W.T.dot(self.D)
        covar = mutual.dot(self.W)
        mean = mutual.dot(self.E)
        y = scipy.sparse.linalg.spsolve(covar, mean, use_umfpack = False)

        if self.gv:
            y = y.reshape((T, D))

            # Better initial value based on eq. (58)
            y_gv = np.zeros((T, D))
            for t in range(T):
                y_gv[t] = np.sqrt(self.gv_mean / y.var(0)) * (y[t, :] - y.mean(0)) + y.mean(0)
            y = y_gv.transpose()

            omega = 1.0 / (T * 2)

            # update y based on gradient decent
            for epoch in range(self.epoch):
                y_delta = omega * (-covar.dot(y.flatten()) + mean.flatten()) + self.__gvgrad(y).flatten()

                # Eq. (52)
                y += self.step * y_delta.reshape((D, T))

            y = y.transpose()

        return y.reshape((T, D))
