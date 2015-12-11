#!/usr/bin/python
# coding: utf-8

import numpy as np

from sklearn.mixture import GMM

import scipy.sparse
import scipy.sparse.linalg

from gmmmap import GMMMap

class TrajectoryGMMMap(GMMMap):
    def __init__(self, gmm, T, gv = None, epoch = 100, step = 1e-5, swap = False):
        super(TrajectoryGMMMap, self).__init__(gmm, swap)

        # Compute D eq.(12) in [Toda 2007]
        D = gmm.means_.shape[1] / 2
        self.D = np.zeros(self.M * D * D).reshape(self.M, D, D)
        for m in range(self.M):
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            self.D[m] = self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy)

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
        # shape[1] = d(src) + d(src_delta) + d(tgt) + d(tgt_delta)
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
