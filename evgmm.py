#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GMM

import scipy.sparse
import scipy.sparse.linalg

M = 32

class EVGMM(object):
    def __init__(self, learn_data):
        S = len(learn_data)
        D = learn_data[0].shape[1] / 2

        initial_gmm = GMM(n_components = M, covariance_type = 'full')
        initial_gmm.fit(np.vstack(learn_data))

        initial_gmm = initial_gmm

        self.weights = initial_gmm.weights_
        self.src_means = initial_gmm.means_[:, :D]
        self.tgt_means = initial_gmm.means_[:, D:]
        self.covarXX = initial_gmm.covars_[:, :D, :D]
        self.covarXY = initial_gmm.covars_[:, :D, D:]
        self.covarYX = initial_gmm.covars_[:, D:, :D]
        self.covarYY = initial_gmm.covars_[:, D:, D:]

        sv = None

        for i in xrange(S):
            gmm = GMM(n_components = M, params = 'm', init_params = 'm', covariance_type = 'full')
            gmm.weights_ = initial_gmm.weights_
            gmm.means_ = initial_gmm.means_
            gmm.covars_ = initial_gmm.covars_
            gmm.fit(learn_data[i])

            if sv is None:
                sv = gmm.means_[:, :D].flatten()
            else:
                sv = np.vstack([sv, gmm.means_[:, :D].flatten()])

        pca = PCA()
        pca.fit(sv)

        self.eigenvectors = pca.components_.reshape((M, D, S))
        self.biasvectors = pca.mean_.reshape((M, D))
        self.means = None

    def fit(self, target, epoch = 100):
        py = GMM(n_components = M, covariance_type = 'full')
        py.weights_ = self.weights
        py.means_ = self.tgt_means
        py.covars_ = self.covarYY

        for x in xrange(epoch):
            predict = py.predict_proba(np.atleast_2d(target))
            y = np.sum([predict[:, i: i + 1] * (target - self.biasvectors[i]) for i in xrange(M)], axis = 1)
            gamma = np.sum(predict, axis = 0)

            left = np.sum([gamma[i] * np.dot(self.eigenvectors[i].T, np.linalg.solve(py.covars_, self.eigenvectors)[i]) for i in xrange(M)], axis = 0)
            right = np.sum([np.dot(self.eigenvectors[i].T, np.linalg.solve(py.covars_, y)[i]) for i in xrange(M)], axis = 0)
            omega = np.linalg.solve(left, right)

            self.fit_means = np.dot(self.eigenvectors, omega) + self.biasvectors
            py.means_ = self.fit_means

    def convert(self, source):
        D = source.shape[0]

        E = np.zeros((M, D))
        for m in xrange(M):
            xx = np.linalg.solve(self.covarXX[m], source - self.src_means[m])
            E[m] = self.fit_means[m] + np.dot(self.covarYX[m], xx)

        px = GMM(n_components = M, covariance_type = 'full')
        px.weights_ = self.weights
        px.means_ = self.src_means
        px.covars_ = self.covarXX

        posterior = px.predict_proba(np.atleast_2d(source))
        return np.dot(posterior, E)

class TrajectoryEVGMM(EVGMM):
    def __init__(self, learn_data, gv = None):
        super(TrajectoryEVGMM, self).__init__(learn_data)

        self.gv = gv
        if gv != None:
            self.gv_mean = gv.means_[0]
            self.gv_covar = gv.covars_[0]
            self.Pv = np.linalg.inv(self.gv_covar)

    def __construct_weight_matrix(self, T, D):
        for t in xrange(T):
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
            if t == 0:
                W = W_t
            else:
                W = scipy.sparse.vstack([W, W_t])

        return scipy.sparse.csr_matrix(W)

    def __gvgrad(self, y):
        D, T = y.shape

        gv = y.var(1)
        mean = y.mean(1)

        v = np.zeros((T, D))
        for t in range(T):
            v[t] = -2.0 / T * self.Pv.dot(gv - self.gv_mean).dot(y[:, t] - mean)
        return v.reshape(D, T)
        
    def convert(self, source, epoch = 100, step = 1e-5):
        T, D = source.shape[0], source.shape[1] / 2
        W = self.__construct_weight_matrix(T, D)

        optimum_mix = self.px.predict(source)

        E = np.zeros((T, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            xx = np.linalg.solve(self.covarXX[m], source[t] - self.src_means[m])
            E[t] = self.fit_means[m] + np.dot(self.covarYX[m], xx)
        E = E.flatten()

        D_ = np.zeros((T, D * 2, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            D_[t] = np.linalg.inv(self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy))
        D_ = scipy.sparse.block_diag(D_, format = 'csr')

        mutual = self.W.T.dot(D_)
        covar = mutual.dot(W)
        mean = mutual.dot(E)
        y = scipy.sparse.linalg.spsolve(covar, mean, use_umfpack = False)

        if self.gv:
            y = y.reshape((T, D))

            y_gv = np.zeros((T, D))
            for t in range(T):
                y_gv[t] = np.sqrt(self.gv_mean / y.var(0)) * (y[t, :] - y.mean(0)) + y.mean(0)
            y = y_gv.transpose()

            omega = 1.0 / (T * 2)
            for e in range(epoch):
                y_delta = omega * (-covar.dot(y.flatten()) + mean.flatten()) + self.__gvgrad(y).flatten()
                y += step * y_delta.reshape((D, T))
            y = y.transpose()

        return y.reshape((T, D))
