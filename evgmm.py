#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GMM

import scipy.sparse
import scipy.sparse.linalg

M = 32

class EVGMM(object):
    def __init__(self, learn_data, swap = False):
        S = len(learn_data)
        D = learn_data[0].shape[1] / 2

        initial_gmm = GMM(n_components = M, covariance_type = 'full')
        initial_gmm.fit(np.vstack(learn_data))

        self.weights = initial_gmm.weights_
        self.source_means = initial_gmm.means_[:, :D]
        self.target_means = initial_gmm.means_[:, D:]
        self.covarXX = initial_gmm.covars_[:, :D, :D]
        self.covarXY = initial_gmm.covars_[:, :D, D:]
        self.covarYX = initial_gmm.covars_[:, D:, :D]
        self.covarYY = initial_gmm.covars_[:, D:, D:]

        sv = []

        for i in xrange(S):
            gmm = GMM(n_components = M, params = 'm', init_params = '', covariance_type = 'full')
            gmm.weights_ = initial_gmm.weights_
            gmm.means_ = initial_gmm.means_
            gmm.covars_ = initial_gmm.covars_
            gmm.fit(learn_data[i])

            sv.append(gmm.means_)

        sv = np.array(sv)

        source_pca = PCA()
        source_pca.fit(sv[:, :, :D].reshape((S, M * D)))

        target_pca = PCA()
        target_pca.fit(sv[:, :, D:].reshape((S, M * D)))

        self.eigenvectors = source_pca.components_.reshape((M, D, S)), target_pca.components_.reshape((M, D, S))
        self.biasvectors = source_pca.mean_.reshape((M, D)), target_pca.mean_.reshape((M, D))

        self.fitted_source = self.source_means
        self.fitted_target = self.target_means

        self.swap = swap

    def fit(self, data, epoch = 1000):
        if self.swap:
            self.fit_source(data, epoch)
        else:
            self.fit_target(data, epoch)

    def fit_source(self, source, epoch):
        px = GMM(n_components = M, covariance_type = 'full')
        px.weights_ = self.weights
        px.means_ = self.source_means
        px.covars_ = self.covarXX

        for x in xrange(epoch):
            predict = px.predict_proba(np.atleast_2d(source))
            x = np.sum([predict[:, i: i + 1] * (source - self.biasvectors[0][i]) for i in xrange(M)], axis = 1)
            gamma = np.sum(predict, axis = 0)

            left = np.sum([gamma[i] * np.dot(self.eigenvectors[0][i].T, np.linalg.solve(px.covars_, self.eigenvectors[0])[i]) for i in xrange(M)], axis = 0)
            right = np.sum([np.dot(self.eigenvectors[0][i].T, np.linalg.solve(px.covars_, x)[i]) for i in xrange(M)], axis = 0)
            weight = np.linalg.solve(left, right)

            self.fitted_source = np.dot(self.eigenvectors[0], weight) + self.biasvectors[0]
            px.means_ = self.fitted_source

    def fit_target(self, target, epoch):
        py = GMM(n_components = M, covariance_type = 'full')
        py.weights_ = self.weights
        py.means_ = self.target_means
        py.covars_ = self.covarYY

        for x in xrange(epoch):
            predict = py.predict_proba(np.atleast_2d(target))
            y = np.sum([predict[:, i: i + 1] * (target - self.biasvectors[1][i]) for i in xrange(M)], axis = 1)
            gamma = np.sum(predict, axis = 0)

            left = np.sum([gamma[i] * np.dot(self.eigenvectors[1][i].T, np.linalg.solve(py.covars_, self.eigenvectors[1])[i]) for i in xrange(M)], axis = 0)
            right = np.sum([np.dot(self.eigenvectors[1][i].T, np.linalg.solve(py.covars_, y)[i]) for i in xrange(M)], axis = 0)
            weight = np.linalg.solve(left, right)

            self.fitted_target = np.dot(self.eigenvectors[1], weight) + self.biasvectors[1]
            py.means_ = self.fitted_target

    def convert(self, source):
        D = source.shape[0]

        E = np.zeros((M, D))
        for m in xrange(M):
            xx = np.linalg.solve(self.covarXX[m], source - self.fitted_source[m])
            E[m] = self.fitted_target[m] + np.dot(self.covarYX[m], xx)

        px = GMM(n_components = M, covariance_type = 'full')
        px.weights_ = self.weights
        px.means_ = self.source_means
        px.covars_ = self.covarXX

        posterior = px.predict_proba(np.atleast_2d(source))
        return np.dot(posterior, E)

class TrajectoryEVGMM(EVGMM):
    def __init__(self, learn_data, swap = False, gv = None):
        super(TrajectoryEVGMM, self).__init__(learn_data, swap)

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

        px = GMM(n_components = M, covariance_type = 'full')
        px.weights_ = self.weights
        px.means_ = self.source_means
        px.covars_ = self.covarXX

        optimum_mix = px.predict(source)

        E = np.zeros((T, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            xx = np.linalg.solve(self.covarXX[m], source[t] - self.fitted_source[m])
            E[t] = self.fitted_target[m] + np.dot(self.covarYX[m], xx)
        E = E.flatten()

        D_ = np.zeros((T, D * 2, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            D_[t] = np.linalg.inv(self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy))
        D_ = scipy.sparse.block_diag(D_, format = 'csr')

        mutual = W.T.dot(D_)
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
