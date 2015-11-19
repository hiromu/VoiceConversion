#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GMM

M = 32

class EVGMM(object):
    def __init__(self, learn_data):
        self.S = len(learn_data)
        self.D = learn_data[0].shape[1] / 2

        initial_gmm = GMM(n_components = M, covariance_type = 'full')
        initial_gmm.fit(np.vstack(learn_data))

        initial_gmm = initial_gmm

        self.weights = initial_gmm.weights_
        self.src_means = initial_gmm.means_[:, :self.D]
        self.tgt_means = initial_gmm.means_[:, self.D:]
        self.covarXX = initial_gmm.covars_[:, :self.D, :self.D]
        self.covarXY = initial_gmm.covars_[:, :self.D, self.D:]
        self.covarYX = initial_gmm.covars_[:, self.D:, :self.D]
        self.covarYY = initial_gmm.covars_[:, self.D:, self.D:]

        sv = None

        for i in xrange(self.S):
            gmm = GMM(n_components = M, params = 'm', init_params = 'm', covariance_type = 'full')
            gmm.weights_ = initial_gmm.weights_
            gmm.means_ = initial_gmm.means_
            gmm.covars_ = initial_gmm.covars_
            gmm.fit(learn_data[i])

            if sv is None:
                sv = gmm.means_[:, :self.D].flatten()
            else:
                sv = np.vstack([sv, gmm.means_[:, :self.D].flatten()])

        pca = PCA()
        pca.fit(sv)

        self.eigenvectors = pca.components_.reshape((M, self.D, S))
        self.biasvectors = pca.mean_.reshape((M, self.D))
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

            self.means = np.dot(self.eigenvectors, omega) + self.biasvectors
            py.means_ = self.means

    def convert(self, source):
        E = np.zeros((M, self.D))
        for m in xrange(M):
            xx = np.linalg.solve(self.covarXX[m], source - self.src_means[m])
            E[m] = self.means[m] + self.covarYX[m].dot(xx.transpose())

        px = GMM(n_components = M, covariance_type = 'full')
        px.weights_ = self.weights
        px.means_ = self.src_means
        px.covars_ = self.covarXX

        posterior = px.predict_proba(np.atleast_2d(source))
        return np.dot(posterior, E)
