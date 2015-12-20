#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GMM

import scipy.sparse
import scipy.sparse.linalg

from gmmmap import GMMMap
from trajectory import TrajectoryGMMMap

M = 32

class MRGMM(GMMMap):
    @profile
    def __init__(self, learn_data, scores, epoch = 1000):
        S, T, D = learn_data.shape
        D /= 2
        K = scores.shape[1]

        initial_gmm = GMM(n_components = M, covariance_type = 'full')
        initial_gmm.fit(np.vstack(learn_data))

        gmms = []

        for i in xrange(S):
            gmm = GMM(n_components = M, params = 'm', init_params = '', covariance_type = 'full')
            gmm.weights_ = initial_gmm.weights_
            gmm.means_ = initial_gmm.means_
            gmm.covars_ = initial_gmm.covars_
            gmm.fit(learn_data[i])

            gmms.append(gmm)

        w = np.array([self.__construct_score_matrix(D, scores[s]) for s in xrange(S)])

        z_t = np.zeros((S, T, 2 * D, 2 * D))
        for s in xrange(S):
            for t in xrange(T):
                features = learn_data[s, t, np.newaxis]
                z_t[s, t] = features.T.dot(features)

        for x in xrange(epoch):
            gamma = np.zeros((S, M))
            z = np.zeros((S, M, 2 * D))
            covars = np.zeros((M, 2 * D, 2 * D))

            for s in xrange(S):
                predict = gmms[s].predict_proba(learn_data[s])

                gamma[s] = predict.sum(axis = 0)
                z[s] = predict.T.dot(learn_data[s])

                for m in xrange(M):
                    for t in xrange(T):
                        covars[m] += predict[t, m] * z_t[s, t]

                    means = gmms[s].means_
                    covars[m] += gamma[s, m] * means.T.dot(means)

                    means = means[m, np.newaxis]
                    z_m = z[s, m, np.newaxis]
                    covars[m] -= means.T.dot(z_m) + z_m.T.dot(means)

            self.weights = gamma.sum(axis = 0) / gamma.sum()
            self.v = np.zeros((M, (K + 2) * D))
            self.covars = np.array([covars[m] / gamma.sum(axis = 0)[m] for m in xrange(M)])

            for m in xrange(M):
                left = np.zeros(((K + 2) * D, (K + 2) * D))
                right = np.zeros(((K + 2) * D,))

                for s in xrange(S):
                    left += gamma[s, m] * w[s].T.dot(np.linalg.solve(self.covars[m], w[s]))
                    right += w[s].T.dot(np.linalg.solve(self.covars[m], z[s, m]))

                self.v[m] = np.linalg.solve(left, right)

            for s in xrange(S):
                gmms[s].weights_ = self.weights
                gmms[s].means_ = np.array([w[s].dot(self.v[m]) for m in xrange(M)])
                gmms[s].covars_ = self.covars

            print x

    def __construct_score_matrix(self, D, score):
        K = score.shape[0]
        w = np.zeros((2 * D, (K + 2) * D))

        w[:D, :D] = np.identity(D)
        w[D:, D: 2 * D] = np.identity(D)
        for k in xrange(K):
            w[D:, (k + 2) * D: (k + 3) * D] = score[k] * np.identity(D)

        return w

    def convert(self, source, score = None):
        D = source.shape[0]

        if score:
            w = self.__construct_score_matrix(D, score)

            gmm = GMM(n_components = M, covariance_type = 'full')
            gmm.weights_ = self.weights
            gmm.means_ = np.array([w.dot(self.v[m]) for m in xrange(M)])
            gmm.covars_ = self.covars

            super(GMMMap, self).__init__(gmm)

        return super(GMMMap, self).convert(source)

class TrajectoryMRGMM(MRGMM, TrajectoryGMMMap):
    def __init__(self, learn_data, scores, epoch = 1000):
        super(MRGMM, self).__init__(learn_data, scores, epoch)

    def convert(self, source, score = None, *args, **kwargs):
        D = source.shape[0]

        if score:
            w = self.__construct_score_matrix(D, score)

            gmm = GMM(n_components = M, covariance_type = 'full')
            gmm.weights_ = self.weights
            gmm.means_ = np.array([w.dot(self.v[m]) for m in xrange(M)])
            gmm.covars_ = self.covars

            super(TrajectoryGMMMap, self).__init__(gmm, *args, **kwargs)

        return super(TrajectoryGMMMap, self).convert(source)
