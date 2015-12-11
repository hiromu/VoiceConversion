#!/usr/bin/python
# coding: utf-8

import numpy as np
from sklearn.mixture import GMM

class GMMMap(object):
    def __init__(self, gmm, swap = False):
        self.M, D = gmm.means_.shape[0], gmm.means_.shape[1] / 2
        self.weights = gmm.weights_

        self.src_means = gmm.means_[:, :D]
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

        # p(x), which is used to compute posterior prob. for a given source
        # spectral feature in mapping stage.
        self.px = GMM(n_components = self.M, covariance_type = "full")
        self.px.means_ = self.src_means
        self.px.covars_ = self.covarXX
        self.px.weights_ = self.weights

    def convert(self, src):
        D = len(src)

        # Eq.(11)
        E = np.zeros((self.M, D))
        for m in range(self.M):
            xx = np.linalg.solve(self.covarXX[m], src - self.src_means[m])
            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx.transpose())
                
        # Eq.(9) p(m|x)
        posterior = self.px.predict_proba(np.atleast_2d(src))

        # Eq.(13) conditinal mean E[p(y|x)]
        return posterior.dot(E)
