#!/usr/bin/env python

import math
import numpy
import pickle
import sklearn
import sys

from stf import STF
from mfcc import MFCC
from dtw import DTW
from gmmmap import GMMMap, TrajectoryGMMMap

DIMENSION = 16

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: %s [list of source stf] [list of target stf] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    assert len(source_list) == len(target_list)

    learn_data = None
    square_mean = numpy.zeros(DIMENSION)
    mean = numpy.zeros(DIMENSION)

    for i in xrange(len(source_list)):
        target = STF()
        target.loadfile(target_list[i])

        mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = DIMENSION)
        target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) for frame in xrange(target.SPEC.shape[0])])
        target_data = numpy.hstack([target_mfcc, mfcc.delta(target_mfcc)])

        source = STF()
        source.loadfile(source_list[i])

        mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency)
        source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])
    
        dtw = DTW(source_mfcc, target_mfcc, window = abs(source.SPEC.shape[0] - target.SPEC.shape[0]) * 2)
        warp_mfcc = dtw.align(source_mfcc)
        warp_data = numpy.hstack([warp_mfcc, mfcc.delta(warp_mfcc)])

        data = numpy.hstack([warp_data, target_data])
        if learn_data is None:
            learn_data = data
        else:
            learn_data = numpy.vstack([learn_data, data])

        square_mean = (square_mean * (learn_data.shape[0] - target_mfcc.shape[0]) + (target_mfcc ** 2).sum(axis = 0)) / learn_data.shape[0]
        mean = (mean * (learn_data.shape[0] - target_mfcc.shape[0]) + target_mfcc.sum(axis = 0)) / learn_data.shape[0]

    gmm = sklearn.mixture.GMM(n_components = 2, covariance_type = 'full')
    gmm.fit(learn_data)

    gv = square_mean - mean ** 2
    gv_gmm = sklearn.mixture.GMM(covariance_type = 'full')
    gv_gmm.fit(gv)

    gmmmap = (TrajectoryGMMMap(gmm, learn_data.shape[0], gv_gmm), TrajectoryGMMMap(gmm, learn_data.shape[0], gv_gmm, swap = True))

    output = open(sys.argv[3], 'wb')
    pickle.dump(gmmmap, output)
    output.close()
