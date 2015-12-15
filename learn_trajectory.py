#!/usr/bin/env python

import math
import numpy
import os
import pickle
import re
import sklearn
import sklearn.mixture
import sys

from trajectory import TrajectoryGMMMap

from stf import STF
from mfcc import MFCC
from dtw import DTW

DIMENSION = 16
K = 32

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: %s [list of source stf] [list of target stf] [dtw cache directory] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    assert len(source_list) == len(target_list)

    learn_data = None

    for i in xrange(len(source_list)):
        print i

        source = STF()
        source.loadfile(source_list[i])

        mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = DIMENSION)
        source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])

        target = STF()
        target.loadfile(target_list[i])

        mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = DIMENSION)
        target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) for frame in xrange(target.SPEC.shape[0])])
    
        cache_path = os.path.join(sys.argv[3], '%s_%s.dtw' % tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), [source_list[i], target_list[i]])))
        if os.path.exists(cache_path):
            dtw = pickle.load(open(cache_path))
        else:
            dtw = DTW(source_mfcc, target_mfcc, window = abs(source.SPEC.shape[0] - target.SPEC.shape[0]) * 2)
            with open(cache_path, 'wb') as output:
                pickle.dump(dtw, output)

        warp_mfcc = dtw.align(source_mfcc)

        warp_data = numpy.hstack([warp_mfcc, mfcc.delta(warp_mfcc)])
        target_data = numpy.hstack([target_mfcc, mfcc.delta(target_mfcc)])

        data = numpy.hstack([warp_data, target_data])
        if learn_data is None:
            learn_data = data
        else:
            learn_data = numpy.vstack([learn_data, data])

    gmm = sklearn.mixture.GMM(n_components = K, covariance_type = 'full')
    gmm.fit(learn_data)

    gmmmap = TrajectoryGMMMap(gmm)

    with open(sys.argv[4], 'wb') as output:
        pickle.dump(gmmmap, output)
