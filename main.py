#!/usr/bin/env python

import numpy
import sklearn
import sys

from stf import STF
from mfcc import MFCC
from dtw import DTW
from gmmmap import GMMMap, TrajectoryGMMMap

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: %s [source] [target] [output]' % sys.argv[0]
        sys.exit()

    source = STF()
    source.loadfile(sys.argv[1])
    target = STF()
    target.loadfile(sys.argv[2])

    mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency)
    target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) for frame in xrange(target.SPEC.shape[0])])
    target_delta = mfcc.delta(target_mfcc)

    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency)
    source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])

    dtw = DTW(source_mfcc, target_mfcc, window = abs(source.SPEC.shape[0] - target.SPEC.shape[0]))
    warp_mfcc = dtw.align(source_mfcc)

    target_data = numpy.hstack([target_mfcc, target_delta])
    warp_data = numpy.hstack([warp_mfcc, mfcc.delta(warp_mfcc)])
    learn_data = numpy.hstack([warp_data, target_data])

    gmm = sklearn.mixture.GMM(n_components = 2, covariance_type = 'full')
    gmm.fit(learn_data)

    gv = target_mfcc.sum(axis = 0) - target_mfcc.mean(axis = 0) ** 2
    gv_gmm = sklearn.mixture.GMM(covariance_type = 'full')
    gv_gmm.fit(gv)

    gmmmap = TrajectoryGMMMap(gmm, learn_data.shape[0], gv_gmm)
    output_mfcc = gmmmap.convert(warp_data)

    output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame]) for frame in xrange(output_mfcc.shape[0])])
    target.SPEC = output_spec

    target.savefile(sys.argv[3])
