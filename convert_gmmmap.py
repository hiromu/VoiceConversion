#!/usr/bin/env python

import math
import numpy
import pickle
import sklearn
import sys

from scipy.stats import multivariate_normal

from stf import STF
from mfcc import MFCC
from dtw import DTW

from gmmmap import GMMMap, TrajectoryGMMMap

K = 32
DIMENSION = 16
Trajectory = False

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: %s [gmm] [f0] [input] [output]' % sys.argv[0]
        sys.exit()

    gmm_file = open(sys.argv[1], 'rb')
    gmm = pickle.load(gmm_file)
    gmm_file.close()

    f0_file = open(sys.argv[2], 'rb')
    f0 = pickle.load(f0_file)
    f0_file.close()

    source = STF()
    source.loadfile(sys.argv[3])

    f0_data = []
    for i in source.F0:
        if i == 0:
            f0_data.append(i)
        else:
            f0_data.append(math.e ** ((math.log(i) - math.log(f0[0][0])) * math.log(f0[1][1]) / math.log(f0[1][0]) + math.log(f0[0][1])))

    source.F0 = numpy.array(f0_data)

    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = DIMENSION)
    source_data = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])

    if Trajectory == True:
        gmmmap = TrajectoryGMMMap(gmm, 100)
        output_mfcc = gmmmap.convert(source_data)
        
        mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = DIMENSION / 2)
        output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame, :DIMENSION / 2]) for frame in xrange(output_mfcc.shape[0])])
    else:
        gmmmap = GMMMap(gmm)
        output_mfcc = numpy.array([gmmmap.convert(source_data[frame])[0] for frame in xrange(source_data.shape[0])])
        output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame, :DIMENSION]) for frame in xrange(output_mfcc.shape[0])])

    source.SPEC = output_spec
    source.savefile(sys.argv[4])
