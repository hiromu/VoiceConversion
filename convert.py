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

K = 32
DIMENSION = 16

def convert(source, gmm):
    gauss = []
    for k in range(K):
        gauss.append(multivariate_normal(gmm.means_[k, 0:DIMENSION], gmm.covars_[k, 0:DIMENSION, 0:DIMENSION]))
 
    ss = []
    for k in range(K):
        ss.append(numpy.dot(gmm.covars_[k, DIMENSION:, 0:DIMENSION], numpy.linalg.inv(gmm.covars_[k, 0:DIMENSION, 0:DIMENSION])))
 
    output = None
    for t in range(len(source)):
        x_t = source[t]
        y_t = convert_frame(x_t, gmm, gauss, ss)
        if output != None:
            output = numpy.vstack([output, y_t])
        else:
            output = y_t
    return output
 
def convert_frame(x, gmm, gauss, ss):
    y = numpy.zeros_like(x)
    for k in range(K):
        y += P(k, x, gmm, gauss) * E(k, x, gmm, ss)
    return y
 
def P(k, x, gmm, gauss):
    denom = numpy.zeros(K)
    for n in range(K):
        denom[n] = gmm.weights_[n] * gauss[n].pdf(x)
    return denom[k] / numpy.sum(denom)

def E(k, x, gmm, ss):
    return gmm.means_[k, DIMENSION:] + numpy.dot(ss[k], x - gmm.means_[k, 0:DIMENSION])

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
    source.F0[source.F0 != 0] = numpy.exp((numpy.log(source.F0[source.F0 != 0]) - f0[0][0]) * f0[1][1] / f0[1][0] + f0[0][1])

    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = DIMENSION)
    source_data = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])

    output_mfcc = convert(source_data, gmm)
    print output_mfcc.shape
    output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame, :DIMENSION]) for frame in xrange(output_mfcc.shape[0])])

    source.SPEC = output_spec
    source.savefile(sys.argv[4])
