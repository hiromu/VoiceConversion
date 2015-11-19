#!/usr/bin/env python

import math
import numpy
import pickle
import sklearn
import sys

from evgmm import GMM

from stf import STF
from mfcc import MFCC
from dtw import DTW

D = 16

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: %s [gmmmap] [f0] [source speaker stf] [target speaker stf] [output]' % sys.argv[0]
        sys.exit()
    
    with open(sys.argv[1], 'rb') as infile:
        evgmm = pickle.load(infile)

    with open(sys.argv[2], 'rb') as infile:
        f0 = pickle.load(infile)

    source = STF()
    source.loadfile(sys.argv[3])
    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
    source_data = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])

    target = STF()
    target.loadfile(sys.argv[4])
    mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
    target_data = numpy.array([mfcc.mfcc(target.SPEC[frame]) for frame in xrange(target.SPEC.shape[0])])

    evgmm.fit(target_data)

    output_mfcc = numpy.array([evgmm.convert(source_data[frame])[0] for frame in xrange(source_data.shape[0])])
    output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame]) for frame in xrange(output_mfcc.shape[0])])

    source.SPEC = output_spec
    source.savefile(sys.argv[5])

    f0_data = []
    for i in source.F0:
        if i == 0:
            f0_data.append(i)
        else:
            f0_data.append(math.e ** ((math.log(i) - math.log(f0[0][0])) * math.log(f0[1][1]) / math.log(f0[1][0]) + math.log(f0[0][1])))

    source.F0 = numpy.array(f0_data)
