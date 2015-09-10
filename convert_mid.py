#!/usr/bin/env python

import math
import numpy
import pickle
import sys

from stf import STF
from mfcc import MFCC

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print 'Usage: %s [gmm_1] [gmm_2] [f0] [input] [output]' % sys.argv[0]
        sys.exit()

    gmm_file = open(sys.argv[1], 'rb')
    gmm_first = pickle.load(gmm_file)[1]
    gmm_file.close()

    gmm_file = open(sys.argv[2], 'rb')
    gmm_second = pickle.load(gmm_file)[0]
    gmm_file.close()

    f0_file = open(sys.argv[3], 'rb')
    f0 = pickle.load(f0_file)
    f0_file.close()

    source = STF()
    source.loadfile(sys.argv[4])

    f0_data = []
    for i in source.F0:
        if i == 0:
            f0_data.append(i)
        else:
            f0_data.append(math.e ** ((math.log(i) - math.log(f0[0][0])) * math.log(f0[1][1]) / math.log(f0[1][0]) + math.log(f0[0][1])))

    source.F0 = numpy.array(f0_data)

    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency)
    source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])
    source_data = numpy.hstack([source_mfcc, mfcc.delta(source_mfcc)])

    middle_mfcc = gmm_first.predict(source_data)
    middle_data = numpy.hstack([middle_mfcc, mfcc.delta(middle_mfcc)])

    output_mfcc = gmm_second.predict(middle_data)
    output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame]) for frame in xrange(output_mfcc.shape[0])])

    source.SPEC = output_spec
    source.savefile(sys.argv[5])
