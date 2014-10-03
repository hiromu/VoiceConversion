#!/usr/bin/env python

import numpy
import pickle
import sys

from stf import STF
from mfcc import MFCC

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: %s [gmm] [input] [output]' % sys.argv[0]
        sys.exit()

    gmm_file = open(sys.argv[1], 'rb')
    gmmmap = pickle.load(gmm_file)
    gmm_file.close()

    source = STF()
    source.loadfile(sys.argv[2])

    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency)
    source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])
    source_data = numpy.hstack([source_mfcc, mfcc.delta(source_mfcc)])

    output_mfcc = gmmmap.convert(source_data)
    output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame]) for frame in xrange(output_mfcc.shape[0])])

    source.SPEC = output_spec
    source.savefile(sys.argv[3])
