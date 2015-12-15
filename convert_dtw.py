#!/usr/bin/env python

import numpy
import os
import pickle
import re
import scipy
import scipy.linalg
import sys

from stf import STF
from mfcc import MFCC
from dtw import DTW

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print 'Usage: %s <source stf> <target stf> <dtw cache directory> <input stf> <output stf>' % sys.argv[0]
        sys.exit()

    source, target = STF(), STF()
    source.loadfile(sys.argv[1])
    target.loadfile(sys.argv[2])

    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency)
    source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])

    mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency)
    target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) for frame in xrange(target.SPEC.shape[0])])

    cache_path = os.path.join(sys.argv[3], '%s_%s.dtw' % tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), sys.argv[1:3])))
    if os.path.exists(cache_path):
        dtw = pickle.load(open(cache_path))
    else:
        dtw = DTW(source_mfcc, target_mfcc, window = int(abs(source.SPEC.shape[0] - target.SPEC.shape[0]) * 0.2))
        with open(cache_path, 'wb') as output:
            pickle.dump(dtw, output)

    origin = STF()
    origin.loadfile(sys.argv[4])

    target.SPEC = dtw.align(origin.SPEC)
    target.savefile(sys.argv[5])
