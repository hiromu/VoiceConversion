#!/usr/bin/env python

import math
import numpy
import pickle
import sklearn
import sys

from stf import STF
from mfcc import MFCC
from dtw import DTW

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: %s [list of source stf] [list of target stf] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    assert len(source_list) == len(target_list)

    f0_count = [0, 0]
    f0_mean = [0.0, 0.0]
    f0_square_mean = [0.0, 0.0]

    for i in xrange(len(source_list)):
        source = STF()
        source.loadfile(source_list[i])

        target = STF()
        target.loadfile(target_list[i])

        for idx, stf in enumerate([source, target]):
            count = (stf.F0 != 0).sum()
            f0_mean[idx] = (f0_mean[idx] * f0_count[idx] + numpy.log(stf.F0[stf.F0 != 0]).sum()) / (f0_count[idx] + count)
            f0_square_mean[idx] = (f0_square_mean[idx] * f0_count[idx] + (numpy.log(stf.F0[stf.F0 != 0]) ** 2).sum()) / (f0_count[idx] + count)
            f0_count[idx] += count

    f0_deviation = [math.sqrt(f0_square_mean[i] - f0_mean[i] ** 2) for i in xrange(2)]
    f0 = (tuple(f0_mean), tuple(f0_deviation))

    print f0
    output = open(sys.argv[3], 'wb')
    pickle.dump(f0, output)
    output.close()
