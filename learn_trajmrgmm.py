#!/usr/bin/env python
# coding: utf-8

from stf import STF
from mfcc import MFCC
from dtw import DTW
from mrgmm import TrajectoryMRGMM

import csv
import numpy
import os
import pickle
import re
import sys

D = 16

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print 'Usage: %s [list of source stf] [list of target] [score tsv] [dtw cache directory] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = [open(target_list).read().strip().split('\n') for target_list in open(sys.argv[2]).read().strip().split('\n')]

    source_data = []

    for i in xrange(len(source_list)):
        source = STF()
        source.loadfile(source_list[i])

        mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
        source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])
        source_data.append(numpy.hstack([source_mfcc, mfcc.delta(source_mfcc)]))

    total_data = []

    for i in xrange(len(target_list)):
        learn_data = None

        for j in xrange(len(target_list[i])):
            print i, j

            target = STF()
            target.loadfile(target_list[i][j])

            mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
            target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) for frame in xrange(target.SPEC.shape[0])])
            target_data = numpy.hstack([target_mfcc, mfcc.delta(target_mfcc)])

            cache_path = os.path.join(sys.argv[4], '%s_%s.dtw' % tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), [source_list[j], target_list[i][j]])))
            if os.path.exists(cache_path):
                dtw = pickle.load(open(cache_path))
            else:
                dtw = DTW(source_data[j], target_data, window = abs(source.SPEC.shape[0] - target.SPEC.shape[0]) * 2)
                with open(cache_path, 'wb') as output:
                    pickle.dump(dtw, output)

            warp_data = dtw.align(target_data, reverse = True)

            data = numpy.hstack([source_data[j], warp_data])
            if learn_data is None:
                learn_data = data
            else:
                learn_data = numpy.vstack([learn_data, data])

        total_data.append(learn_data)

    reader = csv.reader(open(sys.argv[3]), delimiter = '\t')
    scores = numpy.array([row for row in reader][1:], dtype = numpy.float64)

    mrgmm = TrajectoryMRGMM(numpy.array(total_data), scores)

    with open(sys.argv[5], 'wb') as output:
        pickle.dump(mrgmm, output)
