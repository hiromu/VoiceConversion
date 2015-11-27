#!/usr/bin/env python
# coding: utf-8

from stf import STF
from mfcc import MFCC
from dtw import DTW
from evgmm import TrajectoryEVGMM

import numpy
import os
import pickle
import re
import sys

D = 16

def one_to_many(source_list, target_list, dtw_cache):
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

            cache_path = os.path.join(dtw_cache, '%s_%s.dtw' % tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), [source_list[j], target_list[i][j]])))
            if os.path.exists(cache_path):
                dtw = pickle.load(open(cache_path))
            else:
                dtw = DTW(source_data[j], target_data, window = abs(source_data[j].shape[0] - target_data.shape[0]) * 2)
                with open(cache_path, 'wb') as output:
                    pickle.dump(dtw, output)

            warp_data = dtw.align(target_data, reverse = True)

            data = numpy.hstack([source_data[j], warp_data])
            if learn_data is None:
                learn_data = data
            else:
                learn_data = numpy.vstack([learn_data, data])

        total_data.append(learn_data)

    return total_data

def many_to_one(source_list, target_list, dtw_cache):
    target_data = []

    for i in xrange(len(target_list)):
        target = STF()
        target.loadfile(target_list[i])

        mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
        target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) for frame in xrange(target.SPEC.shape[0])])
        target_data.append(numpy.hstack([target_mfcc, mfcc.delta(target_mfcc)]))

    total_data = []

    for i in xrange(len(source_list)):
        learn_data = None

        for j in xrange(len(source_list[i])):
            print i, j

            source = STF()
            source.loadfile(source_list[i][j])

            mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
            source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) for frame in xrange(source.SPEC.shape[0])])
            source_data = numpy.hstack([source_mfcc, mfcc.delta(source_mfcc)])

            cache_path = os.path.join(sys.argv[3], '%s_%s.dtw' % tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), [source_list[i][j], target_list[j]])))
            if os.path.exists(cache_path):
                dtw = pickle.load(open(cache_path))
            else:
                dtw = DTW(source_data, target_data[j], window = abs(source_data.shape[0] - target_data[j].shape[0]) * 2)
                with open(cache_path, 'wb') as output:
                    pickle.dump(dtw, output)

            warp_data = dtw.align(source_data)

            data = numpy.hstack([warp_data, target_data[j]])
            if learn_data is None:
                learn_data = data
            else:
                learn_data = numpy.vstack([learn_data, data])

        total_data.append(learn_data)

    return total_data

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: %s [list of source stf] [list of target] [dtw cache directory] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    if len(filter(lambda s: not s.endswith('.stf'), source_list)) == 0:
        target_list = [open(target).read().strip().split('\n') for target in target_list]
        total_data = one_to_many(source_list, target_list, sys.argv[3])
    elif len(filter(lambda s: not s.endswith('.stf'), target_list)) == 0:
        source_list = [open(source).read().strip().split('\n') for source in source_list]
        total_data = many_to_one(source_list, target_list, sys.argv[3])

    evgmm = TrajectoryEVGMM(total_data)

    with open(sys.argv[4], 'wb') as output:
        pickle.dump(evgmm, output)
