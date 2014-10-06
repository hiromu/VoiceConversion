#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy
import pylab
import sys

import matplotlib.patches

from stf import STF
from mfcc import MFCC

def delta(data, frame = 5):
    assert frame % 2 == 1

    shift = frame / 2
    x = numpy.array([(1, i) for i in xrange(frame)])
    data = numpy.concatenate(([data[0]] * shift, data, [data[-1]] * shift))

    delta = numpy.array([])
    for i in xrange(shift, len(data) - shift):
        solution, residuals, rank, s = numpy.linalg.lstsq(x, data[i - shift: i + shift + 1])
        delta = numpy.append(delta, solution[1])

    return delta[shift: -shift]

def analysis(stf_files):
    stf = STF()

    targets = ['f0', 'f0_delta', 'ap_fc', 'ap_alpha']
    variables = locals()
    for target in targets:
        variables[target] = [numpy.array([]) for i in xrange(3)]

    for stf_file in stf_files:
        stf.loadfile(stf_file)

        voice = (stf.F0 != 0)

        intervals = []
        past = False

        for i in xrange(len(voice)):
            if past and not voice[i]:
                intervals[-1] = (intervals[-1][0], i)
                past = False
            elif not past and voice[i]:
                intervals.append((i, -1))
                past = True
        if intervals[-1][1] == -1:
            intervals[-1] = (intervals[-1][0], len(voice))

        for interval in intervals:
            f0_data = stf.F0[interval[0]: interval[1]]
            f0_delta_data = delta(f0_data)
            ap_fc_data = stf.APSG[interval[0]: interval[1], 0] * stf.APSG[interval[0]: interval[1], 1] * -1
            ap_alpha_data = stf.APSG[interval[0]: interval[1], 0]

            variables = locals()
            for name in targets:
                variables[name][0] = numpy.append(variables[name][0], variables[name + '_data'][:5])
                variables[name][1] = numpy.append(variables[name][1], variables[name + '_data'])
                variables[name][2] = numpy.append(variables[name][2], variables[name + '_data'][-5:])

    variables = locals()
    return [[x.mean() for x in variables[target]] for target in targets]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: %s [base_dir] [num]' % sys.argv[0]

    base_dir = sys.argv[1]
    num = int(sys.argv[2])

    color = ['#57B196', '#FFD25A', '#FF837B']
    hatch = ['.', '/', 'x']
    legend = [u'先頭', u'全体', u'末尾']

    label = [u'F0 (Hz)', u'ΔF0', u'Aperiodic component (Hz)', u'Aperiodic component (exponent)']
    location = ['upper right', 'lower left', 'upper left', 'upper left']

    results = [analysis(glob.glob(base_dir % i)) for i in xrange(1, num + 1)]

    for i in xrange(4):
        for j in xrange(3):
            for k in xrange(3):
                pylab.bar(j * 4 + k + 0.5, results[j][i][k], edgecolor = '#362F3C', facecolor = color[j], hatch = hatch[k])

        pylab.xticks(xrange(2, 11, 4), ['A', 'B', 'C'])
        pylab.ylabel(label[i])

        patches = [matplotlib.patches.Patch(hatch = hatch[x], edgecolor = 'k', facecolor = 'w') for x in xrange(3)]
        pylab.legend(patches, legend, loc = location[i])

        pylab.show()
