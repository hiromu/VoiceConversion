#!/usr/bin/env python

import numpy
import pylab
import sys

from stf import STF
from mfcc import MFCC

if __name__ == '__main__':
    N = len(sys.argv) - 1

    for i in range(1, len(sys.argv)):
        stf = STF()
        stf.loadfile(sys.argv[i])

        mfcc = MFCC(stf.SPEC.shape[1] * 2, stf.frequency)
        data = numpy.array([mfcc.mfcc(stf.SPEC[frame]) for frame in xrange(stf.SPEC.shape[0])])

        pylab.subplot(int('%d1%d' % (N, i)))
        pylab.plot(data[:, 0])
        pylab.plot(data[:, 1])

    pylab.show()
