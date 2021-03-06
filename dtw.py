#!/usr/bin/env python

import numpy
import scipy
import scipy.linalg
import sys

class DTW:
    @staticmethod
    def cosine(A, B):
        return scipy.dot(A, B.transpose()) / scipy.linalg.norm(A) / scipy.linalg.norm(B)

    @staticmethod
    def euclidean(A, B):
        return scipy.linalg.norm(A - B)

    def __init__(self, source, target, distance = None, window = sys.maxint):
        self.window = window
        self.source = source
        self.target = target

        if distance:
            self.distance = distance
        else:
            self.distance = self.euclidean

        self.dtw()

    def dtw(self):
        M, N = len(self.source), len(self.target)
        cost = sys.maxint * numpy.ones((M, N))

        cost[0, 0] = self.distance(self.source[0], self.target[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + self.distance(self.source[i], self.target[0])
        for i in range(1, N):
            cost[0, i] = cost[0, i - 1] + self.distance(self.source[0], self.target[i])

        for i in range(1, M):
            for j in range(max(1, i - self.window), min(N, i + self.window)):
                cost[i, j] = min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]) + self.distance(self.source[i], self.target[j])

        m, n = M - 1, N - 1
        self.path = []
        
        while (m, n) != (0, 0):
            self.path.append((m, n))
            m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key = lambda x: cost[x[0], x[1]])
            if m < 0 or n < 0:
                break

        self.path.append((0, 0))

    def align(self, data, reverse = False):
        if reverse:
            path = [(t[1], t[0]) for t in self.path]
            source = self.target
            target = self.source
        else:
            path = self.path
            source = self.source
            target = self.target

        path.sort(key = lambda x: (x[1], x[0]))

        shape = tuple([path[-1][1] + 1] + list(data.shape[1:]))
        alignment = numpy.ndarray(shape)

        idx = 0
        frame = 0
        candicates = []

        while idx < len(path) and frame < target.shape[0]:
            if path[idx][1] > frame:
                candicates.sort(key = lambda x: self.distance(source[x], target[frame]))
                alignment[frame] = data[candicates[0]]

                candicates = [path[idx][0]]
                frame += 1
            else:
                candicates.append(path[idx][0])
                idx += 1

        if frame < target.shape[0]:
            candicates.sort(key = lambda x: self.distance(source[x], target[frame]))
            alignment[frame] = data[candicates[0]]

        return alignment

if __name__ == '__main__':
    A = numpy.random.rand(30) * 3 + numpy.arange(30)
    B = numpy.random.rand(20) * 3 + numpy.arange(0, 30, 1.5)

    dtw = DTW(A, B, distance = lambda x, y: abs(x - y), window = 15)
    C = dtw.align(A)
    D = dtw.align(B, reverse = True)

    import pylab
    offset = 5
    pylab.xlim([-1, max(len(A), len(B)) + 1])
    pylab.plot(A)
    pylab.plot(B + offset)
    pylab.plot(C + offset * 2)
    pylab.plot(D + offset * 3)
    pylab.show()
