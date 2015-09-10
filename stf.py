#!/usr/bin/env python

import numpy
import os
import struct
import sys

class STF:
    def __init__(self, filename = None):
        self.endian = '>'
        self.chunks = ['APSG', 'F0  ', 'SPEC']

    def loadfile(self, filename):
        with open(filename, 'rb') as stf_file:
            self.load(stf_file)

    def load(self, stf_file):
        filesize = os.fstat(stf_file.fileno()).st_size

        while stf_file.tell() < filesize:
            chunk = stf_file.read(4)

            if chunk == 'STRT':
                if stf_file.read(2) == '\xff\xfe':
                    self.endian = '<'
                chunk_size, self.version, self.channel, self.frequency = struct.unpack(self.endian + 'IHHI', stf_file.read(12))
            else:
                chunk_size, = struct.unpack(self.endian + 'I', stf_file.read(4))

                if chunk == 'CHKL' or chunk == 'NXFL':
                    data = stf_file.read(chunk_size)
                    if chunk == 'CHKL':
                        self.chunks += [data[i: i + 4] for i in range(0, chunk_size, 4) if data[i: i + 4] not in self.chunks]
                else:
                    self.shift_length, frame_count, argument, self.bit_size, self.weight, data_size = struct.unpack(self.endian + 'dIIHdI', stf_file.read(30))
                    data = stf_file.read(data_size)

                    element = data_size / (self.bit_size / 8)
                    matrix = numpy.fromstring(data, count = element)

                    for c in self.chunks:
                        if chunk == c:
                            if element / frame_count == 1:
                                self.__dict__[c.strip()] = matrix
                            else:
                                self.__dict__[c.strip()] = matrix.reshape((frame_count, element / frame_count))
                            break

        for c in self.chunks:
            if c.strip() not in self.__dict__:
                self.__dict__[c.strip()] = None

    def savefile(self, filename):
        with open(filename, 'wb') as stf_file:
            self.save(stf_file)

    def save(self, stf_file):
        stf_file.write('STRT')
        if self.endian == '>':
            stf_file.write('\xfe\xff')
        elif self.endian == '<':
            stf_file.write('\xff\xfe')
        stf_file.write(struct.pack(self.endian + 'IHHI', 8, self.version, self.channel, self.frequency))

        stf_file.write('CHKL')
        stf_file.write(struct.pack(self.endian + 'I', len(''.join(self.chunks))) + ''.join(self.chunks))

        for c in self.chunks:
            if self.__dict__[c.strip()] is None:
                continue

            matrix = self.__dict__[c.strip()]
            if len(matrix.shape) == 1:
                argument = 1
            else:
                argument = matrix.shape[1]
            data_size = matrix.shape[0] * argument * 8

            header = struct.pack(self.endian + 'dIIHdI', self.shift_length, matrix.shape[0], argument, self.bit_size, self.weight, data_size)
            stf_file.write(c + struct.pack(self.endian + 'I', len(header) + data_size) + header)

            for i in xrange(matrix.shape[0]):
                if argument == 1:
                    stf_file.write(struct.pack(self.endian + 'd', matrix[i]))
                else:
                    for j in xrange(matrix.shape[1]):
                        stf_file.write(struct.pack(self.endian + 'd', matrix[i, j]))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: %s <stf_file>' % sys.argv[0]
        sys.exit()

    stf = STF()
    stf.loadfile(sys.argv[1])
    print stf.F0
