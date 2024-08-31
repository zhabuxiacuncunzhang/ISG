import os
import numpy as np
from rich.progress import track

class HEADER:
    width = 0
    length = 0
    xfirst = 0.0
    yfirst = 0.0
    xstep = 0.0
    ystep = 0.0


def readHeader(filename):
    if not os.path.isfile(filename):
        print(filename+" header file not exit")
    print("read header file:", filename)
    header = HEADER()
    with open(filename) as f:
        for line in f:
            data = line.split()
            if data[0] == "WIDTH":
                header.width = int(data[1])
            if data[0] == "FILE_LENGTH":
                header.length = int(data[1])
            if data[0] == "X_FIRST":
                header.xfirst = float(data[1])
            if data[0] == "Y_FIRST":
                header.yfirst = float(data[1])
            if data[0] == "X_STEP":
                header.xstep = float(data[1])
            if data[0] == "Y_STEP":
                header.ystep = float(data[1])
    print("with:", header.width, " length:", header.length)
    return header


def readPhs(filename):
    if not os.path.isfile(filename):
        print(filename+" phs file not exit")
    #print("read phs file:", filename.strip())
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data


def readFile(file, header):
    if not os.path.isfile(file):
        print(file+" file not exit")
    print("read file list:", file)
    f = open(file, 'r')
    filenum = 0
    for line in open(file):
        filenum = filenum+1
    header = readHeader(header)
    phsstack = np.zeros(shape=(header.length*header.width, filenum))
    i=0
    for line in track(open(file), description="reading phs files",total=filenum):
        line = f.readline()
        phs = readPhs(line.strip())
        phsstack[:,i]=phs
        i=i+1

    return header, phsstack, filenum


def writePhs(data, filename):
    #print("write file to:", filename)
    data.tofile(filename)
