#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file use Caffe model to predict landmarks and evaluate the mean error.
"""

import os, sys
import time
from functools import partial
import cv2
import numpy as np
from numpy.linalg import norm
from common import getDataFromTxt, logger


TXT = 'dataset/train/testImageList.txt'
template = '''################## Summary #####################
Test Number: %d
Time Consume: %.03f s
LEVEL - %d
Mean Error:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
Failure:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
'''

def evaluateError(landmarkGt, landmarkP, bbox):
    e = np.zeros(5)
    for i in range(5):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / bbox.w
    print 'landmarkGt'
    print landmarkGt
    print 'landmarkP'
    print landmarkP
    print 'error', e
    return e


def E(level=1):
    if level == 0:
        from common import level1 as P
        P = partial(P, FOnly=True) # high order function, here we only test LEVEL-1 F CNN
    elif level == 1:
        from common import level1 as P
    elif level == 2:
        from common import level2 as P
    else:
        from common import level3 as P

    data = getDataFromTxt(TXT)
    error = np.zeros((len(data), 5))
    for i in range(len(data)):
        imgPath, bbox, landmarkGt = data[i]
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)

        landmarkP = P(img, bbox)

        # real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        error[i] = evaluateError(landmarkGt, landmarkP, bbox)
    return error


nameMapper = ['F_test', 'level1_test', 'level2_test', 'level3_test']

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    level = int(sys.argv[1])

    t = time.clock()
    error = E(level)
    t = time.clock() - t

    print '\n'
    print '################## Summary #####################'
    print 'Test Number:', len(error)
    print 'Time Consume:', t, 's'
    print 'LEVEL -', level
    print 'Mean Error'
    print error.mean(0)
    # failure
    N = len(error)
    failure = np.zeros(5)
    threshold = 0.05
    for i in range(5):
        failure[i] = float(sum(error[:, i] > threshold)) / N
    print 'Average Failure'
    print failure

    logfile = 'log/{0}.log'.format(nameMapper[level])
    with open(logfile, 'w') as fd:
        s = template % (N, t, level, error.mean(0)[0], error.mean(0)[1], \
            error.mean(0)[2], error.mean(0)[3], error.mean(0)[4], \
            failure[0], failure[1], failure[2], failure[3], failure[4])
        fd.write(s)
