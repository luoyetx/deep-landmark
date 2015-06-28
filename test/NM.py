#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file use Caffe model to predict landmarks and evaluate the mean error.
"""

import os, sys
import time
import cv2
import numpy as np
from numpy.linalg import norm
from common import getDataFromTxt, logger, processImage, getCNNs


TXT = 'dataset/train/testImageList.txt'
template = '''################## Summary #####################
Test Number: %d
Time Consume: %.03f s
FPS: %.03f
LEVEL - %d
Mean Error:
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
Failure:
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
'''

def evaluateError(landmarkGt, landmarkP, bbox):
    e = np.zeros(3)
    for i in range(3):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / bbox.w
    print 'landmarkGt'
    print landmarkGt
    print 'landmarkP'
    print landmarkP
    print 'error', e
    return e

def NM(img, bbox):
    """
        LEVEL-1, NM
        img: gray image
        bbox: bounding box of face
    """
    bbox = bbox.expand(0.05)
    face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    face = cv2.resize(face, (39, 39)).reshape((1, 1, 39, 39))
    face = processImage(face)

    F, EN, NM = getCNNs(level=1) # TODO more flexible load needed.
    landmark = NM.forward(face[:, :, 8:, :])
    return landmark


def E():
    data = getDataFromTxt(TXT)
    error = np.zeros((len(data), 3))
    for i in range(len(data)):
        imgPath, bbox, landmarkGt = data[i]
        landmarkGt = landmarkGt[2:, :]
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)

        landmarkP = NM(img, bbox)

        # real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        error[i] = evaluateError(landmarkGt, landmarkP, bbox)
    return error

if __name__ == '__main__':

    t = time.clock()
    error = E()
    t = time.clock() - t

    N = len(error)
    fps = N / t
    errorMean = error.mean(0)
    # failure
    failure = np.zeros(3)
    threshold = 0.05
    for i in range(3):
        failure[i] = float(sum(error[:, i] > threshold)) / N
    # log string
    s = template % (N, t, fps, 1, errorMean[0], errorMean[1], errorMean[2], \
        failure[0], failure[1], failure[2])
    print s

    logfile = 'log/1_NM.log'
    with open(logfile, 'w') as fd:
        fd.write(s)
