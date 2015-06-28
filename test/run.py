#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file use Caffe model to predict data from http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
"""

import os, sys
from functools import partial
import cv2
from common import getDataFromTxt, createDir, logger, drawLandmark
from common import level1, level2, level3


TXT = 'dataset/test/lfpw_test_249_bbox.txt'

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    level = int(sys.argv[1])
    if level == 0:
        P = partial(level1, FOnly=True)
    elif level == 1:
        P = level1
    elif level == 2:
        P = level2
    else:
        P = level3

    OUTPUT = 'dataset/test/out_{0}'.format(level)
    createDir(OUTPUT)
    data = getDataFromTxt(TXT, with_landmark=False)
    for imgPath, bbox in data:
        img = cv2.imread(imgPath)
        assert(img is not None)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger("process %s" % imgPath)

        landmark = P(imgGray, bbox)
        landmark = bbox.reprojectLandmark(landmark)
        drawLandmark(img, bbox, landmark)
        cv2.imwrite(os.path.join(OUTPUT, os.path.basename(imgPath)), img)
