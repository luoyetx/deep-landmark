# coding: utf-8

import cv2
import numpy as np
from .cnns import getCNNs
from .utils import getPatch, processImage


def level1(img, bbox, FOnly=False):
    """
        LEVEL-1
        img: gray image
        bbox: bounding box of face
    """
    bbox = bbox.expand(0.05)
    face = img[bbox.top:bbox.bottom+1,bbox.left:bbox.right+1]
    face = cv2.resize(face, (39, 39)).reshape((1, 1, 39, 39))
    face = processImage(face)

    F, EN, NM = getCNNs(level=1)
    # all landmarks
    f = F.forward(face)
    if FOnly:
        return f
    en = EN.forward(face[:, :, :31, :])
    nm = NM.forward(face[:, :, 8:, :])

    landmark = np.zeros((5, 2))
    landmark[0] = (f[0]+en[0]) / 2
    landmark[1] = (f[1]+en[1]) / 2
    landmark[2] = (f[2]+en[2]+nm[0]) / 3
    landmark[3] = (f[3]+nm[1]) / 2
    landmark[4] = (f[4]+nm[2]) / 2
    return landmark


def _level(img, bbox, landmark, cnns):
    """
        LEVEL-?
    """
    for i in range(5):
        x, y = landmark[i]
        patch, patch_bbox = getPatch(img, bbox, (x, y), 0.16)
        patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
        patch = processImage(patch)
        d1 = cnns[2*i].forward(patch) # size = 1x2
        patch, patch_bbox = getPatch(img, bbox, (x, y), 0.18)
        patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
        patch = processImage(patch)
        d2 = cnns[2*i+1].forward(patch)

        d1 = bbox.project(patch_bbox.reproject(d1[0]))
        d2 = bbox.project(patch_bbox.reproject(d2[0]))
        landmark[i] = (d1 + d2) / 2
    return landmark

def level2(img, bbox):
    """
        LEVEL-2
        img: gray image
        bbox: bounding box of face
    """
    landmark = level1(img, bbox)
    cnns = getCNNs(2)
    landmark = _level(img, bbox, landmark, cnns)
    return landmark

def level3(img, bbox):
    """
        LEVEL-3
        img: gray image
        bbox: bounding box of face
    """
    landmark = level1(img, bbox)
    cnns = getCNNs(2)
    landmark = _level(img, bbox, landmark, cnns)
    cnns = getCNNs(3)
    landmark = _level(img, bbox, landmark, cnns)
    return landmark
