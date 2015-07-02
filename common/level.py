# coding: utf-8

import cv2
import numpy as np
from .cnns import getCNNs
from .utils import getPatch, processImage


def level1(img, bbox, FOnly=True):
    """
        LEVEL-1
        img: gray image
        bbox: bounding box of face
    """
    F, EN, NM = getCNNs(level=1)
    # F
    f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]
    f_face = cv2.resize(f_face, (39, 39))
    en_face = f_face[:31, :]
    nm_face = f_face[8:, :]

    f_face = f_face.reshape((1, 1, 39, 39))
    f_face = processImage(f_face)
    f = F.forward(f_face)
    if FOnly:
        return f
    # EN
    # en_bbox = bbox.subBBox(-0.05, 1.05, -0.04, 0.84)
    # en_face = img[en_bbox.top:en_bbox.bottom+1,en_bbox.left:en_bbox.right+1]
    en_face = cv2.resize(en_face, (31, 39)).reshape((1, 1, 31, 39))
    en_face = processImage(en_face)
    en = EN.forward(en_face)
    # NM
    # nm_bbox = bbox.subBBox(-0.05, 1.05, 0.18, 1.05)
    # nm_face = img[nm_bbox.top:nm_bbox.bottom+1,nm_bbox.left:nm_bbox.right+1]
    nm_face = cv2.resize(nm_face, (31, 39)).reshape((1, 1, 31, 39))
    nm_face = processImage(nm_face)
    nm = NM.forward(nm_face)

    landmark = np.zeros((5, 2))
    landmark[0] = (f[0]+en[0]) / 2
    landmark[1] = (f[1]+en[1]) / 2
    landmark[2] = (f[2]+en[2]+nm[0]) / 3
    landmark[3] = (f[3]+nm[1]) / 2
    landmark[4] = (f[4]+nm[2]) / 2
    return landmark


def _level(img, bbox, landmark, cnns, padding):
    """
        LEVEL-?
    """
    for i in range(5):
        x, y = landmark[i]
        patch, patch_bbox = getPatch(img, bbox, (x, y), padding[0])
        patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
        patch = processImage(patch)
        d1 = cnns[2*i].forward(patch) # size = 1x2
        patch, patch_bbox = getPatch(img, bbox, (x, y), padding[1])
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
    landmark = _level(img, bbox, landmark, cnns, [0.16, 0.18])
    return landmark

def level3(img, bbox):
    """
        LEVEL-3
        img: gray image
        bbox: bounding box of face
    """
    landmark = level1(img, bbox)
    cnns = getCNNs(2)
    landmark = _level(img, bbox, landmark, cnns, [0.16, 0.18])
    cnns = getCNNs(3)
    landmark = _level(img, bbox, landmark, cnns, [0.11, 0.12])
    return landmark
