#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file convert dataset from http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm
    We convert data for LEVEL-1 training data.
    all data are formated as (data, landmark), and landmark is ((x1, y1), (x2, y2)...)
"""

import os
import time
import math
from os.path import join, exists
import cv2
import numpy as np
import h5py
from common import shuffle_in_unison_scary, logger, createDir, processImage
from common import getDataFromTxt
from utils import show_landmark, flip


TRAIN = 'dataset/train'
OUTPUT = 'train'
if not exists(OUTPUT): os.mkdir(OUTPUT)
assert(exists(TRAIN) and exists(OUTPUT))
SIZE_W = SIZE_H = 39


def process_images(ftxt, output):
    """
        give a txt and generate a hdf5 file with output name
    """
    with open(ftxt, 'r') as fd:
        lines = fd.readlines()
    number = len(lines) # how many faces
    imgs = np.zeros((number*2, 1, SIZE_W, SIZE_H))
    landmarks = np.zeros((number*2, 10))

    for idx, line in enumerate(lines):
        line = line.strip()
        components = line.split(' ')
        img_path = join(TRAIN, components[0].replace('\\', '/')) # file path
        # bounding box, (left, right, top, bottom)
        bbox = (components[1], components[2], components[3], components[4])
        bbox = [int(_) for _ in bbox]
        # expand bbox
        w = bbox[1] - bbox[0]
        h = bbox[3] - bbox[2]
        bbox[0] -= int(w * 0.05)
        bbox[1] += int(w * 0.05)
        bbox[2] -= int(h * 0.05)
        bbox[3] += int(h * 0.05)

        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        assert(img is not None)
        logger("process %s" % img_path)

        face = img[bbox[2]:bbox[3]+1,bbox[0]:bbox[1]+1]
        face = cv2.resize(face, (SIZE_W, SIZE_H))

        # landmark
        # left eye center, right eye center, nose, left mouth corner, right mouth corner
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv

        # # rotation
        # (face_rotated_by_alpha, landmark_rotated) = rotate(img, bbox, landmark, 5)
        # face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (SIZE_W, SIZE_H))
        # show_landmark(face_rotated_by_alpha, landmark_rotated)
        # landmarks[idx+1*number] = landmark_rotated.reshape((10))
        # imgs[idx+1*number] = face_rotated_by_alpha.reshape((1, SIZE_W, SIZE_H))
        # # # flip after rotation
        # # (face_flipped_by_x, landmark_flipped) = flip(face_rotated_by_alpha, landmark_rotated)
        # # show_landmark(face_flipped_by_x, landmark_flipped)
        # # landmarks[idx+2*number] = landmark_flipped.reshape((10))
        # # imgs[idx+2*number] = face_flipped_by_x.reshape((1, SIZE_W, SIZE_H))

        # # rotation
        # (face_rotated_by_alpha, landmark_rotated) = rotate(img, bbox, landmark, -5)
        # face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (SIZE_W, SIZE_H))
        # show_landmark(face_rotated_by_alpha, landmark_rotated)
        # landmarks[idx+1*number] = landmark_rotated.reshape((10))
        # imgs[idx+1*number] = face_rotated_by_alpha.reshape((1, SIZE_W, SIZE_H))
        # # # flip after rotation
        # # (face_flipped_by_x, landmark_flipped) = flip(face_rotated_by_alpha, landmark_rotated)
        # # show_landmark(face_flipped_by_x, landmark_flipped)
        # # landmarks[idx+4*number] = landmark_flipped.reshape((10))
        # # imgs[idx+4*number] = face_flipped_by_x.reshape((1, SIZE_W, SIZE_H))

        # origin
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[1]-bbox[0]), (one[1]-bbox[2])/(bbox[3]-bbox[2]))
            landmark[index] = rv
        #show_landmark(face, landmark)
        landmarks[idx] = landmark.reshape((10))
        imgs[idx] = face.reshape((1, SIZE_W, SIZE_H))

        # flip
        (face_flipped_by_x, landmark_flipped) = flip(face, landmark)
        #show_landmark(face_flipped_by_x, landmark_flipped)
        landmarks[idx+1*number] = landmark_flipped.reshape((10))
        imgs[idx+1*number] = face_flipped_by_x.reshape((1, SIZE_W, SIZE_H))

    imgs = processImage(imgs)
    # for idx in range(len(imgs)):
    #     imgs[idx] -= imgs[idx].mean()
    shuffle_in_unison_scary(imgs, landmarks)

    return imgs, landmarks


def generate_hdf5(ftxt, output, fname):
    imgs, landmarks = process_images(ftxt, output)
    ps_len = len(imgs)

    # full face
    base = join(OUTPUT, '1_F')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = imgs.astype(np.float32)
        h5['landmark'] = landmarks.astype(np.float32)

    # eye and nose
    base = join(OUTPUT, '1_EN')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = imgs[:,:,:31,:].astype(np.float32)
        h5['landmark'] = landmarks[:,:6].astype(np.float32)

    # nose and mouth
    base = join(OUTPUT, '1_NM')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = imgs[:,:,8:,:].astype(np.float32)
        h5['landmark'] = landmarks[:,4:].astype(np.float32)


if __name__ == '__main__':
    # train data
    train_txt = join(TRAIN, 'trainImageList.txt')
    generate_hdf5(train_txt, OUTPUT, 'train.h5')

    test_txt = join(TRAIN, 'testImageList.txt')
    generate_hdf5(test_txt, OUTPUT, 'test.h5')

    with open(join(OUTPUT, '1_F/train.txt'), 'w') as fd:
        fd.write('train/1_F/train.h5')
    with open(join(OUTPUT, '1_EN/train.txt'), 'w') as fd:
        fd.write('train/1_EN/train.h5')
    with open(join(OUTPUT, '1_NM/train.txt'), 'w') as fd:
        fd.write('train/1_NM/train.h5')
    with open(join(OUTPUT, '1_F/test.txt'), 'w') as fd:
        fd.write('train/1_F/test.h5')
    with open(join(OUTPUT, '1_EN/test.txt'), 'w') as fd:
        fd.write('train/1_EN/test.h5')
    with open(join(OUTPUT, '1_NM/test.txt'), 'w') as fd:
        fd.write('train/1_NM/test.h5')
    # Done
