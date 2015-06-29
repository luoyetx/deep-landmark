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


def generate_hdf5(ftxt, output, fname, argument=False):

    data = getDataFromTxt(ftxt)
    F_imgs = []
    F_landmarks = []
    EN_imgs = []
    EN_landmarks = []
    NM_imgs = []
    NM_landmarks = []

    for (imgPath, bbox, landmarkGt) in data:
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        # F
        f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
        f_face = img[f_bbox.top:f_bbox.bottom+1,f_bbox.left:f_bbox.right+1]

        ## data argument
        if argument and np.random.rand() > 0.5:
            ### flip
            face_flipped, landmark_flipped = flip(f_face, landmarkGt)
            face_flipped = cv2.resize(face_flipped, (39, 39)).reshape((1, 39, 39))
            landmark_flipped = landmark_flipped.reshape((10))
            F_imgs.append(face_flipped)
            F_landmarks.append(landmark_flipped)

        f_face = cv2.resize(f_face, (39, 39))
        en_face = f_face[:31, :]
        nm_face = f_face[8:, :]

        f_face = f_face.reshape((1, 39, 39))
        f_landmark = landmarkGt.reshape((10))
        F_imgs.append(f_face)
        F_landmarks.append(f_landmark)

        # EN
        # en_bbox = bbox.subBBox(-0.05, 1.05, -0.04, 0.84)
        # en_face = img[en_bbox.top:en_bbox.bottom+1,en_bbox.left:en_bbox.right+1]

        ## data argument
        if argument and np.random.rand() > 0.5:
            ### flip
            face_flipped, landmark_flipped = flip(en_face, landmarkGt)
            face_flipped = cv2.resize(face_flipped, (31, 39)).reshape((1, 31, 39))
            landmark_flipped = landmark_flipped[:3, :].reshape((6))
            EN_imgs.append(face_flipped)
            EN_landmarks.append(landmark_flipped)

        en_face = cv2.resize(en_face, (31, 39)).reshape((1, 31, 39))
        en_landmark = landmarkGt[:3, :].reshape((6))
        EN_imgs.append(en_face)
        EN_landmarks.append(en_landmark)

        # NM
        # nm_bbox = bbox.subBBox(-0.05, 1.05, 0.18, 1.05)
        # nm_face = img[nm_bbox.top:nm_bbox.bottom+1,nm_bbox.left:nm_bbox.right+1]

        ## data argument
        if argument and np.random.rand() > 0.5:
            ### flip
            face_flipped, landmark_flipped = flip(nm_face, landmarkGt)
            face_flipped = cv2.resize(face_flipped, (31, 39)).reshape((1, 31, 39))
            landmark_flipped = landmark_flipped[2:, :].reshape((6))
            NM_imgs.append(face_flipped)
            NM_landmarks.append(landmark_flipped)

        nm_face = cv2.resize(nm_face, (31, 39)).reshape((1, 31, 39))
        nm_landmark = landmarkGt[2:, :].reshape((6))
        NM_imgs.append(nm_face)
        NM_landmarks.append(nm_landmark)

    #imgs, landmarks = process_images(ftxt, output)

    F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
    EN_imgs, EN_landmarks = np.asarray(EN_imgs), np.asarray(EN_landmarks)
    NM_imgs, NM_landmarks = np.asarray(NM_imgs),np.asarray(NM_landmarks)

    F_imgs = processImage(F_imgs)
    shuffle_in_unison_scary(F_imgs, F_landmarks)
    EN_imgs = processImage(EN_imgs)
    shuffle_in_unison_scary(EN_imgs, EN_landmarks)
    NM_imgs = processImage(NM_imgs)
    shuffle_in_unison_scary(NM_imgs, NM_landmarks)

    # full face
    base = join(OUTPUT, '1_F')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = F_imgs.astype(np.float32)
        h5['landmark'] = F_landmarks.astype(np.float32)

    # eye and nose
    base = join(OUTPUT, '1_EN')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = EN_imgs.astype(np.float32)
        h5['landmark'] = EN_landmarks.astype(np.float32)

    # nose and mouth
    base = join(OUTPUT, '1_NM')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)
    with h5py.File(output, 'w') as h5:
        h5['data'] = NM_imgs.astype(np.float32)
        h5['landmark'] = NM_landmarks.astype(np.float32)


if __name__ == '__main__':
    # train data
    train_txt = join(TRAIN, 'trainImageList.txt')
    generate_hdf5(train_txt, OUTPUT, 'train.h5', argument=True)

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
