# coding: utf-8
"""
    functions
"""

import os
import cv2
import numpy as np


def show_landmark(face, landmark):
    """
        view face with landmark for visualization
    """
    face_copied = face.copy().astype(np.uint8)
    for (x, y) in landmark:
        xx = int(face.shape[0]*x)
        yy = int(face.shape[1]*y)
        cv2.circle(face_copied, (xx, yy), 2, (0,0,0), -1)
    cv2.imshow("face_rot", face_copied)
    cv2.waitKey(0)


def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated image, bbox, landmark(relative to bbox)
    """
    center = ((bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, img.shape)
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    for idx, one in enumerate(landmark_):
        rv = ((one[0]-bbox[0])/(bbox[1]-bbox[0]), (one[1]-bbox[2])/(bbox[3]-bbox[2]))
        landmark_[idx] = rv
    face = img_rotated_by_alpha[bbox[2]:bbox[3]+1,bbox[0]:bbox[1]+1]
    return (face, landmark_)


def flip(face, landmark):
    """
        flip face
    """
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return (face_flipped_by_x, landmark_)
