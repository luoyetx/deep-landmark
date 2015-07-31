# coding: utf-8

import os
from os.path import join
import cv2
import caffe
import numpy as np
from config import PROJECT_ROOT, MEDIA_ROOT


VERSION = 'v0.0.6'


class CNN(object):
    """
        Generalized CNN for simple run forward with given Model
    """

    def __init__(self, net, model):
        self.net = net
        self.model = model
        self.cnn = caffe.Net(net, model, caffe.TEST) # failed if not exists

    def forward(self, data, layer='fc2'):
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2*i], x[2*i+1]]) for i in range(len(x)/2)])
        result = t(result)
        return result


class BBox(object):
    """
        Bounding Box of face
    """
    def __init__(self, bbox):
        self.left = int(bbox[0])
        self.right = int(bbox[1])
        self.top = int(bbox[2])
        self.bottom = int(bbox[3])
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

    def cropImage(self, img):
        """
            crop img with left,right,top,bottom
            **Make Sure is not out of box**
        """
        return img[self.top:self.bottom+1, self.left:self.right+1]


class Landmarker(object):
    """
        class Landmarker wrapper functions for predicting facial landmarks
    """

    def __init__(self):
        """
            Initialize Landmarker with files under VERSION
        """
        model_path = join(PROJECT_ROOT, VERSION)
        CNN_TYPES = ['LE1', 'RE1', 'N1', 'LM1', 'RM1', 'LE2', 'RE2', 'N2', 'LM2', 'RM2']
        level1 = [(join(model_path, '1_F.prototxt'), join(model_path, '1_F.caffemodel'))]
        level2 = [(join(model_path, '2_%s.prototxt'%name), join(model_path, '2_%s.caffemodel'%name)) \
                    for name in CNN_TYPES]
        level3 = [(join(model_path, '3_%s.prototxt'%name), join(model_path, '3_%s.caffemodel'%name)) \
                    for name in CNN_TYPES]
        self.level1 = [CNN(p, m) for p, m in level1]
        self.level2 = [CNN(p, m) for p, m in level2]
        self.level3 = [CNN(p, m) for p, m in level3]

    def detectLandmark(self, image, bbox, mode='fast'):
        """
            Predict landmarks for face with bbox in image
            fast mode will only apply level-1 and level-2
        """
        if not isinstance(bbox, BBox) or image is None:
            return None, False
        face = bbox.cropImage(image)
        face = cv2.resize(face, (39, 39)).reshape((1, 1, 39, 39))
        face = self._processImage(face)
        # level-1, only F in implemented
        landmark = self.level1[0].forward(face)
        # level-2
        landmark = self._level(image, bbox, landmark, self.level2, [0.16, 0.18])
        if mode == 'fast':
            return landmark, True
        landmark = self._level(image, bbox, landmark, self.level3, [0.11, 0.12])

    def _level(self, img, bbox, landmark, cnns, padding):
        """
            LEVEL-?
        """
        for i in range(5):
            x, y = landmark[i]
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[0])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d1 = cnns[i].forward(patch) # size = 1x2
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[1])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d2 = cnns[i+5].forward(patch)

            d1 = bbox.project(patch_bbox.reproject(d1[0]))
            d2 = bbox.project(patch_bbox.reproject(d2[0]))
            landmark[i] = (d1 + d2) / 2
        return landmark

    def _getPatch(self, img, bbox, point, padding):
        """
            Get a patch iamge around the given point in bbox with padding
            point: relative_point in [0, 1] in bbox
        """
        point_x = bbox.x + point[0] * bbox.w
        point_y = bbox.y + point[1] * bbox.h
        patch_left = point_x - bbox.w * padding
        patch_right = point_x + bbox.w * padding
        patch_top = point_y - bbox.h * padding
        patch_bottom = point_y + bbox.h * padding
        patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
        patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
        return patch, patch_bbox

    def _processImage(self, imgs):
        """
            process images before feeding to CNNs
            imgs: N x 1 x W x H
        """
        imgs = imgs.astype(np.float32)
        for i, img in enumerate(imgs):
            m = img.mean()
            s = img.std()
            imgs[i] = (img - m) / s
        return imgs

class FaceDetector(object):
    """
        class FaceDetector use VJ detector
    """

    def __init__(self):
        self.cc = cv2.CascadeClassifier(join(VERSION, 'haarcascade_frontalface_alt.xml'))

    def detectFace(self, img):
        rects = self.cc.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, \
                    minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        for rect in rects:
            rect[2:] += rect[:2]
            yield BBox([rect[0], rect[2], rect[1], rect[3]])


def drawLandmark(img, bbox, landmark):
    cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 1, (0,255,0), -1)
    return img

fd = FaceDetector()
fl = Landmarker()

def detectLandmarks(src, dst):
    """
        detect landmarks in `src` and store the result in `dst`
    """
    img = cv2.imread(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bboxes = []
    landmarks = []
    for bbox in fd.detectFace(gray):
        bbox = bbox.subBBox(0.1, 0.9, 0.2, 1)
        bboxes.append(bbox)
        landmark, status = fl.detectLandmark(gray, bbox)
        landmarks.append(bbox.reprojectLandmark(landmark))

    for i in range(len(bboxes)):
        img = drawLandmark(img, bboxes[i], landmarks[i])

    cv2.imwrite(dst, img)
