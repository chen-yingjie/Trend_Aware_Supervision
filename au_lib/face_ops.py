from imutils import face_utils
import dlib
import cv2
import glob
import numpy as np
from sklearn import svm, manifold, decomposition, discriminant_analysis
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import matplotlib.pyplot as plt
from cv2 import WINDOW_NORMAL
import sys
import time
import os
import math
from PIL import Image
import threading
from sklearn.decomposition import FastICA
import xml.dom.minidom

shapePredictorPath = '../../../Dataset/shape_predictor_68_face_landmarks.dat'
# shapePredictorPath = '/mnt/hdd3t/data/au/shape_predictor_68_face_landmarks.dat'
faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)
faceDet = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

def get_face_size(image):
    global faceDetector
    faces = faceDetector(image, 1)
    if len(faces) == 0:
        return -1, -1
    face = faces[0]
    pos_start = tuple([face.left(), face.top()])
    pos_end = tuple([face.right(), face.bottom()])
    height = (face.bottom() - face.top())
    width = (face.right() - face.left())
    # cv2.rectangle(image, pos_start, pos_end, (0, 110, 125), 1)
    return height, width

def get_facelandmark(image):
    global faceDetector, facialLandmarkPredictor

    face = faceDetector(image, 1)
    if len(face) == 0:
        return None

    shape = facialLandmarkPredictor(image, face[0])
    facialLandmarks = face_utils.shape_to_np(shape)

    xyList = []
    for (x, y) in facialLandmarks[0:]:  # facialLandmarks[17:] without face contour
        xyList.append(x)
        xyList.append(y)
        
    return xyList

def find_faces(image, normalize=False, resize=None, gray=None):
    global faceDetector
    faces = faceDetector(image, 1)
    if len(faces) == 0:
        return None

    cutted_faces = [image[face.top():face.bottom(), face.left():face.right()] for face in faces]
    faces_coordinates = [(face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()) for face in faces]

    if normalize:
        if resize is None or gray is None:
            print("Error: resize & gray must be given while normalize is True.")
        normalized_faces = [_normalize_face(face, resize, gray) for face in cutted_faces]
    else:
        normalized_faces = cutted_faces
    return zip(normalized_faces, faces_coordinates)

def _normalize_face(face, resize=350, gray=True):
    if gray:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (resize, resize))
    return face


def draw_face_landmark(featureList, x, y, w, h, img, isContour=False):
    Xs = featureList[::2]
    Ys = featureList[1::2]
    x_nose = Xs[31 - 18] 
    y_nose = Ys[31 - 18] 
    
    for i in range(len(Xs)):
        if i < 17 and not isContour:
            continue
        xx = Xs[i]
        yy = Ys[i] 
        if i > 0 and i != 17 and i != 22 and i != 27 and i != 31 and i != 36 and i != 42 and i != 48 and i != 60: 
            xx_last = Xs[i - 1] 
            yy_last = Ys[i - 1] 
            if i == 41:
                cv2.line(img, (x + Xs[36], y + Ys[36]), (x + xx, y + yy), (0, 255, 0), 1)
            elif i == 47:
                cv2.line(img, (x + Xs[42], y + Ys[42]), (x + xx, y + yy), (0, 255, 0), 1)
            elif i == 59:
                cv2.line(img, (x + Xs[48], y + Ys[48]), (x + xx, y + yy), (0, 255, 0), 1)
            elif i == 67:
                cv2.line(img, (x + Xs[60], y + Ys[60]), (x + xx, y + yy), (0, 255, 0), 1)

            cv2.line(img, (x + xx_last, y + yy_last), (x + xx, y + yy), (0, 255, 0), 1)
        if i in range(len(Xs)):
            cv2.circle(img, (x + xx, y + yy), 2, (0, 255, 0), -1)
            # cv2.putText(img, str(i), (x + xx, y + yy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))
    return img

def nasolabial_folds(featureList, x, y, w, h, img):
    Xs = featureList[::2]
    Ys = featureList[1::2]
    x_nose = Xs[31 - 18]
    y_nose = Ys[31 - 18]
    
    # left nose - mouth
    blackimg = np.zeros(img.shape, dtype='uint8')
    for i in range(y + Ys[29], y + Ys[48]):
        for j in range(x + Xs[41], x + Xs[31]):
            blackimg[i, j, 0:3] = img[i, j, 0:3]
    # right nose - mouth
    for i in range(y + Ys[29], y + Ys[54]):
        for j in range(x + Xs[35], x + Xs[46]):
            blackimg[i, j, 0:3] = img[i, j, 0:3]
    # brows
    for i in range((y + Ys[27] - 2 * (Ys[27] - Ys[21])), y + Ys[27]):
        for j in range(x + Xs[21], x + Xs[22]):
            blackimg[i, j, 0:3] = img[i, j, 0:3]
    
    return blackimg
