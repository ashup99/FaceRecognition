# -*- coding: utf-8 -*
"""
Created on Wed Jul 31 09:22:49 2019

@author: ASHUTOSH
"""

import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32, minNeighbors=5)
    
    return faces,gray_img