# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:37:29 2019

@author: ASHUTOSH
"""

import cv2


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
# nose_cascade=cv2.CascadeClassifier('Nariz.xml')
# mouth_cascade=cv2.CascadeClassifier('Mouth.xml')

def genrate_dataset_s(img, userid, img_id):
    cv2.imwrite("data/usser."+str(userid)+"."+str(img_id)+".jpg",img)
def drawboundary_s(img,classifier,color,text,scalefactor,minneighbors):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features=classifier.detectMultiScale(gray_img,scalefactor,minneighbors)
    coords=[]
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)

        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.8,color,1,cv2.LINE_AA)
        coords=[x,y,w,h]
    return coords
def detect_s(img, faceCasscade,eyecascade,smilecascade,img_id):
    color={"B":(255,0,0),"R":(0,0,255),"G":(0,255,0)}
    coords=drawboundary_s(img,faceCasscade,color["B"],"FACE",1.1,10)
    if len(coords) == 4:
        roiimg=img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        userid=2
        genrate_dataset_s(roiimg, userid, img_id)
        coords_eye=drawboundary_s(roiimg, eyecascade, color["R"], "EYE", 1.1, 14)
        coords_smile=drawboundary_s(roiimg, smilecascade, color["G"], "SMILE", 1.13, 100)
        # coords_nose=drawboundary(roiimg, nosecascade, color["G"], "NOSE", 1.1,14)
        # coords_mouth=drawboundary(roiimg, mouthcascade, color["B"], "MOUTH", 1.1, 14)
    return img
vcap=cv2.VideoCapture(0)
img_id=0
while True:
    rval,img=vcap.read()
    img=detect_s(img,faceCascade,eye_cascade,smile_cascade,img_id)
    img_id+=1
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

vcap.release()
cv2.destroyAllWindows()
