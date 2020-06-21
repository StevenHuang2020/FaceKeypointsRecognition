#python3
#Steven face identification
#process: 1.preprecess 2. face position 3.feature extraction 4.face recognition
import os,sys
import cv2
import numpy as np

from CascadeClassifier import CascadeClassifier
from commonModule.ImageBase import *
from commonModule.imagePlot import plotImagList

newW=364 #must equal to the training cfg,refer to genLabel.py
newH=440
def resizeImg(img,NewW,NewH):
    h,w = getImgHW(img)
    #return cv2.resize(img, (int(h*ratio), int(w*ratio)), interpolation=cv2.INTER_CUBIC) #INTER_LANCZOS4
    return cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA

def CascadeDetect(cascPath=r'./res/haarcascade_frontalface_default.xml'):
    return CascadeClassifier(cascPath)

def main():
    cv2.useOptimized()
    file =  r'./res/obama.jpg'#r'./res/Lenna.png' #
    
    print('Number of parameter:', len(sys.argv))
    print('Parameters:', str(sys.argv))
    if len(sys.argv)>1:
        file = sys.argv[1]


    img = loadImg(file,mode=cv2.IMREAD_COLOR) # IMREAD_GRAYSCALE IMREAD_COLOR

    faceROI = CascadeDetect()
 
    faceR=faceROI.getDetectImg(img)
    face = faceROI.detecvFaceImgOne(img)
    faceGray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    ls,names = [],[]
    ls.append(img),names.append('orignal')
    ls.append(faceR),names.append('faceR')
    ls.append(face),names.append('face')
    ls.append(faceGray),names.append('faceGray')

    plotImagList(ls,names)

    #faceGray = resizeImg(faceGray,newW,newH)
    #img.writeImg(face,'./res/myface_.png')
    #writeImg(faceGray,'./res/myface_gray.png')

if __name__=="__main__":
    main()

