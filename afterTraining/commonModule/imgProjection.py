#python3
#Steven projection black and whight grayscale image
import cv2
import numpy as np
from common import getImagChannel
from ImageBase import *
from imagePlot import plotImagList


def projectionHorizonal(img):
    H,W =getImgHW(img)
    resImg = np.zeros([H,W], np.uint8)+255
    for i in range(H):
        numOfBlack = np.where(img[i]==0,1,0).sum()
        #print(numOfBlack)
        resImg[i][:numOfBlack] = 0
    return resImg
    
def projectionVertical(img):
    H,W = getImgHW(img)
    resImg = np.zeros([H,W], np.uint8)+255

    for col in range(W):
        numOfBlack = np.where(img[:,col]==0,1,0).sum()
        resImg[H-numOfBlack:H,col] = 0
    return resImg

def main():
    file = r'../res/myface_gray.png' #r'./res/obama.jpg'
    img = loadImg(file,mode=cv2.IMREAD_COLOR)
    imgGray = grayImg(img)
    bImg = binaryImage(imgGray, thresH=110)

    #plotImagList([img,imgGray,bImg])
    pHImg = projectionHorizonal(bImg)
    pVImg = projectionVertical(bImg)
    plotImagList([img,imgGray,bImg,pHImg,pVImg],['Original','gray','bimage','hProject','vProject'])
    #plotImagList([img,imgGray,bImg,pHImg])

if __name__=="__main__":
    main()
