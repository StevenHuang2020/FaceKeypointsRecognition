#python3
#Steven matplot hisgram of image

import numpy as np
import cv2.cv2 as cv2
from matplotlib import pyplot as plt

def plotHist(file,mode=cv2.IMREAD_COLOR):
    img = cv2.imread(file,mode)
    return plotHistImg(img)
    
def getImagChannel(img):
    if img.ndim == 3: #color r g b channel
        return 3
    return 1  #only one channel

def plotHistImg(img):
    color = ('b','g','r')
    chn = getImagChannel(img)
    print(chn)
    for i in range(chn):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        print(type(histr),histr.shape)
        #print(histr)
        #print(histr.ravel())
        plt.plot(histr,color = color[i])
        #plt.hist(histr.ravel(), 255, facecolor='blue', alpha=0.5)
        plt.xlim([0,256])
    plt.show()

def getHist(file,mode=cv2.IMREAD_COLOR):
    img = cv2.imread(file,mode)
    return getImgHist(img)

def getImgHist(img):
    chn = getImagChannel(img)
    hists = []
    for i in range(chn):
        hists.append(cv2.calcHist([img],[i],None,[256],[0,256]))
    return hists

def getImgHist256Img(img):
    chn = getImagChannel(img)
    hist256Imgs = []
    for i in range(chn):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        hist256Imgs.append(getHist256ImgFromHist(hist))
    return hist256Imgs

def getHist256ImgFromHist(hist,color=[255,255,255]):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8) #[256,256,3] [256,256]
    hpt = int(0.9* 256)

    for h in range(256):
        #intensity = int(hist[h]*hpt/maxVal)
        intensity = hist[h]*hpt/maxVal
        #print(h,intensity)
        cv2.line(histImg,(h,256), (h,256-intensity), color)
    return histImg

def plotHistGray(file):
    img = cv2.imread(file,0)
    return plotHistGrayImg(img)

def plotHistGrayImg(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

def plotColorHist(file):
    img = cv2.imread(file)
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def main():
    file=r'./res/Lenna.png'
    #plotHistGray(file)
    #plotColorHist(file)
    plotHist(file,cv2.IMREAD_GRAYSCALE)
    pass

if __name__=='__main__':
    main()
