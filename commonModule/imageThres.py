#python3
#Steven image threshold modoule
import cv2.cv2 as cv2 #pip install opencv-python
import matplotlib.pyplot as plt
from ImageBase import *
from mainImagePlot import plotImagList

def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag) #cv2.WINDOW_NORMAL)
    cv2.imshow(str,img)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    file = './res/Lenna.png' #r'./res/obama.jpg'#
    img = loadImg(file,mode=cv2.IMREAD_GRAYSCALE) # IMREAD_GRAYSCALE IMREAD_COLOR
    #infoImg()
    img1 = binaryImage2(img,thresHMin=50,thresHMax=150)
    img2 = binaryImage2(img,thresHMin=50,thresHMax=100)
    img3 = thresHoldImage(img,mode = cv2.THRESH_BINARY)
    img4 = OtsuMethodThresHold(img)
    img5 = thresHoldModel(img,mode = cv2.ADAPTIVE_THRESH_MEAN_C)
    img6 = thresHoldModel(img,mode = cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    imgList = []
    nameList = []
    imgList.append(img), nameList.append('Original')
    imgList.append(img1), nameList.append('thrHMin')
    imgList.append(img2), nameList.append('thrHMin')
    imgList.append(img3), nameList.append('thrImg_Binary')
    imgList.append(img4), nameList.append('OtsuMethod')
    imgList.append(img5), nameList.append('thr_Mean')
    imgList.append(img6), nameList.append('thr_Gaussian')
    plotImagList(imgList,nameList) 

    
if __name__=='__main__':
    main()
