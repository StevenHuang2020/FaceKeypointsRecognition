#python3
#Steven image histgram display modoule
import cv2.cv2 as cv2 #pip install opencv-python
import matplotlib.pyplot as plt

from imageBase import ImageBase
from matplotHist import plotHistImg, getImgHist, getImgHist256Img

def plotImagHistListImgHist256(imgList):
    nImg = len(imgList)
    nColumn = 3
    for n in range(nImg):
        img = imgList[n]
        plt.subplot(nImg, nColumn, n*nColumn + 1) #raw img    
        plt.imshow(img)

        plt.subplot(nImg, nColumn, n*nColumn + 2) #hist of raw img   
        plotImgHist(img)

        plt.subplot(nImg, nColumn, n*nColumn + 3) #hist256 of raw img   
        plotImgHist256Img(img)

    #plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotImagHistListImg(imgList):
    nImg = len(imgList)
    nColumn = 2
    for n in range(nImg):
        img = imgList[n]
        plt.subplot(nImg, nColumn, n*nColumn + 1) #raw img    
        plt.imshow(img)

        plt.subplot(nImg, nColumn, n*nColumn + 2) #hist of raw img   
        plotImgHist(img)

    #plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotImgHist(img):
    color = ('b','g','r')
    hists = getImgHist(img)
    cn = 0
    for i in hists:
        plt.plot(i,color = color[cn])
        plt.xlim([0,256])
        cn+=1

def plotImgHist256Img(img):
    #color = ('b','g','r')
    hist256Img = getImgHist256Img(img)
    for i in hist256Img:
        plt.imshow(i)
        break

def plotImagAndHist4(img):  #must color img
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgList=[]
    imgList.append(img)
    imgList.append(imgGray)
    return plotImagHistListImg(imgList)

def plotImagAndHist(img):
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])     # switch it to rgb

    plt.subplot(1, 2, 1)    
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plotImgHist(img)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag) #cv2.WINDOW_NORMAL)
    cv2.imshow(str,img)
   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    file = r'./res/obama.jpg'#'./res/Lenna.png' #
    img = ImageBase(file,mode=cv2.IMREAD_COLOR) # IMREAD_GRAYSCALE IMREAD_COLOR
    print(img.infoImg())
    #img.showimage(autoSize=False)
    #img.plotImg()
    #showimage(img.calcAndDrawHist())
    #showimage(img.thresHoldImage())
    #img.showimage()
    
    #plotHistImg(img.image)
    #plotImagAndHist(img.image)
    #plotImagAndHist4(img.image)
    
    imgList=[]
    imgGray = cv2.cvtColor(img.image, cv2.COLOR_BGR2GRAY)
    imgList.append(img.image)
    imgList.append(imgGray)
    imgList.append(img.equalizedHist(imgGray))

    #plotImagHistListImg(imgList)
    plotImagHistListImgHist256(imgList)

if __name__=='__main__':
    main()
