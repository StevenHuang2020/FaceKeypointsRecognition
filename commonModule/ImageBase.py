#python3
#Steven Image base operation Class
import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

def changeBgr2Rbg(img): #input color img
    b,g,r = cv2.split(img)       # get b,g,r
    img = cv2.merge([r,g,b])
    return img

def loadImg(file,mode=cv2.IMREAD_COLOR):
    #mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    img = cv2.imread(file,mode)
    #if img == None:
    #    print("Load image error,file=",file)
    return img

def writeImg(img,filePath):
    cv2.imwrite(filePath,img)

def infoImg(img,str='image:'):
    return(str,'shape:',img.shape,'size:',img.size,'dims=',img.ndim,'dtype:',img.dtype)

def grayImg(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def getImagChannel(img):
    if img.ndim == 3: #color r g b channel
        return 3
    return 1  #only one channel

def resizeImg(img,NewW,NewH):
    rimg = cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA
    return rimg

def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag)
    cv2.imshow(str,img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def plotImg(img):
    plt.imshow(img)
    #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
    plt.show()

def getImgHW(img):
    return img.shape[0],img.shape[1]


"""-----------------------operation start-------"""
def calcAndDrawHist(img,color=[255,255,255]): #color histgram
    hist= cv2.calcHist([img], [0], None, [256], [0.0,255.0])
    #print(hist)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8) #[256,256,3] [256,256]
    hpt = int(0.9* 256)

    for h in range(256):
        intensity = int(hist[h]*hpt/maxVal)
        #print(h,intensity)
        cv2.line(histImg,(h,256), (h,256-intensity), color)
    return histImg

def equalizedHist(img):
    return cv2.equalizeHist(img)

def binaryImage(img,thresH):
    """img must be gray"""
    H, W = getImgHW(img)
    newImage = img.copy()
    for i in range(H):
        for j in range(W):
            #print(newImage[i,j])
            if newImage[i,j] < thresH:
                newImage[i,j] = 0
            else:
                newImage[i,j] = 255

    return newImage

def binaryImage2(img,thresHMin=0,thresHMax=0):
    """img must be gray"""
    H, W = getImgHW(img)
    newImage = img.copy()
    for i in range(H):
        for j in range(W):
            #print(newImage[i,j])
            if newImage[i,j] < thresHMin:
                newImage[i,j] = 0
            if newImage[i,j] > thresHMax:
                newImage[i,j] = 255

    return newImage

def thresHoldImage(img,thres=127,mode=cv2.THRESH_BINARY):
    #mode = cv2.THRESH_BINARY cv2.THRESH_BINARY_INV
    #cv2.THRESH_TRUNC cv2.THRESH_TOZERO_INV
    _, threshold = cv2.threshold(img,thres,255,mode)
    return threshold

def OtsuMethodThresHold(img):
    _, threshold = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold
    
def thresHoldModel(img,mode=cv2.ADAPTIVE_THRESH_MEAN_C):
    #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    return cv2.adaptiveThreshold(img,255,mode,cv2.THRESH_BINARY,11,2)

def convolutionImg(img,kernel):
    return cv2.filter2D(img,-1,kernel)
    
"""-----------------------operation end---------"""






class ImageBase:
    def __init__(self,file,mode=cv2.IMREAD_COLOR):
        self.file = file
        try:
            self.image = self.loadImg(file,mode)
        except:
            print("Load image error!")

        print(self.image.shape)
        #assert self.image != None #"Load image error!"
        
        """#change b g r channel order avoid color not correct when plot"""
        if mode == cv2.IMREAD_COLOR:  
            b,g,r = cv2.split(self.image)       # get b,g,r
            self.image = cv2.merge([r,g,b])     # switch it to rgb

    def writeImg(self,img,filePath):
        cv2.imwrite(filePath,img)

    def loadImg(self,filename,mode=cv2.IMREAD_COLOR):
        #mode = cv2.IMREAD_COLOR
        #mode = cv2.IMREAD_GRAYSCALE
        #mode = cv2.IMREAD_UNCHANGED
        return cv2.imread(filename,mode)

    
    def infoImg(self,str='image:'):
        return(str,'shape:',self.image.shape,'size:',self.image.size,'dtype:','dims=',self.image.ndim,self.image.dtype)
    
    def getImagChannel(self):
        if self.image.ndim == 3: #color r g b channel
            return 3
        return 1  #only one channel

    def showimage(self,str='image',autoSize=False):
        flag = cv2.WINDOW_NORMAL
        if autoSize:
            flag = cv2.WINDOW_AUTOSIZE

        cv2.namedWindow(str, flag)
        cv2.imshow(str,self.image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def plotImg(self):
        plt.imshow(self.image)
        #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
        plt.show()

    def getImgHW(self):
        return self.image.shape[0],self.image.shape[1]

    def calcAndDrawHist(self, color=[255,255,255]): #color histgram
        hist= cv2.calcHist([self.image], [0], None, [256], [0.0,255.0])
        #print(hist)

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
        histImg = np.zeros([256,256,3], np.uint8) #[256,256,3] [256,256]
        hpt = int(0.9* 256)

        for h in range(256):
            intensity = int(hist[h]*hpt/maxVal)
            #print(h,intensity)
            cv2.line(histImg,(h,256), (h,256-intensity), color)
        return histImg

    def equalizedHist(self,img=None):
        if img.all() == None:
            return cv2.equalizeHist(self.image.copy())
        else:
            return cv2.equalizeHist(img.copy())

    """----------operation------start----------"""
    def binaryImage(self,img,thresH):
        """img must be gray"""
        H, W = self.getImgHW()
        newImage = self.image.copy()
        for i in range(H):
            for j in range(W):
                #print(newImage[i,j])
                if newImage[i,j] < thresHMin:
                    newImage[i,j] = 0
                if newImage[i,j] > thresHMax:
                    newImage[i,j] = 255

        return newImage

    def binaryImage2(self,thresHMin=0,thresHMax=0):
        """img must be gray"""
        H, W = self.getImgHW()
        newImage = self.image.copy()
        for i in range(H):
            for j in range(W):
                #print(newImage[i,j])
                if newImage[i,j] < thresHMin:
                    newImage[i,j] = 0
                if newImage[i,j] > thresHMax:
                    newImage[i,j] = 255

        return newImage
    
    def thresHoldImage(self,thres=127,mode=cv2.THRESH_BINARY):
        newImage = self.image.copy()
        #cv2.THRESH_BINARY
        #cv2.THRESH_BINARY_INV
        #cv2.THRESH_TRUNC
        #cv2.THRESH_TOZERO_INV
        _, threshold = cv2.threshold(newImage,thres,255,mode)
        return threshold
    
    def OtsuMethodThresHold(self):
        newImage = self.image.copy()
        _, threshold = cv2.threshold(newImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(img,(5,5),0)
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return threshold
        
    def thresHoldModel(self,mode=cv2.ADAPTIVE_THRESH_MEAN_C):
        newImage = self.image.copy()
        #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        return cv2.adaptiveThreshold(newImage,255,mode,cv2.THRESH_BINARY,11,2)
    
    
            

    """----------operation------end------------"""


    """----------operation------start----------"""
    """----------operation------start----------"""
    pass
