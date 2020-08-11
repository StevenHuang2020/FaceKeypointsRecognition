#Steven
import os
import cv2
import numpy as np 
from numpy import save,load

from genLabel import getLabelFileLabels,newW,newH

def loadImg(file,mode=cv2.IMREAD_COLOR):
    #mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    return cv2.imread(file,mode)

def getImgHW(img):
    return img.shape[0],img.shape[1]  

def getImgShape(img):
    return img.shape

def fileList(file):
    with open(file,'r') as srcF:        
        return srcF.readlines()

gStartIndex = 0
gBachSize=10
gLines = []

def getDataBatch(file,bachSize=10):
    global gStartIndex
    global gBachSize
    global gLines
    
    gBachSize = bachSize
    if len(gLines) == 0:
        gLines = fileList(file)
    size = len(gLines)
    print('len=',size,type(gLines))
    
    #trainX = np.empty([newH,newW],dtype=int)
    #trainY = np.empty([68,2])
    # trainX = np.empty([newH,newW],dtype=int)
    # trainY = np.empty([68,2])
    
    trainX = []
    trainY = []
    for _ in range(gStartIndex,gStartIndex+gBachSize):      
        i = gLines[gStartIndex].strip().split(',')
        #print('i0=',i[0])
        #print('i1=',i[1])#print(type(i),i)
        fImg = i[0]
        label=i[1]
        
        img = loadImg(fImg)
        H,W = getImgHW(img)
        print(H,W,type(img),getImgShape(img),img[:,:,0].shape)
        
        #trainX = np.concatenate((trainX, img[:,:,0]), axis=0)
        trainX.append(img[:,:,0])
        
        pts = getLabelFileLabels(label)
        pts = np.array(pts).flatten()
        #print(pts,pts.shape)
        #trainY = np.concatenate((trainY, pts), axis=0)
        trainY.append(pts)
        
        gStartIndex +=1
        gStartIndex = gStartIndex%size
        
    print('gStartIndex=',gStartIndex)
    return trainX,trainY

def getData(file):
    lines = fileList(file)
    size = len(lines)
    print('len=',size,type(lines))
        
    trainX = []
    trainY = []
    for _ in range(size):      
        i = lines[_].strip().split(',')
        #print('i0=',i[0])
        #print('i1=',i[1])#print(type(i),i)
        fImg = i[0]
        label=i[1]
        
        img = loadImg(fImg)
        H,W = getImgHW(img)
        #print(H,W,type(img),getImgShape(img),img[:,:,0].shape)
        
        #trainX = np.concatenate((trainX, img[:,:,0]), axis=0)
        trainX.append(img[:,:,0])
        
        pts = getLabelFileLabels(label)
        pts = np.array(pts).flatten()
        #print(pts,pts.shape)
        #trainY = np.concatenate((trainY, pts), axis=0)
        trainY.append(pts)
        
    trainX=np.asarray(trainX)
    trainY=np.asarray(trainY)
    return trainX,trainY

def saveDataset():
    file = r'.\db\train\trainList.list'
    print(newW,newH)
    x,y = getData(file)
    print(x.shape)
    print(y.shape)
    
    save('./db/dataImg.npy', x)
    save('./db/dataLabel.npy', y)
    return

def loadDataset():
    x = load(r'./db/dataImg.npy')
    y = load(r'./db/dataLabel.npy')
    
    print(x.shape)
    print(y.shape)
    return x,y

def main():
    #saveDataset()
    #loadDataset()
    pass

if __name__=='__main__':
    main()
    