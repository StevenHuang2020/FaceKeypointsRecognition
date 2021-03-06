#python3 steven
#10/04/2020 test dataset
import sys,os
import cv2
import numpy as np 
import argparse
import os,sys
#print(sys.path)
sys.path.append(r'.\afterTraining\\')

from genLabel import getLabelFileLabels
from predictKeyPoints import preditImg
from commonModule.imagePlot import plotImagList
from commonModule.ImageBase import changeBgr2Rbg,grayImg,loadImg,getImgHW,writeImg,showimage
from makeDB import UpdatePtsToLocation

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help = 'image')
    parser.add_argument('-l', '--label', help = 'iamge pts label file')
    parser.add_argument('-s', '--save', action="store_true", help = 'save label image')
    return parser.parse_args()

def drawPointIndexImg(img, i,pt,color = (255, 0, 0),fontsize=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org =(int(pt[0]), int(pt[1]))
    fontScale = fontsize
    thickness = 1
    return cv2.putText(img, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA) 

def drawPointImg(img, pt,color=(0, 0, 255)):
    return cv2.circle(img, (int(pt[0]),int(pt[1])), radius=2, color=color, thickness=-1)

def drawPointImgFromPts(img,pts,color):
    for i,pt in enumerate(pts):
        img = drawPointImg(img,pt,color=color)

        #img = drawPointIndexImg(img,i,pt,fontsize=0.4)
        if 0:
            if i<27:
                img = drawPointIndexImg(img,i,pt,fontsize=0.5)
            elif i<48:
                img = drawPointIndexImg(img,i-27,pt,color=(0,255,0),fontsize=0.5)
            else:
                img = drawPointIndexImg(img,i-47,pt,color=(0,255,255),fontsize=0.5)
    return img


'''
def getLabelFileLabels(fileLabel):
    numbers = {'1','2','3','4','5','6','7','8','9'}
    labels = []
    with open(fileLabel,'r') as srcF:        
        for i in srcF.readlines():
            if i[0] in numbers:
                #print(i)
                pts = i.split(' ')
                #print(pts)
                labels.append((float(pts[0]),float(pts[1])))
    return labels
'''

def testFaceLabelPredict(file):
    img = loadImg(file)
    #img=grayImg(img)
    
    pts = preditImg(img)
    print(type(pts),pts)
    #pts = pts.reshape((68,2))
    return testFaceLabelPts(img,pts)

def testFaceLabel(file,label,color=(0,0,255),locCod=False,gray=False): #locCod coordinats or ratio
    img = loadImg(file)
    if gray:
        img=grayImg(img)
    return testFaceLabelImg(img,label,color=color,locCod=locCod)

def testFaceLabelImg(img,label,color,locCod=False): #locCod coordinats or ratio
    pts = getLabelFileLabels(label)
    pts = np.reshape(pts, (68,2))
    return testFaceLabelPts(img,pts,color=color,locCod=locCod)

def testFaceLabelPts(img,pts,color=(0,0,255),locCod=False):
    #print('test pts pts.shape=',type(pts), pts.shape)
    H,W = getImgHW(img)
    
    if not locCod:
        pts = UpdatePtsToLocation(pts,H,W)

    #print(H,W,len(pts))
    img = drawPointImgFromPts(img,pts,color)
    return img

def showPredictImage():
    file1 = r'.\res\\' + '001A29.jpg'
    label1 = r'.\res\\' + '001A29.pts' 
    file2 = r'.\res\\' + '003A25.jpg'
    label2 = r'.\res\\' + '003A25.pts' 
    
    predictImg1 = testFaceLabelPredict(file1)
    trueImg1 = testFaceLabel(file1,label1,locCod=False)
    c1Img = testFaceLabelImg(predictImg1.copy(),label1,locCod=False,color=(255,0,0))
    c1Img = changeBgr2Rbg(c1Img)
    
    predictImg2 = testFaceLabelPredict(file2)
    trueImg2 = testFaceLabel(file2,label2,locCod=False)
    c2Img = testFaceLabelImg(predictImg2.copy(),label2,locCod=False,color=(255,0,0))
    c2Img = changeBgr2Rbg(c2Img)
    
    imgList,nameList = [],[]
    imgList.append(c1Img),nameList.append('c1Img')
    imgList.append(c2Img),nameList.append('c2Img')
    plotImagList(imgList,nameList,showticks=False,showTitle=False)
    
def showPredictOutdatasetImage():
    file1 = r'.\res\\' + 'anna.png'
    file2 = r'.\res\\' + 'trump.png' #obama.png
    
    predictImg1 = testFaceLabelPredict(file1)   
    predictImg2 = testFaceLabelPredict(file2)
    predictImg1 = changeBgr2Rbg(predictImg1)
    predictImg2 = changeBgr2Rbg(predictImg2)
    
    imgList,nameList = [],[]
    imgList.append(predictImg1),nameList.append('predictImg1')
    imgList.append(predictImg2),nameList.append('predictImg2')
    plotImagList(imgList,nameList,showticks=False,showTitle=False)
    
def testFromDrawpt():
    if 0:
        newBase = r'.\db\train\\'    
        file = newBase + 'images\\' + '006A31.jpg'
        label = newBase + 'labels\\' + '006A31.pts'
    else:
        # file = r'.\res\\' + 'myface_gray.png'
        # label = r'.\res\\' + 'myface_gray.pts'
        file = r'.\res\\' + '001A29.jpg'
        label = r'.\res\\' + '001A29.pts' # 
        labelTrue = r'.\res\\' + '001A29_True.pts'
        
    print(file)
    print(label)
    
    #showimage(testFaceLabel(file,label,False))
    predictImg = testFaceLabelPredict(file)
    trueImg = testFaceLabel(file,labelTrue,locCod=True)
    c1Img = testFaceLabelImg(predictImg.copy(),labelTrue,locCod=True,color=(255,0,0))
        
    predictImg = changeBgr2Rbg(predictImg)
    trueImg = changeBgr2Rbg(trueImg)
    c1Img = changeBgr2Rbg(c1Img) #true and predict overlap
    
    imgList,nameList = [],[]
    imgList.append(predictImg),nameList.append('Predict')
    imgList.append(trueImg),nameList.append('trueImg')
    #imgList.append(c1Img),nameList.append('c1Img')
    plotImagList(imgList,nameList,showticks=False,showTitle=False)
        
    imgList,nameList = [],[]
    imgList.append(c1Img),nameList.append('c1Img')
    plotImagList(imgList,nameList,showticks=False,showTitle=False)
    #showimage(c1Img)
    #showimage(trueImg)
    
def drawPtsAndNumber():
    file = r'.\res\\' + '030A26.jpg'
    labelTrue = r'.\res\\' + '030A26.pts'
    
    file2 = r'.\res\\' + '082A31.jpg'
    labelTrue2 = r'.\res\\' + '082A31.pts'
    
    trueImg = testFaceLabel(file,labelTrue,locCod=False)
    trueImg = changeBgr2Rbg(trueImg)
 
    trueImg2 = testFaceLabel(file2,labelTrue2,locCod=False)
    trueImg2 = changeBgr2Rbg(trueImg2)

    #showimage(trueImg)
    imgList,nameList = [],[]
    imgList.append(trueImg),nameList.append('trueImg')
    imgList.append(trueImg2),nameList.append('trueImg2')
    plotImagList(imgList,nameList,showticks=False, showTitle=False)
    
def showRawPictures():
    file1 = r'.\res\\' + '006A31.jpg'
    label1 = r'.\res\\' + '006A31.pts' # 
    
    file2 = r'.\res\\' + '003A25.jpg'
    label2 = r'.\res\\' + '003A25.pts' # 

    img1 = loadImg(file1)
    trueImg1 = testFaceLabel(file1,label1,locCod=False)
    img2 = loadImg(file2)
    trueImg2 = testFaceLabel(file2,label2,locCod=False)
    
    img1 = changeBgr2Rbg(img1)
    trueImg1 = changeBgr2Rbg(trueImg1)
    img2 = changeBgr2Rbg(img2)
    trueImg2 = changeBgr2Rbg(trueImg2)
    
    imgList,nameList = [],[]
    imgList.append(img1),nameList.append('img1')
    imgList.append(img2),nameList.append('img2')
    imgList.append(trueImg1),nameList.append('trueImg1')
    imgList.append(trueImg2),nameList.append('trueImg2')
    plotImagList(imgList,nameList,showticks=False,showTitle=False)
    
    
def main():
    return showPredictOutdatasetImage()
    #return showRawPictures()
    return showPredictImage()
    #return drawPtsAndNumber()
    #return testFromDrawpt()

    arg = argCmdParse()
    file = arg.image  #r'./res/myface_gray.png'
    label = arg.label
    save = arg.save
    
    print('file=', file)
    print('label=', label)
    print('save=',save)
    
    img = testFaceLabel(file,label)
    showimage(img)
    
    if save:
        file = file[:file.rfind('.')]+'_label.png'
        print(file)
        writeImg(img,file)
    
    
if __name__ == '__main__':
    main()