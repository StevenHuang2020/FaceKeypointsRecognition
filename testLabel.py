#python3 steven
#10/04/2020 test dataset
import sys,os
import cv2
import numpy as np 
import argparse

from genLabel import getLabelFileLabels

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help = 'image')
    parser.add_argument('-l', '--label', help = 'iamge pts label file')
    parser.add_argument('-s', '--save', action="store_true", help = 'save label image')
    
    return parser.parse_args()
     
def showimage(img,str='image',autoSize=False):
    flag = cv2.WINDOW_NORMAL
    if autoSize:
        flag = cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(str, flag)
    cv2.imshow(str,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def writeImg(img,filePath):
    cv2.imwrite(filePath,img)

def drawPointIndexImg(img, i,pt):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org =(int(pt[0]), int(pt[1]))
    fontScale = 1
    color = (255, 0, 0) 
    thickness = 1
    return cv2.putText(img, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA) 

def drawPointImg(img, pt):
    return cv2.circle(img, (int(pt[0]),int(pt[1])), radius=2, color=(0, 0, 255), thickness=-1)

def drawPointImgFromPts(img,pts):
    for i,pt in enumerate(pts):
        img = drawPointImg(img,pt)

        # if i%2:
        #     img = drawPointIndexImg(img,i,pt)
    return img
 
def loadImg(file,mode=cv2.IMREAD_COLOR):
    #mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    return cv2.imread(file,mode)

def getImgHW(img):
    return img.shape[0],img.shape[1]

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

def testFaceLabel(file,label,locCod=False): #locCod coordinats or ratio
    img = loadImg(file)
    H,W = getImgHW(img)
    pts = getLabelFileLabels(label)

    if not locCod:
        for pt in pts:
            pt[0] = pt[0]*W
            pt[1] = pt[1]*H
    
    print(file,H,W,len(pts))
    img = drawPointImgFromPts(img,pts)
    return img
    
def testFromDrawpt():
    if 0:
        newBase = r'.\db\train\\'    
        file = newBase + 'images\\' + '001A02.jpg'
        label = newBase + 'labels\\' + '001A02.pts'
    else:
        # file = r'.\afterTraining\res\\' + 'myface_gray.png'
        # label = r'.\afterTraining\res\\' + 'myface_gray.pts'
        file = r'.\afterTraining\res\\' + '001A29.jpg'
        label = r'.\afterTraining\res\\' + '001A29.pts' # 
        
    print(file)
    print(label)
    showimage(testFaceLabel(file,label,False))
    
def main():
    return testFromDrawpt()

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