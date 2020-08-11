import sys,os
import cv2
import numpy as np 
import pandas as pd
from genLabel import loadImg,getImgHW,getFileName,getLabelFileLabels,pathsFiles,writeAnotationFile

def distanceAB(a,b):
    return np.sqrt(np.sum((a-b)**2))
    #return np.sqrt(np.sum(np.abs(a-b)))

def calculateRatio(pts,id1,id2,id3,id4):
    assert(id1<len(pts))
    assert(id2<len(pts))
    assert(id3<len(pts))
    assert(id4<len(pts))
    return distanceAB(pts[id1], pts[id2])/distanceAB(pts[id3],pts[id4])

def UpdatePtsToLocation(pts,H,W):
    for i in range(len(pts)):
        #print(pts[i][0],pts[i][1])
        pts[i][0] = pts[i][0]*W
        pts[i][1] = pts[i][1]*H
    return pts

def calculateFeature(pts,H,W,file=r'./res/test.pts'):
    pts = UpdatePtsToLocation(pts,H,W)
    
    #print(pts)
    
    #writeAnotationFile(file,pts)
    
    pts = np.array(pts)
    #print(pts.shape)
    
    # print('pts24=',pts[24])
    # print('pts18=',pts[18])
    # print(pts[24][0],pts[18][0])

    x = np.min([pts[24][0],pts[18][0]])
    y = np.min([pts[24][1],pts[18][1]])
    eyebrowCenter = np.array([x,y]) + np.abs(pts[24]-pts[18])/2
    #print('eyebrowCenter=',eyebrowCenter)
    
    #print(distanceAB(3,4))
    #print(distanceAB(np.array([1,2]),np.array([4,6])))
    #print(distanceAB(np.array([4,6]),np.array([1,2])))
    F_Facial_Index = distanceAB(eyebrowCenter, pts[7])/distanceAB(pts[14],pts[0])
    F_Mandibular_Index = calculateRatio(pts,66,7,4,10)
    
    F_Intercanthal = calculateRatio(pts,37,45,32,27)
    F_OrbitalWidth = calculateRatio(pts,27,29,37,45)
    F_EyeFissure  = calculateRatio(pts,28,30,27,29)
    F_VermilionHeight = calculateRatio(pts,51,66,66,57)
    F_MouthFaceWidth = calculateRatio(pts,48,54,0,14)
    F_Nose1 = calculateRatio(pts,39,46,39,41)
    F_Nose2 = calculateRatio(pts,39,40,39,41)
    F_Nose3 = calculateRatio(pts,39,38,39,41)
    F_Nose4 = calculateRatio(pts,39,38,67,41)
    F_Nose5 = calculateRatio(pts,39,41,67,41)
    #print('name:',name,'Features=',F_Facial_Index,F_Mandibular_Index,F_Intercanthal,F_OrbitalWidth,F_EyeFissure,F_VermilionHeight,F_MouthFaceWidth,F_Nose1,F_Nose2,F_Nose3,F_Nose4,F_Nose5)
    
    return [F_Facial_Index,F_Mandibular_Index,F_Intercanthal,F_OrbitalWidth,F_EyeFissure,F_VermilionHeight,F_MouthFaceWidth,F_Nose1,F_Nose2,F_Nose3,F_Nose4,F_Nose5]


dbFile = r'.\db\facial.csv'
def makeDb():
    base = r'.\db\train\\'
    base = os.path.abspath(base)
    imgPath = base + r'\images'
    LabelPath = base + r'\labels'
    
    imgPath = r'E:\opencv\project\facialRecognition\db\recognitionDb'
    df = pd.DataFrame()
    #columns = ['F_Facial_Index', 'F_Mandibular_Index', 'F_Intercanthal','F_OrbitalWidth', 'F_EyeFissure', 'F_VermilionHeight','F_MouthFaceWidth', 'F_Nose1', 'F_Nose2', 'F_Nose3', 'F_Nose4','F_Nose5']
    columns = ['F_Facial_Index', 'F_Mandibular_Index', 'F_Intercanthal','F_OrbitalWidth', 'F_EyeFissure', 'F_VermilionHeight','F_MouthFaceWidth']
    
    for i in pathsFiles(imgPath,'jpg'):
        #print(i)
        img = loadImg(i)
        H,W = getImgHW(img)
        fileName = getFileName(i)
        fileName = fileName[:fileName.rfind('.')]
        
        label = LabelPath + '\\' + fileName + '.pts'
        pts = getLabelFileLabels(label)
        
        pts = np.reshape(pts,(68,2))
        print(fileName,'label=', label,'pts=', len(pts),'H,W=',H,W)
        if 1:
            data = calculateFeature(pts,H,W)
            data = np.array(data).reshape(-1,len(data))
            line = pd.DataFrame(data,columns=columns)
        else:
            pts = pts.reshape(1,136)
            line = pd.DataFrame(pts)
            
        line.insert(0, "Id", fileName, True)
        
        df = df.append(line)
    
    df.set_index(["Id"], inplace=True)
    print(df.head())
    print(df.columns)
    df.to_csv(dbFile)

def getDb():
    return pd.read_csv(dbFile)

def main():
    makeDb()

if __name__ == '__main__':
    main()