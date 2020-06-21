#python3 steven
#10/04/2020 test dataset
import sys,os
import cv2
import numpy as np 

def deleteFile(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def pathsFiles(dir,filter=''): #"cpp h txt jpg"
    def getExtFile(file):
        return file[file.rfind('.')+1:]

    def getFmtFile(path):
        #/home/User/Desktop/file.txt    /home/User/Desktop/file     .txt
        root_ext = os.path.splitext(path) 
        return root_ext[1]

    fmts = filter.split()    
    if fmts:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if getExtFile(getFmtFile(filename)) in fmts:
                    yield dirpath+'\\'+filename
    else:
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                yield dirpath+'\\'+filename
                
def createPath(dirs):
    def deleteFolder(file_path):
        if os.path.exists(file_path):
            #shutil.rmtree(file_path)
            for lists in os.listdir(file_path):
                f = os.path.join(file_path, lists)
                if os.path.isfile(f):
                    os.remove(f)
    deleteFolder(dirs)            
    if not os.path.exists(dirs):
        os.makedirs(dirs)
   
def getFileName(path):  #full fileName
    return os.path.basename(path)

def getLabelFileLabels(fileLabel):
    numbers = {'0','1','2','3','4','5','6','7','8','9'}
    labels = []
    with open(fileLabel,'r') as srcF:        
        for i in srcF.readlines():
            if i[0] in numbers:
                #print(i)
                pts = i.split(' ')
                #print(pts)
                labels.append([float(pts[0]),float(pts[1])])
    return labels
    
def writeToDst(file,content):
        with open(file,'a',newline='\n') as dstF: #linux--'\n'
            dstF.write(content)
            
def writeAnotationFile(file,pts):
    deleteFile(file)
    for i in pts:
        a,b = i[0],i[1]
        #print(i,a,b, str(a)+ ' ' + str(b))
        writeToDst(file,str(a) + ' ' + str(b) + '\n')
    
def sourceDatasetOper():
    base = r'.\db\FGNET\\'
    imgPath = base + 'images\\'
    LabelPath = base + r'points\\'
    dstRecImgPath = base + r'test'
    
    dstPath = r'.\db\FGNET\labels\\'
    
    fList = base + 'trainList.list'
    deleteFile(fList)
    
    createPath(dstPath)
    for i in pathsFiles(imgPath,'JPG'):
        print(i)
        img = loadImg(i)
        
        fileName = getFileName(i)
        fileName = fileName[:fileName.rfind('.')] 
        
        #writeToDst(fList,i+'\n')
        writeToDst(fList,'.\images\\' + fileName + '.jpg' + '\n')
        
        label = LabelPath + '\\' + fileName + '.pts'
        pts = getLabelFileLabels(label)
        
        dstFile = dstPath + fileName + '.pts'
        print('dstFile=', dstFile)
        writeAnotationFile(dstFile,pts)
   
def loadImg(file,mode=cv2.IMREAD_COLOR):
    #mode = cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE cv2.IMREAD_UNCHANGED
    return cv2.imread(file,mode)

def getImgHW(img):
    return img.shape[0],img.shape[1]   
  
def writeImg(img,filePath):
    cv2.imwrite(filePath,img)
    
def resizeImg(img,NewW,NewH,pts):
    h,w = getImgHW(img)
    #return cv2.resize(img, (int(h*ratio), int(w*ratio)), interpolation=cv2.INTER_CUBIC) #INTER_LANCZOS4
    rimg = cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA
    
    newPts=[]
    for i in pts:
        x,y = i[0],i[1]
        #newPts.append((round(x*NewW/w,4),round(y*NewH/h,4)))
        if 0:
            newPts.append((x*NewW/w, y*NewH/h)) #use location  coordinates
        else:
            newPts.append((x/w, y/h))  #use location/size ratio
    return rimg,newPts

newW=364
newH=440
    
def newImageScale():
    base = r'.\db\FGNET\\'
    base = os.path.abspath(base)
    
    imgPath = base + r'\images'
    LabelPath = base + r'\points'

    newBase = r'.\db\train\\'
    newBase = os.path.abspath(newBase)
    
    newimgPath = newBase + r'\images'
    newLabelPath = newBase + r'\labels'
    newfList = newBase + r'\trainList.list'
    deleteFile(newfList)
    
    createPath(newimgPath)
    createPath(newLabelPath)
    
    print('base=',base)
    print('newBase=',newBase)
    print('imgPath=',imgPath)
    print('LabelPath=',LabelPath)
    print('newimgPath=',newimgPath)
    print('newLabelPath=',newLabelPath)
    print('newfList=',newfList)

    print(imgPath)
    for i in pathsFiles(imgPath,'JPG'):
        print(i)
        img = loadImg(i)
        H,W = getImgHW(img)
        fileName = getFileName(i)
        fileName = fileName[:fileName.rfind('.')]
        
        label = LabelPath + '\\' + fileName + '.pts'
        print('label=', label)
        pts = getLabelFileLabels(label)
        #print('pts=',len(pts),pts)
        newImg,newPts = resizeImg(img,newW,newH,pts)
        
        
        #print(fileName,H,W)
        a = newimgPath + '\\' + fileName  + '.jpg'
        b = newLabelPath + '\\' + fileName  + '.pts'
        writeToDst(newfList, a + ',' + b + '\n')
        
        dstFile = newimgPath + '\\' + fileName + '.jpg'
        print('dstFile=', dstFile)
        writeImg(newImg,dstFile)
        
        dstLabelFile = newLabelPath + '\\' + fileName + '.pts'
        print('dstLabelFile=', dstLabelFile)
        writeAnotationFile(dstLabelFile,newPts)
    
def main():
    #sourceDatasetOper()
    newImageScale()
    
if __name__ == '__main__':
    main()