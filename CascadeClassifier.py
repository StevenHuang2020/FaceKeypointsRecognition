#python3
import cv2.cv2 as cv2
import numpy as np

class CascadeClassifier:
    def __init__(self,cascPath):
        self.cascPath = cascPath
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detecvFace(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                    )
        return faces

    def getNewLocFromDetection(self,centerX,centerY,img,reSize=None):       
        newW = reSize[0]
        newH = reSize[1]
        H,W = img.shape[0],img.shape[1]
        if newW>W:
            newW = W
        if newH>H:
            newH = H 
        
        newX = int(centerX-newW/2)
        newY = int(centerY-newH/2)
        if newX<0:
            newX = 0
        if newY<0:
            newY = 0
        return newX,newY,newW,newH
    '''
    def getNewLocFromDetection(self,x,y,w,h,img,reSize=None):
        if reSize is None:
            return x,y,w,h
        
        centerX = int(x + w/2)
        centerY = int(y + h/2)
        
        newW = reSize[0]
        newH = reSize[1]
        H,W = img.shape[0],img.shape[1]
        if newW>W:
            newW = W
        if newH>H:
            newH = H 
        
        newX = int(centerX-newW/2)
        newY = int(centerY-newH/2)
        if newX<0:
            newX = 0
        if newY<0:
            newY = 0
        return newX,newY,newW,newH
        '''
                
    def detecvFaceImgOne(self,img,reSize=None): #return one
        faces = self.detecvFace(img)
        for (x, y, w, h) in faces:
            #return img[x:x+w,y:y+h]
            #print(x, y, w, h)
            if reSize is None:
                return img[y:y+h,x:x+w]
            else:
                centerX = int(x + w/2)
                centerY = int(y + h/2)
                #print('center=',centerX,centerY)
                newX,newY,newW,newH = self.getNewLocFromDetection(centerX,centerY,img,reSize)    
                #print('changesize=',newW,newH,reSize[0],reSize[1])              
                return img[newY:newY+newH, newX:newX+newW]
                
        print('Warning!,No detection of face!')
        return None
    
    def getDetectImg(self,image,reSize=None):
        faces = self.detecvFace(image)
        #print("Found {0} faces!".format(len(faces)))
        newImg = image.copy() 
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            if reSize is None:
                cv2.rectangle(newImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                #cv2.rectangle(newImg, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
                centerX = int(x + w/2)
                centerY = int(y + h/2)
                print('center=',centerX,centerY)
                x,y,w,h = self.getNewLocFromDetection(centerX,centerY,newImg,reSize)
                cv2.rectangle(newImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break
        return newImg
    
    def detecvFaceImgs(self,img): #return one
        faces = self.detecvFace(img)
        imgList=[]
        for (x, y, w, h) in faces:
            imgList.append(img[x:x+w,y:y+h].copy())
        return imgList

    def showDetectImg(self,image):
        faces = self.detecvFace(image)
        #print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Faces found", image)
        cv2.waitKey(0)

