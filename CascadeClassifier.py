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
                    minSize=(30, 30)
                    )
        return faces

    def detecvFaceImgOne(self,img): #return one
        faces = self.detecvFace(img)
        if len(faces)>0:
            (x, y, w, h) = faces[-1]
            #return img[x:x+w,y:y+h]
            return img[y:y+h,x:x+w]
        return None
        # for (x, y, w, h) in faces:
        #     print(x,y,w,h)
        #     #return img[x:x+w,y:y+h].copy()
        #     return img[x:x+w,y:y+h]
    
    def detecvFaceImgs(self,img): #return one
        faces = self.detecvFace(img)
        imgList=[]
        for (x, y, w, h) in faces:
            imgList.append(img[x:x+w,y:y+h].copy())
        return imgList

    def getDetectImg(self,image):
        faces = self.detecvFace(image)
        #print("Found {0} faces!".format(len(faces)))
        newImg = image.copy() 
        for (x, y, w, h) in faces:
            #print(x, y, w, h)
            cv2.rectangle(newImg, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return newImg

    def showDetectImg(self,image):
        faces = self.detecvFace(image)
        #print("Found {0} faces!".format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Faces found", image)
        cv2.waitKey(0)

