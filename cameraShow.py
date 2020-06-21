import numpy as np
import cv2
import tensorflow.keras as ks
from testLabel import testFaceLabelPts
from faceIdentification import CascadeDetect

newW=364
newH=440

def getImgHW(img):
    return img.shape[0],img.shape[1]

def resizeImg(img,NewW,NewH):
    rimg = cv2.resize(img, (NewW,NewH), interpolation=cv2.INTER_CUBIC) #INTER_CUBIC INTER_NEAREST INTER_LINEAR INTER_AREA
    return rimg

def preditImgKeyPoints(img, model):
    global newW
    global newH
    img = resizeImg(img,newW,newH)
    pts = preditImgPts(img,model)
    return testFaceLabelPts(img,pts)

def preditImgPts(img, model): 
    x = img[:,:,0]
    #print(x.shape)
    x = x.reshape((1,x.shape[0],x.shape[1],1))
    #print(x.shape)
    pts = model.predict(x)
    pts = pts.reshape((68,2))
    return pts

def InitNet():
    weights=r'./weights/trainFacialRecognition.h5'
    model = ks.models.load_model(weights)
    return model

def showCamera():
    model  = InitNet()
         
    saveVideo=False
    
    #fourcc = cv2.VideoWriter.fourcc('X','2','6','4')
    #fourcc = cv2.VideoWriter.fourcc('v','p','8','0')
    fourcc = cv2.VideoWriter.fourcc('M','J','P','G')
    out = cv2.VideoWriter('output2.mp4', fourcc, 25.0, (640,480))
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps=',fps)
    
    while(True):
        ret, frame = cap.read()
        #Our operations on the frame come here
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Display the resulting frame
        
        #frame=CascadeDetect().getDetectImg(frame)
        face = CascadeDetect().detecvFaceImgOne(frame)
        if face is None:
            continue
        
        frame = preditImgKeyPoints(face,model)
        if saveVideo:
            out.write(frame)
            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def main():
    showCamera()
    
if __name__=='__main__':
    main()
    