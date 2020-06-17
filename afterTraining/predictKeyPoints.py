import os,sys
#print(sys.path)
sys.path.append('..')
#print(sys.path)

import tensorflow.keras as ks
import datetime
import argparse
from commonModule.ImageBase import *
from genLabel import writeAnotationFile

def getImg(file):
    return loadImg(file)

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help = 'source image')
    parser.add_argument('-d', '--dst', help = 'save iamge')
    
    return parser.parse_args()

def preditImg(img, modelName = r'./weights/trainFacialRecognition.h5'):
    model = ks.models.load_model(modelName)
    print(img.shape)
    x = img[:,:,0]
    print(x.shape)
    x = x.reshape((1,x.shape[0],x.shape[1],1))
    print(x.shape)
    pts = model.predict(x)
    return pts

def main():
    arg = argCmdParse()    
    file=arg.source  #r'./res/myface_gray.png'
    dstFile = arg.dst
    print(file,dstFile)
    
    img = getImg(file)
    pts = preditImg(img)
    '''
    print(img.shape)
    x = img[:,:,0]
    print(x.shape)
    x = x.reshape((1,x.shape[0],x.shape[1],1))
    print(x.shape)
    #return

    pts = model.predict(x)
    '''
    print('pts=',len(pts),pts.shape,pts)
    pts = pts.reshape((68,2))
    print('pts=',len(pts),pts.shape,pts)

    
    writeAnotationFile(dstFile,pts) #'./res/myface_gray.pts'
    
if __name__=='__main__':
    main()