import os,sys
from commonModule.ImageBase import *
from predictKeyPoints import *
from makeDB import calculateFeature,getDb,distanceAB
from testLabel import testFaceLabelPredict,showimage,testFaceLabelPts
from genLabel import newW,newH

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help = 'source image')
    #parser.add_argument('-d', '--dst', help = 'save iamge')
    return parser.parse_args()

def predictPerson(feature):
    df = getDb()
    print(df.head())
    #allFeatures = df.iloc[:,1:].values
    #print('allFeatures.shape=',allFeatures.shape)
    allDistance=[]
    allIds = []
    for i in range(df.shape[0]):
        id = df.iloc[i,0]
        i = df.iloc[i,1:].values
        #print(id,i)
        allIds.append(id)
        
        dis = distanceAB(i,feature)
        #print('dis=',dis)
        allDistance.append(dis)

    #print('allDistance=',allDistance)
    disMin = min(allDistance)
    minIndex = allDistance.index(disMin)
    print('disMin=',disMin,'minIndex=',minIndex,'id=',allIds[minIndex])
    
    
def main():
    arg = argCmdParse()    
    file = r'./res/myface_.png'  #r'./res/001A29_ex2.jpg' #r'./res/001A29.jpg' #arg.source  #
    
    img = loadImg(file)
    img = resizeImg(img,newW,newH)
    
    H,W = getImgHW(img)
    pts = preditImg(img)
    print('pts=',len(pts),pts.shape,'H,W=',H,W)
    print('pts=',pts)
    
    showimage(testFaceLabelPts(img,pts,locCod=False))
    if 1:
        feature = np.array(calculateFeature(pts,H,W,r'./res/predict.pts'))
    else:
        feature = pts.reshape(1,136)
    print('feature=',len(feature),feature)
    predictPerson(feature)
    
if __name__=='__main__':
    main()