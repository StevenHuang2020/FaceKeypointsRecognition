#python3
#Steven 11/03/2020 image plot modoule

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from commonModule.common import getRowAndColumn

def plotImagList2(imgList,nameList,gray=False,showTitle=True,showticks=True):
    nImg = len(imgList)
    nRow,nColumn = getRowAndColumn(nImg)

    #f, axarr = plt.subplots(nRow, nColumn,gridspec_kw = {'wspace':0, 'hspace':0}) #
    f, axarr = plt.subplots(nRow, nColumn, constrained_layout=True) #
    
    gs1 = gridspec.GridSpec(nRow, nColumn)
    gs1.update(wspace=0, hspace=0)
    
    print(len(f.axes),nRow,nColumn)
    
    for n in range(nImg):
        ax = plt.subplot(gs1[n])
        if 1:
            img = imgList[n]
            
            if showTitle:
                ax.title.set_text(nameList[n])
            if gray:
                ax.imshow(img,cmap="gray")
            else:
                ax.imshow(img)
            
            if not showticks:
                ax.set_yticks([])
                ax.set_xticks([])
            
            ax.margins(0, 0) 
            #ax.xaxis.set_major_locator(plt.NullLocator())
            #ax.yaxis.set_major_locator(plt.NullLocator())
   
    
    #plt.grid(True)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.show()

def plotImagList(imgList,nameList,gray=False,showTitle=True,showticks=True):
    nImg = len(imgList)
    nRow,nColumn = getRowAndColumn(nImg)
    
    for n in range(nImg):
        img = imgList[n]
        ax = plt.subplot(nRow, nColumn, n + 1)
        if showTitle:
            ax.title.set_text(nameList[n])
        if gray:
            plt.imshow(img,cmap="gray")
        else:
            plt.imshow(img)
        
        if not showticks:
            ax.set_yticks([])
            ax.set_xticks([])
        
        #ax.margins(0, 0) 
        #ax.xaxis.set_major_locator(plt.NullLocator())
        #ax.yaxis.set_major_locator(plt.NullLocator())
    #plt.grid(True)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.show()

def main():
    #file = r'./res/obama.jpg'#'./res/Lenna.png' #
    #img = ImageBase(file,mode=cv2.IMREAD_GRAYSCALE) # IMREAD_GRAYSCALE IMREAD_COLOR
    #print(img.infoImg())
    #showimage(img.binaryImage(thresHMin=50,thresHMax=150))
    pass

if __name__=='__main__':
    main()
