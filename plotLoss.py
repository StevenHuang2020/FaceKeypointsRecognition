#Steven 24/04/2020 
import argparse 
import sys
import matplotlib.pyplot as plt

#----------------------------------------------
#usgae: python plotloss.py .\facialRecognition.log
#----------------------------------------------

def getLoss(log_file,startIter=0,stopIter=None):
    numbers = {'1','2','3','4','5','6','7','8','9'}
    with open(log_file, 'r') as f:
        lines  = [line.rstrip("\n") for line in f.readlines()]
        
        iters = []
        loss = []
        val_loss=[]
        for line in lines:
            trainIterRes = line.split(' ')
            #print(line)
            epoch = 0
            if trainIterRes[0] == 'Epoch' and trainIterRes[1][-1:]!=':':
                str = trainIterRes[1]
                epoch = int(str[:str.find('/')])
                #print(trainIterRes[1],epoch)
                if(epoch<startIter):
                    continue       
                if stopIter and  epoch > stopIter:
                    break
            
                iters.append(epoch)
           
            if trainIterRes[0] == '9/9' and trainIterRes[3] != 'ETA:':
                print(line)
                print(trainIterRes[7],trainIterRes[10])
       
                loss.append(float(trainIterRes[7]))
                val_loss.append(float(trainIterRes[10]))
                
    return iters,loss,val_loss

def plotLoss(ax,iters,loss,label='',name='Training loss'):
    #ax.set_title(name)
    ax.plot(iters,loss,label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
     
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--list', nargs='+', help='path to log file', required=True)
    parser.add_argument('-s', '--start', help = 'startIter')
    parser.add_argument('-t', '--stop', help = 'stopIter')
    
    args = parser.parse_args()
    startIter = 0
    stopIter = None
    if args.start:
        startIter = int(args.start)
    if args.stop:
        stopIter = int(args.stop)
        
    print(args.list,startIter,stopIter)
    
    ax = plt.subplot(1,1,1)
    file = args.list[0]
    iters,loss,val_loss = getLoss(file,startIter,stopIter)
    
    plotLoss(ax,iters,loss,label='On train set')
    plotLoss(ax,iters,val_loss,label='On validation set')
    #plt.ylim(0, 4)
    plt.yscale("log")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv)
    