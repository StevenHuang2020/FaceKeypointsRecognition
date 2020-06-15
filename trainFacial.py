import tensorflow.keras as ks
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
import datetime

from loadImages import loadDataset
from genLabel import getLabelFileLabels,newW,newH
    
def prepareData():
    img_rows, img_cols = newH,newW
    
    X, Y = loadDataset()
    print('X.shape=',X.shape)
    print('Y.shape=',Y.shape)
    
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
    
    test_size=0.2
    trainLen = int(len(X)*(1-test_size))
    x_train = X[:trainLen]
    y_train = Y[:trainLen]
    x_test = X[trainLen:]
    y_test = Y[trainLen:]
    print('total:',len(X),len(Y),'train:',trainLen,'test:',len(X)-trainLen)
    
    print('X_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', x_test.shape)
    print('Y_test.shape = ', y_test.shape)
    
    return x_train, y_train, x_test, y_test, (img_rows, img_cols, 1)

def createModel(input_shape,ptSize=68):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    #model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(ptSize*2))
    
    lr = 0.0001
    #opt = optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    #opt = optimizers.Adadelta(learning_rate=lr, rho=0.95)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9)
    #opt = optimizers.Adagrad(learning_rate=lr)    
    opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #model.compile(loss=ks.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.compile(loss='mean_squared_error',
              optimizer=opt)
    
    model.summary()
    return model

def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help = 'epochs')
    
    return parser.parse_args()

def main():
    arg = argCmdParse()
    
    epoch = 300
    if arg.epoch:
        epoch = int(arg.epoch)
    
    print('epoch=',epoch)

    x_train, y_train, x_test, y_test, input_shape = prepareData() #prepareMnistData() #
    print('input_shape = ', input_shape)

    modelName = r'./weights/trainFacialRecognition.h5'
    #weightsFiles = r'./weights'
    #model = createModel(input_shape)
    model = ks.models.load_model(modelName)
    #model.load_weights(weightsFiles)
    
    log_dir = r"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = ks.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_filepath = r'./checkpoint'
    checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=False,save_freq=100)
        
    #model.fit(x_train, y_train, epochs=10, callbacks = [tensorboard_callback,checkpointer])
    model.fit(x_train, y_train, epochs=epoch, verbose=1, batch_size=100,
              validation_data=(x_test, y_test),callbacks = [tensorboard_callback,checkpointer]) #
    
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1]
    
    model.save(modelName)
    model.save(r'./weights/' + 'trainFacialRecognition' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5')
    #model.save_weights(weightsFiles)
 
if __name__=='__main__':
    main()