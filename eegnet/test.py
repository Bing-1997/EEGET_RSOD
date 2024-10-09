from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
#from keras.optimizers import Adadelta
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import EEGNet
import numpy as np
import os


def eegnet_test(file):
    kernels, chans, samples = 1, 35, 1001
    '''
    global_step=tf.Variable(0,name='global_step', trainable=False)
    learning_rate_1 = tf.compat.v1.train.exponential_decay(  # 阶梯型衰减
        learning_rate=0.01, global_step=global_step, decay_steps=10,
        decay_rate=0.5,
        staircase=True
    )
    '''

    #file = "K:/object detection/eegnet/data/202021051196.set"
    #X_train, X_validate, X_test, Y_train, Y_validate, Y_test = EEGNet.get_data4EEGNet_v(file,kernels, chans, samples)
    X_train, X_test, Y_train, Y_test = EEGNet.get_data4EEGNet(file,kernels, chans, samples)

    model = EEGNet.EEGNet(nb_classes = 2, Chans = chans, Samples = samples,
                   dropoutRate = 0.25, kernLength = 32, F1 = 16, D = 2, F2 = 32,
                   dropoutType = 'Dropout')
    optimizer = SGD(lr=0.005)
    #optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics = ['accuracy'])

    checkpointer = ModelCheckpoint(filepath='./baseline.h5', verbose=1,)
    #scores = cross_val_score(model,X_train, Y_train,cv=3)

    fittedModel = model.fit(X_train, Y_train, batch_size =32, epochs = 50,
                            verbose = 2, callbacks=[checkpointer]) #, validation_data=(X_validate, Y_validate)


    model.load_weights('./baseline.h5')

    probs       = model.predict(X_test)
    preds       = probs.argmax(axis = -1)
    acc         = np.mean(preds == Y_test.argmax(axis=-1))
    #print("Classification accuracy: %f " % (acc))
    return acc

if __name__ == "__main__":
    file_path = 'K:/object detection/eegnet/data/'
    filesname = os.listdir(file_path)
    set_files = [file for file in filesname if file.endswith('.set')]
    #print(set_files)
    #set_files = ['202021051196.set']
    acc = []
    for name in set_files:
        acc_temp = eegnet_test(file_path+name)
        acc.append(str(acc_temp))
    with open('./EEGNet_acc.txt', 'a') as file:
        file.write('\n'.join(acc))

