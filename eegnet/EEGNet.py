

import mne
from mne import io
from mne.datasets import sample

from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import pathlib
import tensorflow as tf

def get_data4EEGNet_v(file,kernels, chans, samples):
    K.set_image_data_format('channels_last')

    tmin, tmax = 0, 2

    raw = mne.io.read_raw_eeglab(file, preload=False)
    #raw = mne.io.read_epochs_eeglab(file)
    # 提取通道
    # channels_to_extract = ['F3', 'F4', 'F7', 'F8', 'Fz', 'O1', 'O2', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp1',
    #                        'Fp2']  # 例如，提取Cz和Pz通道
    # extracted_channels = raw.copy().pick_channels(channels_to_extract)
    # events, event_id = mne.events_from_annotations(extracted_channels)

    events, event_id = mne.events_from_annotations(raw)
    #print(event_id)
    #print(events[:])

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    #print(picks)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                        picks=picks, baseline=None, preload=True, verbose=False)

    X = epochs.get_data() * 1000
    labels = epochs.events[:, -1]
    # print(len(X))
    y = labels-1
    # y=y[:][0:1]
    # x_trainVal, x_test, y_trainVal, y_test = train_test_split(data, labels.ravel(), shuffle=True, stratify=labels, random_state=0)

    # 首先分割出训练集和临时测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True)
    #print("y_train, y_temp", y_train, y_temp)
    # 然后将临时测试集分割成测试集和验证集
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.4, shuffle=True)

    y_train = np_utils.to_categorical(y_train)
    y_validate = np_utils.to_categorical(y_validate)
    y_test = np_utils.to_categorical(y_test)
    y_train = y_train[:, 0:2]
    y_validate = y_validate[:, 0:2]
    y_test = y_test[:, 0:2]
    # print("y_train:", y_train)
    # print("y_validate:" , y_validate)
    # print("y_test:",y_test)
    # print(X_train.shape[0])
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
    # print(X_train.shape, X_validate.shape, X_test.shape, y_train.shape, y_validate.shape, y_test.shape)
    '''
    X = epochs.get_data()*1000
    y = labels

    X_train      = X[0:144,]
    Y_train      = y[0:144]
    X_validate   = X[144:216,]
    Y_validate   = y[144:216]
    X_test       = X[216:,]
    Y_test       = y[216:]


    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)
    Y_test       = np_utils.to_categorical(Y_test-1)


    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
    '''
    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_data4EEGNet(file,kernels, chans, samples):
    K.set_image_data_format('channels_last')

    tmin, tmax = 0, 1

    raw = mne.io.read_raw_eeglab(file, preload=False)
    #raw = mne.io.read_epochs_eeglab(file)
    #提取通道
    channels_to_extract = ['F3', 'F4', 'F7', 'F8', 'Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp1', 'Fp2'
                           'Pz', 'P3', 'P4', 'P7','P8']  # 例如，提取Cz和Pz通道
    extracted_channels = raw.copy().pick_channels(channels_to_extract)
    events, event_id = mne.events_from_annotations(extracted_channels)

    #events, event_id = mne.events_from_annotations(raw)
    #print(event_id)
    #print(events[:])

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    #print(picks)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                        picks=picks, baseline=None, preload=True, verbose=False)

    X = epochs.get_data() * 1000
    labels = epochs.events[:, -1]
    # print(len(X))
    y = labels-1
    # y=y[:][0:1]
    # x_trainVal, x_test, y_trainVal, y_test = train_test_split(data, labels.ravel(), shuffle=True, stratify=labels, random_state=0)

    # 首先分割出训练集和临时测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True)
    #print("y_train, y_temp", y_train, y_temp)
    # 然后将临时测试集分割成测试集和验证集

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_temp)
    y_train = y_train[:, 0:2]
    y_test = y_test[:, 0:2]
    # print("y_train:", y_train)
    # print("y_validate:" , y_validate)
    # print("y_test:",y_test)
    # print(X_train.shape[0])
    X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_test = X_temp.reshape(X_temp.shape[0], chans, samples, kernels)
    # print(X_train.shape, X_validate.shape, X_test.shape, y_train.shape, y_validate.shape, y_test.shape)
    '''
    X = epochs.get_data()*1000
    y = labels

    X_train      = X[0:144,]
    Y_train      = y[0:144]
    X_validate   = X[144:216,]
    Y_validate   = y[144:216]
    X_test       = X[216:,]
    Y_test       = y[216:]


    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)
    Y_test       = np_utils.to_categorical(Y_test-1)


    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
    '''
    return X_train, X_test, y_train,  y_test


def EEGNet(nb_classes, Chans=35, Samples=1001,
           dropoutRate=0.2, kernLength=64, F1=8,
           D=3, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


