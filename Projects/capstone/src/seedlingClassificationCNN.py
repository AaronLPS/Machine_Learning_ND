import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import random
import shutil


# Invesigate the datasets

data_dir = '../data/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
sample_submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))
sample_submission.head(2)

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)

for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))


trainData = []
for categoryID,category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir,category)):
        trainData.append(['train/{}/{}'.format(category, file), categoryID, category])
trainData = pd.DataFrame(trainData, columns=['fileName', 'categoryID', 'category'])
trainData.head(2)
trainData.shape

# The following block only needs to be run once.
# Split validation data from  train data

# os.mkdir('../data/val/')
# for category in CATEGORIES:
#     os.mkdir('../data/val/' + category)
#     name = os.listdir('../data/train/' + category)
#     random.shuffle(name)
#     toVal = name[:int(len(name) * 0.2)] # split 20%
#     for file in toVal:
#         shutil.move(os.path.join('../data/train/', category, file), os.path.join('../data/val/', category))
#         # shutil - High level file operations
#         # https://docs.python.org/2/library/shutil.html


# Change the dimension of the input
#DIM = 48
DIM = 128

# Training data generator
trainDataGenerator = ImageDataGenerator(rescale=1. / 255,  # Normalization
                                        rotation_range=50,
                                        width_shift_range=0.2,  # fraction of total width
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True)

# flow_from_directory(directory): Takes the path to a directory,
# and generates batches of augmented/normalized data. Yields batches indefinitely,
# in an infinite loop.
trainGenerated = trainDataGenerator.flow_from_directory('../data/train',
                                                        # The dimensions to which all images found will be resized.
                                                        target_size=(DIM, DIM),
                                                        #batch_size=16,
                                                        batch_size=16,
                                                        class_mode='categorical',
                                                        shuffle=True
                                                        )

# Validation data generator
valDataGenerator = ImageDataGenerator(rescale=1. / 255)

valGenerated = valDataGenerator.flow_from_directory('../data/val',
                                                    target_size=(DIM, DIM),
                                                    #batch_size=16,
                                                    batch_size=16,
                                                    class_mode='categorical',
                                                    shuffle=True
                                                    )



# Use tensorboard to monitor the training process
# Please note the folders has been renamed while doing the report
from keras.callbacks import TensorBoard
#todo
#tensorboard = TensorBoard('../logs/AWS')
#tensorboard = TensorBoard('../logs/AWS-dim128')
#tensorboard = TensorBoard('../logs/nvidia')
#tensorboard = TensorBoard('../logs/cnn-kaggle-no-batchnormalization')
#tensorboard = TensorBoard('../logs/nvidia-add-bn')
#tensorboard = TensorBoard('../logs/nvidia-bn-gap')
#tensorboard = TensorBoard('../logs/nvidia-bn-gal-dp')
#tensorboard = TensorBoard('../logs/nvidia-bn-gal-dp-adadelta')
#tensorboard = TensorBoard('../logs/nvidia-bn-gal-dp-33-dan')
#tensorboard = TensorBoard('../logs/nvidia-bn-gal-dp-33-256dim')#batch600
#tensorboard = TensorBoard('../logs/nvidia-bn-gal-dp-33-batch800')
tensorboard = TensorBoard('../logs/nvidia-bn-gal-dp-33-batch800-deep')


# Implement the CNN architecture

from keras.layers import Dropout, Input, Dense, Activation,GlobalMaxPooling2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential, Model

#----------------------------------------------------------------------------
#CNN-KAGGLE model (Benchmark Model)

# model = Sequential()
# model.add(Conv2D(16, (3, 3), input_shape=(DIM, DIM, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(16, (3, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(GlobalMaxPooling2D())
#
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=1e-4), metrics=['acc'])
#
#----------------------------------------------------------------------------

# # CNN-KAGGLE model
# # increase the dimension to 128
# # increase the layers of filters

# model = Sequential()
# model.add(Conv2D(16, (7, 7), strides=(2,2),input_shape=(DIM, DIM, 3))) #changed
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# # model_dim128_deep.add(Conv2D(16, (3, 3)))
# # model_dim128_deep.add(BatchNormalization(axis=3))
# # model_dim128_deep.add(Activation('relu'))
# #model_dim128_deep.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (3, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# #model_dim128_deep.add(Conv2D(32, (3, 3)))
# #model_dim128_deep.add(BatchNormalization(axis=3))
# #model_dim128_deep.add(Activation('relu'))
# model.add(GlobalMaxPooling2D())
#
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=1e-4), metrics=['acc'])
#
#=============================================================================================
#
# # NVIDIA CNN model
# # Try the NVIDIA CNN Architecture
#
# model = Sequential()
#
# model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Activation('relu'))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('relu'))
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=1e-4), metrics=['acc'])

#note:
#the traing shows that the acc only increases to 0.13 and can not moving up.

#===========================================================================
# # CNN-KAGGLE model
# # what if delete batchNormalization

# model = Sequential()
# model.add(Conv2D(16, (7, 7), strides=(2,2),input_shape=(DIM, DIM, 3))) #changed
# model.add(Activation('relu'))
# # model_dim128_deep.add(Conv2D(16, (3, 3)))
# # model_dim128_deep.add(BatchNormalization(axis=3))
# # model_dim128_deep.add(Activation('relu'))
# #model_dim128_deep.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# #model_dim128_deep.add(Conv2D(32, (3, 3)))
# #model_dim128_deep.add(BatchNormalization(axis=3))
# #model_dim128_deep.add(Activation('relu'))
# model.add(GlobalMaxPooling2D())
#
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=1e-4), metrics=['acc'])
#
#note:
#the traing shows that the acc only increases to 0.13 and can not moving up.
#
#===========================================================================
#
# # NVIDIA CNN model
# # add batch normalization
#
#
# model = Sequential()
#
# model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Flatten())
# model.add(Activation('relu'))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('relu'))
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=1e-4), metrics=['acc'])

#===========================================================================

# # NVIDIA CNN model
# # add batch normalization
# # use global average layer to replace FC
#
# from keras.layers import GlobalAveragePooling2D, AveragePooling2D
#
# model = Sequential()
#
# model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# # model.add(Flatten())
# model.add(Activation('relu'))
# # model.add(Dense(100))
# # model.add(Activation('relu'))
# # model.add(Dense(50))
# # model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
#
# model.add(GlobalAveragePooling2D())
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=1e-4), metrics=['acc'])


#===========================================================================
#
# # NVIDIA CNN model
# # add batch normalization
# # use global average layer to replace FC
# # add more filters to be deeper
#
# from keras.layers import GlobalAveragePooling2D, AveragePooling2D
# from keras import optimizers
# model = Sequential()
#
# model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# # model.add(Flatten())
# model.add(Activation('relu'))
# # model.add(Dense(100))
# # model.add(Activation('relu'))
# # model.add(Dense(50))
# # model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
#
# model.add(GlobalAveragePooling2D())
# model.add(Dense(12, activation='softmax'))
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=1e-4), metrics=['acc'])
#
# # Note: It doesn't work. I suppose it is caused by too much dimension reduction(subsample)
#===========================================================================
#
# # NVIDIA CNN model
# # add batch normalization
# # use global average layer to replace FC
#
#
# from keras.layers import GlobalAveragePooling2D, AveragePooling2D
#
# model = Sequential()
#
# model.add(Conv2D(24, 3, 3, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(36, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# # model.add(Flatten())
# model.add(Activation('relu'))
# # model.add(Dense(100))
# # model.add(Activation('relu'))
# # model.add(Dense(50))
# # model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#                optimizer=Adam(lr=1e-4), metrics=['acc'])

#===========================================================================

# # NVIDIA CNN model
# # add batch normalization
# # use global average layer to replace FC
#
# # test DAN
# #https://zhuanlan.zhihu.com/p/23176872
#
# from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, concatenate, merge
# from keras.layers import Lambda
# import keras.backend as K
#
# def l2_norm(x):
#     x = x ** 2
#     x = K.sum(x, axis=1)
#     x = K.sqrt(x)
#     return x
#
#
# CNNM_input = Input((DIM, DIM, 3))
# CNNM = Conv2D(24, 3, 3, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3))(CNNM_input)
# CNNM = BatchNormalization(axis=3)(CNNM)
# CNNM = Activation('relu')(CNNM)
# CNNM = Conv2D(36, 3, 3, subsample=(2, 2), border_mode="same")(CNNM)
# CNNM = BatchNormalization(axis=3)(CNNM)
# CNNM = Activation('relu')(CNNM)
# CNNM = Conv2D(48, 3, 3, subsample=(2, 2), border_mode="same")(CNNM)
# CNNM = BatchNormalization(axis=3)(CNNM)
# CNNM = Activation('relu')(CNNM)
# CNNM = Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same")(CNNM)
# CNNM = BatchNormalization(axis=3)(CNNM)
# CNNM = Activation('relu')(CNNM)
# CNNM = Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same")(CNNM)
# CNNM = BatchNormalization(axis=3)(CNNM)
# CNNM = Activation('relu')(CNNM)
# CNNM = Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same")(CNNM)
# CNNM = BatchNormalization(axis=3)(CNNM)
# CNNM = Activation('relu')(CNNM)
#
# CNNM_1 = GlobalAveragePooling2D()(CNNM)
# #CNNM_1_l2N = Lambda(lambda x: l2_norm(x))(CNNM_1)
# CNNM_1 = BatchNormalization(axis=1)(CNNM_1)
#
#
# CNNM_2 = GlobalMaxPooling2D()(CNNM)
# CNNM_2 = BatchNormalization(axis=1)(CNNM_2)
# #CNNM_2_l2N = Lambda(lambda x: l2_norm(x))(CNNM_2)
#
# #merged = concatenate([CNNM_1_l2N, CNNM_2_l2N])
#
# merged = merge([CNNM_1, CNNM_2], mode='concat', concat_axis=1)
# merged3 = Dense(12, activation='softmax')(merged)
#
# model = Model(input=CNNM_input, outputs=merged3)
# model.summary()
# model.compile(loss='categorical_crossentropy',
#                optimizer=Adam(lr=1e-4), metrics=['acc'])

# # Note: no improvement

#===========================================================================
#
# # NVIDIA CNN model
# # add batch normalization
# # use global average layer to replace FC
# # dim256 doesn't work
# # batch_size = 600 has good increce comparing to 400
# # try batch_size = 800
#
# from keras.layers import GlobalAveragePooling2D, AveragePooling2D
#
# model = Sequential()
#
# model.add(Conv2D(24, 3, 3, subsample=(2, 2), border_mode="same", input_shape=(DIM, DIM, 3)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(36, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# # model.add(Flatten())
# model.add(Activation('relu'))
# # model.add(Dense(100))
# # model.add(Activation('relu'))
# # model.add(Dense(50))
# # model.add(Activation('relu'))
# model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(12, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy',
#                optimizer=Adam(lr=1e-4), metrics=['acc'])

#===========================================================================

# NVIDIA CNN model (Final Model)
# add batch normalization
# use global average layer to replace FC
# increase the depth of the model

from keras.layers import GlobalAveragePooling2D, AveragePooling2D

model = Sequential()

model.add(Conv2D(24, 3, 3, border_mode="same", input_shape=(DIM, DIM, 3)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(24, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(36, 3, 3, border_mode="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(36, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(48, 3, 3, border_mode="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(48, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(BatchNormalization(axis=3))
# model.add(Flatten())
model.add(Activation('relu'))
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dense(50))
# model.add(Activation('relu'))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(12, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
               optimizer=Adam(lr=1e-4), metrics=['acc'])


#=============================================================================




#Note about the CNN model:
#To understand the Dropout: http://blog.csdn.net/u013007900/article/details/78120669
#The function of FC：https://www.zhihu.com/question/41037974
#GAP replaces FC： https://zhuanlan.zhihu.com/p/23176872
#ResNet, AlexNet, VGG, Inception: https://zhuanlan.zhihu.com/p/32116277


# Train the model
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint
from IPython.display import display
from PIL import Image

# Adjust the batch size
#todo
#batch_size = 64
#batch_size = 400
#batch_size = 600
batch_size = 800

#note：the influence of the batch size？https://www.zhihu.com/question/32673260


lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
earlystop = EarlyStopping(patience=10)
modelsave = ModelCheckpoint(
    #todo
    #filepath='../checkpoint/aws-weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/aws-dim128-weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia/nvidia-weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/cnn-kaggle-no-batchnormalization/cnn-kaggle-no-BN-weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia-add-bn/weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia-bn-gap/weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia-bn-gal-dp/train2-weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia-bn-gal-dp-adadelta/weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia-bn-gal-dp-33-dan/weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia-bn-gal-dp-33-256dim/2nd-weights-{epoch:02d}-{val_loss:.2f}.h5', # filepath='../checkpoint/nvidia-bn-gal-dp-33-256dim/weights-{epoch:02d}-{val_loss:.2f}.h5',
    #filepath='../checkpoint/nvidia-bn-gal-dp-33-batch800/2nd-weights-{epoch:02d}-{val_loss:.2f}.h5',
    filepath='../checkpoint/nvidia-bn-gal-dp-33-batch800-deep/weights-{epoch:02d}-{val_loss:.2f}.h5',
    save_best_only=True,
    verbose=1)

# Restart the Model training:
# from keras.models import load_model
# model = load_model('../model/CNN_AWS_TEST3-NVIDIA-bn-gal-deep.h5')
# from keras.models import load_model
# model = load_model('../checkpoint/nvidia-bn-gal-dp-33-256dim/weights-23-0.18.h5')

# Train the model
model.fit_generator(trainGenerated,
                    steps_per_epoch=batch_size,
                    epochs=200,
                    validation_data=valGenerated,
                    callbacks=[modelsave, tensorboard,earlystop, lr],
                    #validation_steps=10,
                    validation_steps=50,
                    #validation_steps=100,
                    workers = 16
                    )

# Save the model
#todo
#model.save('../model/CNN_AWS_TEST1.h5')
#model.save('../model/CNN_AWS_TEST2-dim128.h5')
#model.save('../model/CNN_AWS_TEST3-NVIDIA.h5')
#model.save('../model/CNN_AWS_TEST4-cnn-kaggle-no-bn.h5')
#model.save('../model/CNN_AWS_TEST3-NVIDIA-add-bn.h5')
#model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gap.h5')
#model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gal-deep-test.h5') #model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gal-deep.h5')
#model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gap-adadelta.h5') #0.935
#model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gap-33-dan.h5')
#model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gap-33-256dim.h5') #batch600
#model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gap-33-batch800.h5') # 0.95
model.save('../model/CNN_AWS_TEST3-NVIDIA-bn-gap-33-batch800-deep.h5') # 0.959 (Final Model)
