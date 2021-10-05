from keras.layers import Input, concatenate, Flatten, Dense, add, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Model
import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import deepfool
# import model as mdl
from torchvision import transforms
import torchvision
import time
from sklearn.metrics import accuracy_score

start = time.time()
 
K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 224, 224, 62


def build(height=224, width=224, channels=62, classes=8):

    im = Input(shape=(224, 224, 62))
    Conv_S = Conv2D(16, (3, 3), activation='relu',
                    padding='same', strides=2, name='Conv_S')(im)
    # -------------------------------------------------------------------

    Conv_1_1 = Conv2D(16, (1, 1), activation='relu',
                      padding='same', strides=2, name='Conv_1_1')(Conv_S)
    Conv_1_2 = Conv2D(32, (3, 3), activation='relu',
                      padding='same', strides=2, name='Conv_1_2')(Conv_1_1)
    Conv_1_3 = Conv2D(64, (5, 5), activation='relu',
                      padding='same', strides=2, name='Conv_1_3')(Conv_1_2)
    # ------------------------------------------------------------------

    Conv_2_1 = Conv2D(16, (1, 1), activation='relu',
                      padding='same', strides=2, name='Conv_2_1')(Conv_S)
    add_2_1 = add([Conv_1_1, Conv_2_1])
    batch_r11 = BatchNormalization()(add_2_1)
    Conv_2_2 = Conv2D(32, (3, 3), activation='relu',
                      padding='same', strides=2, name='Conv_2_2')(batch_r11)
    add_2_2 = add([Conv_1_2, Conv_2_2])
    batch_r12 = BatchNormalization()(add_2_2)
    Conv_x_2 = Conv2D(64, (5, 5), activation='relu',
                      padding='same', strides=2, name='Conv_x_2')(batch_r12)
    # ------------------------------------------------------------------

    Conv_3_1 = Conv2D(16, (1, 1), activation='relu',
                      padding='same', strides=2, name='Conv_3_1')(Conv_S)
    Conv_3_2 = Conv2D(32, (3, 3), activation='relu',
                      padding='same', strides=2, name='Conv_3_2')(Conv_3_1)
    Conv_3_3 = Conv2D(64, (5, 5), activation='relu',
                      padding='same', strides=2, name='Conv_3_3')(Conv_3_2)
    # ------------------------------------------------------------------

    Conv_4_1 = Conv2D(16, (1, 1), activation='relu',
                      padding='same', strides=2, name='Conv_4_1')(Conv_S)
    add_4_1 = add([Conv_3_1, Conv_4_1])
    batch_r13 = BatchNormalization()(add_4_1)
    Conv_4_2 = Conv2D(32, (3, 3), activation='relu',
                      padding='same', strides=2, name='Conv_4_2')(batch_r13)
    add_4_2 = add([Conv_3_2, Conv_4_2])
    batch_r14 = BatchNormalization()(add_4_2)
    Conv_x_4 = Conv2D(64, (5, 5), activation='relu',
                      padding='same', strides=2, name='Conv_x_4')(batch_r14)

    # --------------------------------------------------------
    concta1 = concatenate([Conv_1_3, Conv_x_2, Conv_3_3, Conv_x_4])
    batch_X = BatchNormalization()(concta1)

    #-----------------------------------------------------#
    Conv_5_1 = Conv2D(256, (3, 3), activation='relu',
                      padding='same', strides=2, name='Conv_5_1')(batch_X)
    # -----Fully Connected layer--------
    F1 = Flatten()(Conv_5_1)
    FC1 = Dense(256, activation='relu')(F1)
    drop = Dropout(0.5)(FC1)

    # ------clasisfication layer-------

    out = Dense(classes, activation='softmax')(drop)

    model = Model(inputs=[im], outputs=out)
    return model


training_list = []
comtemptpath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/comtempt/'
disgustpath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/disgust/'
fearpath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/fear/'
happinesspath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/happiness/'
repressionpath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/repression/'
sadnesspath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/sadness/'
surprisepath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/surprise/'
tensepath = 'D:/Projects/FIT3161 - FYP/CASME/sortedvideo/tense/'
directorylisting = os.listdir(comtemptpath)

for video in directorylisting:
    frames = []
    videopath = comtemptpath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
        except:
            full = False
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)


print(len(training_list))

directorylisting = os.listdir(disgustpath)
for video in directorylisting:
    frames = []
    videopath = disgustpath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
        except:
            full = False
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)

print(len(training_list))

directorylisting = os.listdir(fearpath)
for video in directorylisting:
    frames = []
    videopath = fearpath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
        except:
            full = False
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)

print(len(training_list))

directorylisting = os.listdir(happinesspath)
for video in directorylisting:
    frames = []
    videopath = happinesspath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
        except:
            full = False
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)

print(len(training_list))

directorylisting = os.listdir(repressionpath)
for video in directorylisting:
    frames = []
    videopath = repressionpath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
            full = False
        except:
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)

print(len(training_list))

directorylisting = os.listdir(sadnesspath)
for video in directorylisting:
    frames = []
    videopath = sadnesspath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
        except:
            full = False
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)

print(len(training_list))

directorylisting = os.listdir(surprisepath)
for video in directorylisting:
    frames = []
    videopath = surprisepath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
        except:
            full = False
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)

print(len(training_list))

directorylisting = os.listdir(tensepath)
for video in directorylisting:
    frames = []
    videopath = tensepath + video
    loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
    framerange = [x for x in range(62)]
    full = True
    for frame in framerange:
        try:
            image = loadedvideo.get_data(frame)
        except:
            full = False
            break
        imageresize = cv2.resize(
            image, (image_rows, image_columns), interpolation=cv2.INTER_AREA)
        grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
        frames.append(grayimage)
    if full:
        frames = numpy.asarray(frames)
        videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
        training_list.append(videoarray)


print(len(training_list))


training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)

traininglabels = numpy.zeros((trainingsamples, ), dtype=int)

traininglabels[0:2] = 0
traininglabels[2:34] = 1
traininglabels[34:36] = 2
traininglabels[36:44] = 3
traininglabels[44:44] = 4
traininglabels[44:48] = 5
traininglabels[48:67] = 6
traininglabels[67:132] = 7

traininglabels = np_utils.to_categorical(traininglabels, 8)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros(
    (trainingsamples, image_rows, image_columns, image_depth))
for h in range(trainingsamples):
    training_set[h][:][:][:] = trainingframes[h, :, :, :]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)

# Save training images and labels in a numpy array
# numpy.save('numpy_training_datasets/microexpstcnn_images.npy', training_set)
# numpy.save('numpy_training_datasets/microexpstcnn_labels.npy', traininglabels)

# Load training images and labels that are stored in numpy array
"""
training_set = numpy.load('numpy_training_datasets/microexpstcnn_images.npy')
traininglabels =numpy.load('numpy_training_datasets/microexpstcnn_labels.npy')
"""


# Learnet model
model = build(height=224, width=224, channels=62, classes=8)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])
model.summary()

# Load pre-trained weights
# """
# model.load_weights('weights_microexpstcnn/weights-improvement-53-0.88.hdf5')
# """

# filepath="weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# training_set2 = training_set.copy()
# training_set.resize(96,64,64,96)


# Save validation set in a numpy array
"""
numpy.save('numpy_validation_dataset/microexpstcnn_val_images.npy', validation_images)
numpy.save('numpy_validation_dataset/microexpstcnn_val_labels.npy', validation_labels)
"""

# Load validation set from numpy array
"""
validation_images = numpy.load('numpy_validation_datasets/microexpstcnn_val_images.npy')
validation_labels = numpy.load('numpy_validation_datasets/microexpstcnn_val_labels.npy')
"""
# Training the model
avg = 0.0
import random

for i in range(1, 11):
    n = random.randint(0,22)
    x_train, x_test, y_train, y_test = train_test_split(training_set, traininglabels, test_size=0.2, random_state=n)
    hist = model.fit(x_train, y_train, batch_size=25, epochs=100, shuffle=True)
    predictions = model.predict(x_test)
    y_pred = numpy.argmax(predictions, axis=1)
    y_true = numpy.argmax(y_test, axis=1)
    cfm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(acc)
    avg += acc
print(avg/10)

# def project_perturbation(data_point,p,perturbation ):
#     if p == 2:
#         perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
#     elif p == np.inf:
#         perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
#     return perturbation

# delta = 0.2
# max_iter_uni=20
# num_classes = 10
# overshoot=0.2
# max_iter_df=20
# xi=10
# p=np.inf
# fooling_rate = 0.0
# iter = 0
# v=np.zeros([64,64])
# net = mdl.ConvNet()
# #validation_labels = numpy.argmax(validation_labels, axis=1)
# transformer = transforms.ToTensor()
# train_images2, validation_images2, train_labels2, validation_labels2 =  train_test_split(training_set2, traininglabels, test_size=0.5, random_state=2)
# validation_labels2 = numpy.argmax(validation_labels2, axis=1)

# while fooling_rate < 1-delta and iter < max_iter_uni:
#         print("Iteration  ", iter)
#         predictions = model.predict(validation_images)
#         predictions_labels = numpy.argmax(predictions, axis=1)


#         path1 = os.path.join(ori_path,str(iter)+"/")
#         os.mkdir(path1)

#         path3 = os.path.join(pert_path,str(iter)+"/")
#         os.mkdir(path3)
#         for index in range (len(validation_labels)):
#                 v = v.reshape((v.shape[0], -1))

#                 # Feeding the original image to the network and storing the label returned
#                 r2 = validation_labels[index]

#                 # # Generating a perturbed image from the current perturbation v and the original image
#                 # per_img = Image.fromarray(transformer2(cur_img)+v.astype(np.uint8))
#                 # per_img1 = transformer1(transformer2(per_img))[np.newaxis, :].to(device)

#                 # Feeding the perturbed image to the network and storing the label returned
#                 r1 = predictions_labels[index]

#                 # If the label of both images is the same, the perturbation v needs to be updated
#                 if r1 == r2:
#                         print(">> k =", index, ', pass #', iter, end='      ')

#                 # Finding a new minimal perturbation with deepfool to fool the network on this image
#                 img_tensor = torch.tensor(validation_images2[index])

#                 path2 = os.path.join(path1+str(index)+"/")
#                 os.mkdir(path2)
#                 for i in range(96):
#                         name = str(iter)+"/"+str(index)+"/"+str(i)+".jpg"
#                         frame_img = img_tensor[:, :, :, i]

#                         torchvision.utils.save_image(frame_img, ori_path+name, normalize = True)

#                         dr, iter_k, label, k_i, pert_image = deepfool.deepfool(frame_img, net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

#                         # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
#                         if iter_k < max_iter_df-1:

#                                 # framing = imagenet.ImageNet.get_framing(1)
#                                 # input_att, _ = framing(input=img_tensor)
#                                 # with_frame = input_att.numpy()
#                                 # with_frame = cv2.resize(with_frame, (image_rows, image_columns), interpolation = cv2.INTER_AREA)

#                                 v[:, :] += dr[0,0, :, :]

#                                 v = project_perturbation( xi, p,v)

#         name = str(iter)+".jpg"
#         torchvision.utils.save_image(transformer(v), v_path+name, normalize = True)
#         val_imgs = validation_images2.copy()
#         for i in range (len(validation_images2)):
#                 path4 = os.path.join(path3,str(i)+"/")
#                 os.mkdir(path4)
#                 val_img = validation_images2[i]
#                 val_img_tensor = torch.tensor(validation_images2[i])
#                 for j in range(96):
#                         name = str(iter)+"/"+str(i)+"/"+str(j)+".jpg"
#                         frame_img_tensor = val_img_tensor[:, :, :, i]
#                         frame_img_tensor += transformer(v).float()

#                         frame_img = val_img[:, :, :, i]
#                         frame_img += v

#                         torchvision.utils.save_image(frame_img_tensor, pert_path+name, normalize = True)
#                         val_imgs[i][:, :, :, i] = frame_img
#         val_imgs.resize(48,64,64,96)
#         predictions = model.predict(val_imgs)
#         predictions_labels = numpy.argmax(predictions, axis=1)
#         cfm = confusion_matrix(validation_labels2, predictions_labels)
#         print(cfm)
#         if len(cfm)==3: fooling_rate = (cfm[0][0]+cfm[1][1]+cfm[2][2])/48
#         else: fooling_rate = (cfm[0][0]+cfm[1][1])/48
#         print(fooling_rate)
#         iter = iter + 1

# end = time.time()

# print(" ============ Time taken :",end=" ")
# print(end-start,end =" ===========\n")
