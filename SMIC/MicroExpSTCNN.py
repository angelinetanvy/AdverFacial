import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
import numpy as np
import torch
import torch.nn as nn
import deepfool
import Convnet as mdl
from torchvision import transforms
import torchvision
import time

start = time.time()

K.set_image_dim_ordering('th')

image_rows, image_columns, image_depth = 64, 64, 18

training_list = []

negativepath = 'D:/Projects/FIT3161 - FYP/SMIC/SMIC-E_raw image/HS_long/Sorted/negative/'
positivepath = 'D:/Projects/FIT3161 - FYP/SMIC/SMIC-E_raw image/HS_long/Sorted/positive/'
surprisepath = 'D:/Projects/FIT3161 - FYP/SMIC/SMIC-E_raw image/HS_long/Sorted/surprise/'
ori_path = 'D:/Projects/FIT3161 - FYP/Main/AdverFacial/SMIC/original/'
v_path = 'D:/Projects/FIT3161 - FYP/Main/AdverFacial/SMIC/v/'
pert_path = 'D:/Projects/FIT3161 - FYP/Main/AdverFacial/SMIC/perturbed/'

directorylisting = os.listdir(negativepath)
for video in directorylisting:
    videopath = negativepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

directorylisting = os.listdir(positivepath)
for video in directorylisting:
    videopath = positivepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

directorylisting = os.listdir(surprisepath)
for video in directorylisting:
    videopath = surprisepath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv2.imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    training_list.append(videoarray)

training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)
traininglabels = numpy.zeros((trainingsamples, ), dtype = int)

traininglabels[0:66] = 0
traininglabels[66:113] = 1
traininglabels[113:156] = 2

traininglabels = np_utils.to_categorical(traininglabels, 3)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))

for h in range(trainingsamples):
    training_set[h][0][:][:][:] = trainingframes[h, :, :, :]

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

# MicroExpSTCNN Model
model = Sequential()
model.add(Convolution3D(32, (3, 3, 15), input_shape=(1, image_rows, image_columns, image_depth), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

model.summary()

filepath="weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Load pre-trained weights

model.load_weights('weights_microexpstcnn/weights-improvement-26-0.69.hdf5')


# Save validation set in a numpy array
# numpy.save('numpy_validation_dataset/microexpstcnn_val_images.npy', validation_images)
# numpy.save('numpy_validation_dataset/microexpstcnn_val_labels.npy', validation_labels)

# Load validation set from numpy array
"""
validation_images = numpy.load('numpy_validation_datasets/microexpstcnn_val_images.npy')
validation_labels = numpy.load('numpy_validation_dataset/microexpstcnn_val_labels.npy')
"""
avg = 0.0
for i in range(1,11):

    # Spliting the dataset into training and validation sets
    train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=i)


    # Training the model
    hist = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 16, nb_epoch = 100, shuffle=True)

    # Finding Confusion Matrix using pretrained weights

    predictions = model.predict(validation_images)
    predictions_labels = numpy.argmax(predictions, axis=1)
    validation_labels = numpy.argmax(validation_labels, axis=1)
    cfm = confusion_matrix(validation_labels, predictions_labels)
    print (cfm)
    acc = (cfm[0][0]+cfm[1][1]+cfm[2][2])/32
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
# max_iter_uni=np.inf
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


# while fooling_rate < 1-delta and iter < max_iter_uni:
#         print("Iteration  ", iter)
#         predictions = model.predict(validation_images)
#         predictions_labels = numpy.argmax(predictions, axis=1)

#         path1 = os.path.join(ori_path,str(iter)+"/")    
#         #os.mkdir(path1)

#         path3 = os.path.join(pert_path,str(iter)+"/")    
#         #os.mkdir(path3)
#         for index in range (len(validation_labels)):
#             v = v.reshape((v.shape[0], -1))

#             # Feeding the original image to the network and storing the label returned
#             r2 = validation_labels[index]

#             # # Generating a perturbed image from the current perturbation v and the original image
#             # per_img = Image.fromarray(transformer2(cur_img)+v.astype(np.uint8))
#             # per_img1 = transformer1(transformer2(per_img))[np.newaxis, :].to(device)

#             # Feeding the perturbed image to the network and storing the label returned
#             r1 = predictions_labels[index]

#             # If the label of both images is the same, the perturbation v needs to be updated
#             if r1 == r2:
#                     print(">> k =", index, ', pass #', iter, end='      ')

#             # Finding a new minimal perturbation with deepfool to fool the network on this image
#             img_tensor = torch.tensor(validation_images[index])

#             path2 = os.path.join(path1+str(index)+"/")
#             #os.mkdir(path2)
#             for i in range(18):
#                 name = str(iter)+"/"+str(index)+"/"+str(i)+".jpg"
#                 frame_img = img_tensor[:, :, :, i]
                
#                 #torchvision.utils.save_image(frame_img, ori_path+name, normalize = True)

#                 dr, iter_k, label, k_i, pert_image = deepfool.deepfool(frame_img, net, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

#                 # Adding the new perturbation found and projecting the perturbation v and data point xi on p.
#                 if iter_k < max_iter_df-1:

#                         # framing = imagenet.ImageNet.get_framing(1)
#                         # input_att, _ = framing(input=img_tensor)
#                         # with_frame = input_att.numpy()
#                         # with_frame = cv2.resize(with_frame, (image_rows, image_columns), interpolation = cv2.INTER_AREA)

#                         v[:, :] += dr[0,0, :, :]

#                         v = project_perturbation( xi, p,v)

#         name = str(iter)+".jpg"
#         #torchvision.utils.save_image(transformer(v), v_path+name, normalize = True)
#         val_imgs = validation_images.copy()
#         for i in range (len(validation_images)):
#             path4 = os.path.join(path3,str(i)+"/")    
#             #os.mkdir(path4)
#             val_img = validation_images[i]
#             val_img_tensor = torch.tensor(validation_images[i])
#             for j in range(18):
#                 name = str(iter)+"/"+str(i)+"/"+str(j)+".jpg"
#                 frame_img_tensor = val_img_tensor[:, :, :, j]
#                 frame_img_tensor += transformer(v).float()

#                 frame_img = val_img[:, :, :, j]
#                 frame_img += v

#                 #torchvision.utils.save_image(frame_img_tensor, pert_path+name, normalize = True)
#                 val_imgs[i][:, :, :, j] = frame_img               
#         predictions = model.predict(val_imgs)
#         predictions_labels = numpy.argmax(predictions, axis=1)
#         cfm = confusion_matrix(validation_labels, predictions_labels)
#         print(cfm)
#         acc = (cfm[0][0]+cfm[1][1]+cfm[2][2])/32
#         fooling_rate = 1-acc
#         print(acc)
#         iter = iter + 1

# end = time.time()
# print ("==== Time: ",end="")
# print(end-start,end=" ====")