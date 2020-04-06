##
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Dropout,Flatten,Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import time
from keras.applications.resnet50 import preprocess_input,decode_predictions
from keras.applications.resnet50 import ResNet50
from extract_bottleneck_features import *
##
# Step 0 | Import Datasets

def load_dataset(path):
    data=load_files(path)
    dog_files=np.array(data['filenames'])
    dog_targets=np_utils.to_categorical(np.array(data['target']),133)
    return dog_files,dog_targets

train_files,train_targets=load_dataset('data/dog_images/train')
valid_files,valid_targets=load_dataset('data/dog_images/valid')
test_files,test_targets=load_dataset('data/dog_images/test')

dog_names=[item[20:-1] for item in sorted(glob('data/dog_images/train/*/'))]

print('There are %d total dog categories.'%len(dog_names))
print('There are %s total dog images.'%len(np.hstack([train_files,valid_files,test_files])))
print('There are %d training dog images.'%len(train_files))
print('There are %d validation dog images.'%len(valid_files))
print('There are %d test dog images.'%len(test_files))

##
# Import human datas
import random
random.seed(8675309)

human_files=np.array(glob('data/lfw/lfw/*/*'))
random.shuffle(human_files)

print('There are %d total human images.'%len(human_files))
##
# Step 1 | Detect humans

# use opencv's implementation of Haar feature-based cascade classifiers to detect
# human faces in images.

import cv2
import matplotlib.pyplot as plt

face_cascade=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
# for i in range(200):
#     img=cv2.imread(human_files[i])
#     # Before using any of the face detectors, it is standard procedure
#     # to convert the images to grayscale.
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # The detectMultiScale function executes the classifier stored in
#     # face_cascade and takes the grayscale image as a parameter.
#     faces=face_cascade.detectMultiScale(gray)
#     print('Number of faces detected:',len(faces))
#     for (x,y,w,h) in faces:
#        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     cv_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     plt.imshow(cv_rgb)
#     plt.show()
#     time.sleep(0.2)

# A function returns True if a human face is detected.
def face_detector(img_path):
    img=cv2.imread(img_path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray)
    return len(faces)>0

# TODO: Following code is used to test the accuracy of the human
# face detector.
human_files_short=human_files[:100]
dog_files_short=train_files[:100]

humans_count=0
dogs_count=0

for img in human_files_short:
    if face_detector(img)==True:
        humans_count+=1
for img in dog_files_short:
    if face_detector(img)==True:
        dogs_count+=1
print('%.1f%% images of humans are correctly classified as humans.'%humans_count)
print('%.1f%% images of dogs are misclassified as humans.'%dogs_count)

# This algorithmic choice necessitates that we communicate to the user
# that we accept human images only when they provide a clear view of a
# face (otherwise, we risk having unneccessarily frustrated users!).
#  Instead we should build an algorithm based on CNN with a training
#  data that should include a diverse set of images from a wide variety
#  of angles, and lighting conditions.

##
# Step 2 | Detect Dogs

# We use a pre-trained ResNet-50 model to detect dogs in images.The weights
# of ResNet-50 we used have been trained on ImageNet.

# define ResNet50 model
ResNet50_model=ResNet50(weights='imagenet')

# TODO: Pre-process the Data
# When using Tensorflow as backend, Keras CNNs require a 4D array(4D tensor)
# as input,with shape
# (nb_samples,rows,colums,channels)
# where nb_samples corresponds to the total number of images(or samples)
# and rows,columns, and channels correspond to the properties of each
# image,respectively.

from keras.preprocessing import image
from tqdm import tqdm

# The path_to_tensor function below takes a string-valued file path to
# a color image as input and returns a 4D tensor suitable for supplying
# to a keras CNN.
# load image -> resize to a square image that is 224x224 pixels
# -> the image is converted to an array -> resize to a 4D tensor
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img=image.load_img(img_path,target_size=(224,224))
    # convert PIL.Image.Image type to 3D tensor with shape (224,224,3)
    x=image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1,224,224,3)
    return np.expand_dims(x,axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors=[path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# TODO: Some additional and must processing for Keras

# The below processing is implemented in the imported function preprocess_input
# 4D tensor for ResNet-50 -> RGB image is converted to BGR
# -> normalization, every pixel must be subtracted the mean pixel
# (expressed in RGB as [103.939, 116.779, 123.68])



def ResNet50_predict_labels(img_path):
    # return prediction vector for image located at img_path
    img=preprocess_input(path_to_tensor(img_path))
    # The predict method returns an array whose i-th entry is the model's
    # predicted probability that the image belongs to the i-th ImageNet category.
    return np.argmax(ResNet50_model.predict(img))

# While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a),
# you will notice that the categories corresponding to dogs appear in an uninterrupted
# sequence and correspond to dictionary keys 151-268, inclusive,
# to include all categories from `'Chihuahua'` to `'Mexican hairless'`.
# Thus, in order to check to see if an image is predicted to contain
# a dog by the pre-trained ResNet-50 model, we need only check if the
# `ResNet50_predict_labels` function above returns a value between
# 151 and 268 (inclusive).

callback_count=0
def dog_detector(img_path):
    global callback_count
    callback_count+=1
    prediction=ResNet50_predict_labels(img_path)
    print('%d time test, ResNet50_predict_labels : %d'%(callback_count,prediction))
    return ((prediction<=268)&(prediction>=151))

def dog_breed_detector(img_file):
    prediction=ResNet50_predict_labels(img_file)
    return prediction

# TODO: Test the performance of the dog_detector function

# human_files_short=human_files[:100]
# dog_files_short=train_files[:100]
# validation_files=[item for item in sorted(valid_files)]
#
# humans_count=0
# dogs_count=0
# for img in human_files_short:
#     if dog_detector(img)==True:
#         humans_count+=1
#
# for img in dog_files_short:
#     if dog_detector(img)==True:
#         dogs_count+=1
# This is using for test the performance on validation_dataset,
# if needed,turn it on.

# for img in valid_files:
#     if dog_detector(img)==True:
#         dogs_count+=1

# print('%.1f%% images of humans are misclassified as dogs.'%humans_count)
# print('%.1f%% images of dogs are correctly classified as dogs.'%dogs_count)

# TODO:These files are the locale images corresponding with the ImageNet directory of 1000 categories.
# The ImageNet catogories link is https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a,
# The dog labels' range is (151,268)
#
# affenpiinscher_252 = validation_files[0:8]
# afghan_160 = validation_files[8:15]
# airedale_191 = validation_files[15:22]
# austrilian_terrier_193 = validation_files[87:93]
# beagle_162 = validation_files[110:117]
# chihuahua_151 = validation_files[333:340]
# maltese_153 = validation_files[666:672]
# pekingese_154 = validation_files[753:759]
# papillon_157 = validation_files[741:749]
# basset_161 = validation_files[101:110]
# bloodhound_163 = validation_files[183:191]
#
# list_of_breeds=[affenpiinscher_252,afghan_160,airedale_191,austrilian_terrier_193,beagle_162,
#                 chihuahua_151,maltese_153,pekingese_154,papillon_157,basset_161,bloodhound_163]
# list_of_breeds_name=['affenpiinscher_252','afghan_160','airedale_191','austrilian_terrier_193','beagle_162',
#                      'chihuahua_151','maltese_153','pekingese_154','papillon_157','basset_161','bloodhound_163']
# list_of_breeds_num=[]
# for i in range(list_of_breeds_name.__len__()):
#     list_of_breeds_num.append(int(list_of_breeds_name[i][-3:]))

# TODO: Use ResNet-50 to classify dog breeds
# This is for testing the ResNet-50's performence on clssifying dog breed
# total_test_count_of_resnet50=0
# right_test_count_of_resnet50=0
# for i in range(list_of_breeds.__len__()):
#     for img in list_of_breeds[i]:
#         total_test_count_of_resnet50+=1
#         predict=dog_breed_detector(img)
#         if predict==list_of_breeds_num[i]:
#             right_test_count_of_resnet50+=1
#         print('The prediction of '+list_of_breeds_name[i]+' is: %d.'%predict)
# print('Accuracy of ResNet50 is %.4f%%.'%(100.0*(right_test_count_of_resnet50/total_test_count_of_resnet50)))

##
# TODO: Step 3 | Create a CNN to classify Dog Breeds (from scratch)

# TODO: Pre-process the Data
#
#
# train_tensors=paths_to_tensor(train_files).astype('float32')/255
# valid_tensors=paths_to_tensor(valid_files).astype('float32')/255
# test_tensors=paths_to_tensor(test_files).astype('float32')/255


#
# model=Sequential()
#
# model.add(Conv2D(filters=16,kernel_size=2,padding='same',activation='relu',input_shape=(224,224,3)))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Dropout(0.4))
# model.add(Flatten())
#
# model.add(Dense(512,activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(133,activation='softmax'))
#
# model.summary()
#
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#
#
#
# epochs=60
#
# # The filepath directory 'saved_models' must be created earlierly.
# # at the same folder with the .py file.
# checkpointer=ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
#                              verbose=1,save_best_only=True)
#
# start=time.time()
# # Cause our dataset is small, it's recommended to augment the training
# # data, see https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html,
# # but this is not a requirement.
# model.fit(train_tensors,train_targets,
#           validation_data=(valid_tensors,valid_targets),
#           epochs=epochs,batch_size=20,callbacks=[checkpointer],
#           verbose=1)
# end=time.time()
# print('Time taken (in minutes): ',(end-start)/60)
#
# # Load the Model with the Best Validation Loss
# model.load_weights('saved_models/weights.best.from_scratch.hdf5')
#
# # Test the model.
# # Understand the difference among train, valid and test dataset,
# # see https://blog.csdn.net/bornfree5511/article/details/105249766
# dog_breed_predictions=[np.argmax(model.predict(np.expand_dims(tensor,axis=0))) for tensor in test_tensors]
# test_accuracy=100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets,axis=1))/len(dog_breed_predictions)
# print('Test accuracy: %.4f%%'%test_accuracy)

##
# TODO:Step 4 | Use transfer learning to train a CNN

# Download Bottleneck Features(BF). BF is a neccessery part for
# transfer learning. It's the bridge between pre-trained model's
# weights and your own layers. More information please visit Google.
# Download link: https://pan.baidu.com/s/1SSExIExmSurG85MqE9eF4Q, key:iq6w
# Create a folder named 'bottleneck_features' at the same folder with current .py file
# Extract the 'bottleneck_feature.zip' and move all the files to the
# 'bottleneck_features' folder.

epochs=5000

# bottleneck_features_VGG16=np.load('bottleneck_features/DogVGG16Data.npz')
# train_VGG16=bottleneck_features_VGG16['train']
# valid_VGG16=bottleneck_features_VGG16['valid']
# test_VGG16=bottleneck_features_VGG16['test']
#
# bottleneck_features_VGG19=np.load('bottleneck_features/DogVGG19Data.npz')
# train_VGG19=bottleneck_features_VGG19['train']
# valid_VGG19=bottleneck_features_VGG19['valid']
# test_VGG19=bottleneck_features_VGG19['test']
#
# # This ResNet-50 is different from the above one, because we add some layers after the ResNet-50 model's
# # output called transfer learning.
# bottleneck_features_Resnet50=np.load('bottleneck_features/DogResnet50Data.npz')
# train_Resnet50=bottleneck_features_Resnet50['train']
# valid_Resnet50=bottleneck_features_Resnet50['valid']
# test_Resnet50=bottleneck_features_Resnet50['test']

# bottleneck_features_InceptionV3=np.load('bottleneck_features/DogInceptionV3Data.npz')
# train_InceptionV3=bottleneck_features_InceptionV3['train']
# valid_InceptionV3=bottleneck_features_InceptionV3['valid']
# test_InceptionV3=bottleneck_features_InceptionV3['test']

bottleneck_features_Xception=np.load('bottleneck_features/DogXceptionData.npz')
train_Xception=bottleneck_features_Xception['train']
valid_Xception=bottleneck_features_Xception['valid']
test_Xception=bottleneck_features_Xception['test']

# # The model uses the the pre-trained VGG-16 model as a fixed feature extractor,
# # where the last convolutional output of VGG-16 is fed as input
# # to our model. We only add a global average pooling layer and a
# # fully connected layer, where the latter contains one node for
# # each dog category and is equipped with a softmax.
#
# VGG16_model=Sequential()
# VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
# VGG16_model.add(Dense(133,activation='softmax'))
# VGG16_model.summary()
# VGG16_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
#                     metrics=['accuracy'])
# VGG16_checkpointer=ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5',
#                                    verbose=1,save_best_only=True)
# VGG16_start=time.time()
# VGG16_model.fit(train_VGG16,train_targets,
#                 validation_data=(valid_VGG16,valid_targets),
#                 epochs=epochs,batch_size=20,callbacks=[VGG16_checkpointer],
#                 verbose=1)
# VGG16_end=time.time()
# print('VGG16 time taken (in minutes): ',(VGG16_end-VGG16_start)/60)
#
# VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
# # get index of predicted dog breed for each image in test set
# VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]
#
# # report test accuracy
# VGG16_test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
# print('VGG16 test accuracy: %.4f%%' % VGG16_test_accuracy)
#
#
# VGG19_model=Sequential()
# VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
# VGG19_model.add(Dense(133,activation='softmax'))
# VGG19_model.summary()
# VGG19_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
#                     metrics=['accuracy'])
# VGG19_checkpointer=ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5',
#                                    verbose=1,save_best_only=True)
# VGG19_start=time.time()
# VGG19_model.fit(train_VGG19,train_targets,
#                 validation_data=(valid_VGG19,valid_targets),
#                 epochs=epochs,batch_size=20,callbacks=[VGG19_checkpointer],
#                 verbose=1)
# VGG19_end=time.time()
# print('VGG19 time taken (in minutes): ',(VGG19_end-VGG19_start)/60)
#
# VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')
# # get index of predicted dog breed for each image in test set
# VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]
#
# # report test accuracy
# VGG19_test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
# print('VGG19 test accuracy: %.4f%%' % VGG19_test_accuracy)
#
#
#
# Resnet50_model=Sequential()
# Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
# Resnet50_model.add(Dense(133,activation='softmax'))
# Resnet50_model.summary()
# Resnet50_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
#                        metrics=['accuracy'])
# Resnet50_checkpointer=ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
#                                       verbose=1,save_best_only=True)
# Resnet50_start=time.time()
# Resnet50_model.fit(train_Resnet50,train_targets,
#                    validation_data=(valid_Resnet50,valid_targets),
#                    epochs=epochs,batch_size=20,callbacks=[Resnet50_checkpointer],
#                    verbose=1)
# Resnet50_end=time.time()
# print('Resnet50 time taken (in minutes): ',(Resnet50_end-Resnet50_start)/60)
#
# Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')
# # get index of predicted dog breed for each image in test set
# Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]
#
# # report test accuracy
# Resnet50_test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
# print('Resnet50 test accuracy: %.4f%%' % Resnet50_test_accuracy)

#
#
# #
# InceptionV3_model=Sequential()
# InceptionV3_model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))
# InceptionV3_model.add(Dense(133,activation='softmax'))
# InceptionV3_model.summary()
# InceptionV3_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
#                           metrics=['accuracy'])
# InceptionV3_checkpointer=ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5',
#                                          verbose=1,save_best_only=True)
# InceptionV3_start=time.time()
# InceptionV3_model.fit(train_InceptionV3,train_targets,
#                       validation_data=(valid_InceptionV3,valid_targets),
#                       epochs=epochs,batch_size=20,callbacks=[InceptionV3_checkpointer],
#                       verbose=1)
# InceptionV3_end=time.time()
# print('InceptionV3 time taken (in minutes): ',(InceptionV3_end-InceptionV3_start)/60)
#
# InceptionV3_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
# # get index of predicted dog breed for each image in test set
# InceptionV3_predictions = [np.argmax(InceptionV3_model.predict(np.expand_dims(feature, axis=0))) for feature in test_InceptionV3]
#
# # report test accuracy
# InceptionV3_test_accuracy = 100*np.sum(np.array(InceptionV3_predictions)==np.argmax(test_targets, axis=1))/len(InceptionV3_predictions)
# print('InceptionV3 test accuracy: %.4f%%' % InceptionV3_test_accuracy)



Xception_model=Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dense(133,activation='softmax'))
Xception_model.summary()
Xception_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
                       metrics=['accuracy'])
Xception_checkpointer=ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5',
                                      verbose=1,save_best_only=True)
Xception_start=time.time()
Xception_model.fit(train_Xception,train_targets,
                   validation_data=(valid_Xception,valid_targets),
                   epochs=epochs,batch_size=20,callbacks=[Xception_checkpointer],
                   verbose=1)
Xception_end=time.time()
print('Xception time taken (in minutes): ',(Xception_end-Xception_start)/60)

Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
# get index of predicted dog breed for each image in test set
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception]

# report test accuracy
Xception_test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Xception test accuracy: %.4f%%' % Xception_test_accuracy)

# print('VGG16 time taken: ',(VGG16_end-VGG16_start)/60)
# print('VGG19 time taken: ',(VGG19_end-VGG19_start)/60)
# print('Resnet50 time taken: ',(Resnet50_end-Resnet50_start)/60)
# print('InceptionV3 time taken: ',(InceptionV3_end-InceptionV3_start)/60)
# print('Xception time taken: ',(Xception_end-Xception_start)/60)

# print('VGG16 accuracy: %.4f%%'%VGG16_test_accuracy)
# print('VGG19 accuracy: %.4f%%'%VGG19_test_accuracy)
# print('Resnet50 accuracy: %.4f%%'%Resnet50_test_accuracy)
# print('InceptionV3 accuracy: %.4f%%'%InceptionV3_test_accuracy)
# print('Xception accuracy: %.4f%%'%Xception_test_accuracy)
#
# # TODO:Step 5 | Write and test final algorithms
#
# # TODO: Write a function that takes a path to an image as input and returns the dog breed

dog_names=[item[6:].replace('_',' ') for item in dog_names]
# def VGG16_dog_predictor(img_path):
#     bottleneck_feature=extract_VGG16(path_to_tensor(img_path))
#     predicted_vector=VGG16_model.predict(bottleneck_feature)
#     return dog_names[np.argmax(predicted_vector)]
#
# def VGG19_dog_predictor(img_path):
#     bottleneck_feature=extract_VGG19(path_to_tensor(img_path))
#     predicted_vector=VGG16_model.predict(bottleneck_feature)
#     return dog_names[np.argmax(predicted_vector)]
#
# def Resnet50_dog_predictor(img_path):
#     bottleneck_feature=extract_Resnet50(path_to_tensor(img_path))
#     bottleneck_feature=np.expand_dims(bottleneck_feature,axis=0)
#     bottleneck_feature=np.expand_dims(bottleneck_feature,axis=0)
#     predicted_vector=Resnet50_model.predict(bottleneck_feature)
#     return dog_names[np.argmax(predicted_vector)]
# #
# def InceptionV3_dog_predictor(img_path):
#     bottleneck_feature=extract_InceptionV3(path_to_tensor(img_path))
#     predicted_vector=InceptionV3_model.predict(bottleneck_feature)
#     return dog_names[np.argmax(predicted_vector)]
#

def Xception_dog_predictor(img_path):
    bottleneck_feature=extract_Xception(path_to_tensor(img_path))
    print('bottleneck_feature shape is : ',bottleneck_feature.shape)
    predicted_vector=Xception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

# # TODO: Write final algorithm
def final_dog_predictor(img_path):
    img=cv2.imread(img_path)
    cv_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()
# # We use the weights of ResNet-50 to judge whether an image is dog,
# # cause though breeds classifier's accuracy is about 87%,
# # the accuracy of whether a dog is 100%

    if dog_detector(img_path):
        #print("This is a Dog.")
        # why use return? Cause return will send the control flow to main
        # thread, it's like if ... else if... else
        # It reduce the judgement times.
        return print('Predicted breed is ...\n{}'.format(Xception_dog_predictor(img_path)))
        #return print('Predicted breed is ...\n{}'.format(Resnet50_dog_predictor(img_path)))
        #return print('Predicted breed is ...\n{}'.format(InceptionV3_dog_predictor(img_path)))
# # opencv's cascade's accuracy is not 100%,so later we will use
# # DL to replace cascade.
    if face_detector(img_path):
        print('This is a Human.')
        return

    else:
        return print('Sorry! no Dog or Human is detected.')

# Test prediction

print('\n')
print('These images are taidi dogs. Test results are below:')
print('----------------------------------------------------')
final_dog_predictor('sample_images/sample_taidi_1.jpg')
final_dog_predictor('sample_images/sample_taidi_2.jpg')
final_dog_predictor('sample_images/sample_taidi_3.jpg')
final_dog_predictor('sample_images/sample_taidi_4.jpeg')
final_dog_predictor('sample_images/sample_taidi_5.jpg')
final_dog_predictor('sample_images/sample_taidi_6.jpg')
print('\n')

print('These images are labuladuo dogs. Test results are below:')
print('--------------------------------------------------------')
final_dog_predictor('sample_images/sample_labuladuo_1.jpg')
final_dog_predictor('sample_images/sample_labuladuo_2.png')
final_dog_predictor('sample_images/sample_labuladuo_3.jpg')
final_dog_predictor('sample_images/sample_labuladuo_4.jpg')
final_dog_predictor('sample_images/sample_labuladuo_5.jpg')
final_dog_predictor('sample_images/sample_labuladuo_6.jpg')
final_dog_predictor('sample_images/sample_6.jpg')
print('\n')

print('These images are jinmao dogs. Test results are below:')
print('-----------------------------------------------------')
final_dog_predictor('sample_images/sample_jinmao_1.jpg')
final_dog_predictor('sample_images/sample_jinmao_2.jpeg')
final_dog_predictor('sample_images/sample_jinmao_3.jpg')
final_dog_predictor('sample_images/sample_jinmao_4.jpeg')
final_dog_predictor('sample_images/sample_jinmao_5.jpg')
final_dog_predictor('sample_images/sample_jinmao_6.jpg')
final_dog_predictor('sample_images/sample_jinmao_7.jpg')
final_dog_predictor('sample_images/sample_jinmao_8.jpg')
final_dog_predictor('sample_images/sample_jinmao_9.jpg')
print('\n')

print('These images are beagle dogs. Test results are below:')
print('-----------------------------------------------------')
final_dog_predictor('sample_images/sample_11.jpg')
print('\n')

print('These images are humans. Test results are below:')
print('------------------------------------------------')
final_dog_predictor('sample_images/sample_1.jpg')
final_dog_predictor('sample_images/sample_2.jpg')
final_dog_predictor('sample_images/sample_3.jpg')
print('\n')

print('These images are not humans or dogs. Test results are below:')
print('------------------------------------------------------------')
final_dog_predictor('sample_images/sample_5.jpg')
final_dog_predictor('sample_images/sample_8.jpg')
final_dog_predictor('sample_images/sample_9.jpg')
print('\n')