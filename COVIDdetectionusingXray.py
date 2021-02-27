#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 01:16:15 2021

@author: intel123
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

Classifier=Sequential();

Classifier.add(Conv2D(64,(3,3),input_shape=(64,64,3),activation='relu'))

Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Conv2D(32,(3,3),activation='relu'))

Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Flatten())

Classifier.add(Dense(units=104, activation='relu'))
Classifier.add(Dense(units=1, activation='sigmoid'))

Classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.4,
                                   zoom_range = 0.3,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/home/intel123/AS/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 4,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/home/intel123/AS/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 4,
                                            class_mode = 'binary')


Classifier.fit_generator(training_set,
                         steps_per_epoch = 40,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 8)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/intel123/AS/dataset/covid_or_normal.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = Classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Normal'
    print(prediction)
else:
    prediction = 'COVID'
    print(prediction)