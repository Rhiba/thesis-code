import sys
import os
import sys
sys.path.append('../..')
sys.path.append('../../IntegratedGradients')
import numpy as np
import deepbayesHF
import deepbayesHF.optimizers as optimizers
from deepbayesHF import PosteriorModel
from deepbayesHF.analyzers import FGSM
from deepbayesHF.analyzers import eps_LRP
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import cv2
import random
import matplotlib.pyplot as plt
from collections import namedtuple
import tensorflow as tf

from IntegratedGradients import *

if not len(sys.argv) == 6:
    print('Usage: ./get_mnist_exp.py index model_path input_path output_path num_classes')
    sys.exit(0)

model_name = sys.argv[2]
bayes_model = PosteriorModel(model_name)

X = np.load(sys.argv[3])

num_classes = int(sys.argv[5])

model_weights = bayes_model.sample()

tf.compat.v1.disable_eager_execution()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                                  input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model_weights = bayes_model.sample()
model.set_weights(model_weights)
#print(model.predict(X))
y = np.argmax(np.array(model.predict(X)).flatten())
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X,y.reshape(1,*y.shape),epochs=1,batch_size=1)
model.set_weights(model_weights)
#print(model.predict(X))

'''
bayes_model.set_weights(bayes_model.sample())
y = np.argmax(np.array(bayes_model._predict(X)).flatten())

model = bayes_model.model
'''

ig = integrated_gradients(model)
exs = []
for i in range(num_classes):
    exs.append(ig.explain(X.reshape(28,28,1),outc=i))

exs = np.array(exs)

np.save(f'{sys.argv[4]}/e{sys.argv[1]}.npy',exs,False)
