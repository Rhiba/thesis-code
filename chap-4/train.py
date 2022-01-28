import tensorflow as tf
import pickle
import json
import keras
import keras.backend as K
from keras.layers import Convolution2D, Input, Dense, Lambda, Wrapper
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD # subject to change
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, RemoteMonitor
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import scipy

keras.backend.set_image_dim_ordering('tf')
dropout = 0.1
p = 1 - dropout
lengthscale = 0.1
# got tau from BO experiment using spearmint
tau = 53.596496582
intervals = 151
threshold = 31

def pilotnet_model(reg):

	# Try changing activations etc, or adding dropout
	model = Sequential()
	model.add(Convolution2D(filters=24,kernel_size=5,strides=2,activation='relu',input_shape=(66,200,3),W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Convolution2D(filters=36,kernel_size=5,strides=2,activation='relu',W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Convolution2D(filters=48,kernel_size=5,strides=2,activation='relu',W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Convolution2D(filters=64,kernel_size=3,strides=1,activation='relu',W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Convolution2D(filters=64,kernel_size=3,strides=1,activation='relu',W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Flatten())
	model.add(Dense(1164,W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Dense(512,activation='relu',W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Dense(256,activation='relu',W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Dense(128,activation='relu',W_regularizer=reg()))
	model.add(Dropout(dropout))
	model.add(Dense(intervals+1,activation='softmax',W_regularizer=reg()))
	model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['acc'])
	return model

def bucket(x,prec,base):
	return round(base*round(float(x)/base),prec)

def main():

	data_folder = '../pilotnet-data/data'
	f = open(os.path.join(data_folder,'driving_log.csv'),'r')
	reader = csv.reader(f)
	headers = next(reader,None)
	data_columns = {}
	for h in headers:
		data_columns[h] = []

	for row in reader: 
		for h,v in zip(headers,row): 
			if h == 'steering' or h=='speed': 
				data_columns[h].append(float(str.strip(v)))
			else:
				data_columns[h].append(v)
	
	image_paths = []
	steering_angles = []
	for idx, angle in enumerate(data_columns['steering']):
		if angle == 0:
			if data_columns['speed'][idx] > threshold:
				steering_angles = steering_angles + [angle]
				image_paths = image_paths + [data_columns['center'][idx]]
		else:
			steering_angles = steering_angles + [angle]
			image_paths = image_paths + [data_columns['center'][idx]]


	image_paths_orig = image_paths
	steering_angles2 = [-x for x in steering_angles]
	steering_angles = steering_angles + steering_angles2

	# convert into a classification problem, we have from -1 to 1, try dividing into 200 buckets?
	interval = 2/intervals
	steering_angles = [bucket(x,3,interval) for x in steering_angles]
	#need to scale angles to be integers from 0 up for to_categorical
	steering_angles = keras.utils.to_categorical([(x+1)/interval for x in steering_angles],intervals+1)

	images = np.array([np.float32(imresize(imread(os.path.join(data_folder,im)), size=(66, 200))) / 255 for im in image_paths]+[np.fliplr(np.float32(imresize(imread(os.path.join(data_folder,im)), size=(66, 200)))) / 255 for im in image_paths_orig])


	image_mean = np.mean(images,0)
	image_std = np.std(images,0)
	image_std[image_std==0]=1

	images = (images - np.full(images.shape,image_mean))/np.full(images.shape,image_std)
	with open('stats_im.txt','wb') as outfile:
		pickle.dump(image_mean,outfile)
	with open('stats_is.txt','wb') as outfile:
		pickle.dump(image_std,outfile)

	test_percent = 0.1
	test_images = []
	train_images = []
	test_angles = []
	train_angles = []
	print(images.shape)
	print("steering angles:" + str(steering_angles.shape))
	for i in range(len(images)):
		if i % (test_percent*100) == 0:
			test_images.append(images[i])
			test_angles.append(steering_angles[i])
		else:
			train_images.append(images[i])
			train_angles.append(steering_angles[i])

	test_images = np.array(test_images)
	train_images = np.array(train_images)
	test_angles = np.array(test_angles)
	train_angles = np.array(train_angles)
	print("test angles: " + str(test_angles.shape))

	N = len(train_images)
	reg = lambda: l2(lengthscale**2 * (1 - dropout) / (2. * N * tau))

	model = pilotnet_model(reg)

	checkpointer = ModelCheckpoint(
		filepath="../intermediate-models/{epoch:02d}-{val_loss:.12f}.hdf5",
		verbose=1,
		save_best_only=True
	)

	lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1, mode=min)

	epochs = 20
	batch_size = 128

	print('Training')
	model.fit(train_images,train_angles,validation_data=(test_images,test_angles),epochs=epochs,batch_size=batch_size,callbacks=[lr_plateau,checkpointer],shuffle=True)

	model.save('model-p_'+ str(dropout) + '-l_'+ str(lengthscale) + '-t_' + str(tau) +'.h5')
	model.save_weights('weights-p_'+ str(dropout) + '-l_'+ str(lengthscale) + '-t_' + str(tau) +'.h5')
	print('Done')

if __name__=="__main__":
	main()
