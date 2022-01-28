import tensorflow as tf
import math
import operator
import pickle
from tensorflow import atan
import keras
import keras.backend as K
from keras.layers import Convolution2D, Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD # subject to change
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, RemoteMonitor
from keras.utils.generic_utils import get_custom_objects
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.misc import imread, imresize
import scipy
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, render_template
from io import BytesIO
from PIL import Image
import base64
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

keras.backend.set_image_dim_ordering('tf')

sio = socketio.Server()
app = Flask(__name__)
model = None
tau = (((1e-2)**2)*0.95)/(2*14464*1e-6)
model = None
T = 50
predict_stochastic = None
image_mean = None
image_std = None
angle_mean = None
angle_std = None

plotter = 1

entropy_buffer = []
went_buffer = []
mutinf_buffer = []
avmutinf_buffer = []
vratio_buffer = []
var_buffer = []
highest_entropy = 0
highest_vratio = 0
highest_mutinf = 0
highest_var = 0
highest_went = 0
av_entropy = 0
av_vratio = 0
av_mutinf = 0
av_var = 0
av_went = 0
entropy_sum = 0
vratio_sum = 0
mutinf_sum = 0
var_sum = 0
went_sum = 0
number_samples = 0

fig = plt.figure(figsize=(14,8))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
#ax4 = fig.add_subplot(5,1,4)
#ax5 = fig.add_subplot(5,1,5)

@sio.on('telemetry')
def telemetry(sid,data):
	global plotter
	global number_samples
	global highest_entropy
	global highest_vratio
	global highest_mutinf
	global highest_var
	global highest_went
	global av_entropy
	global av_vratio
	global av_mutinf
	global av_var
	global av_went
	global entropy_sum
	global vratio_sum
	global mutinf_sum
	global var_sum
	global went_sum
	imgString = data["image"]
	image = Image.open(BytesIO(base64.b64decode(imgString)))
	image = np.asarray(image)
	image_transform = np.float32(imresize(image, size=(66, 200, 3))) / 255
	steering_angle = 0
	try:
		image_transform = (image_transform - image_mean)/image_std
		Yt_hat = np.array(predict_stochastic([[image_transform for i in range(T)],1]))
		intervals=201
		interval=2/intervals
		steering_angles = np.array([(np.argmax(steering_angle)*interval)-1 for steering_angle in Yt_hat[0]])
		prediction = stats.mode(steering_angles)[0]
		count = stats.mode(steering_angles)[1] 

		preds = Yt_hat[0]
		avs = np.mean(preds,axis=0)

		summation = 0
		for cls in range(preds.shape[1]):
			for steps in range(preds.shape[0]):
				if not preds[steps][cls] ==0:
					summation += preds[steps][cls]*np.log(preds[steps][cls])

		summation = summation/T
		we = []
		for idx, s in enumerate(avs):
			dist = abs(idx-prediction[0])
			v = ((dist+1)/intervals) * s * np.log(s) if s > 0 else 0
			we.append(v)

		weighted_entropy = -1 *np.sum(we)
		entropy = -1 *np.sum([s * np.log(s) for s in avs])
		mutinf = entropy + summation
		vratio = 1 - (count/len(steering_angles))
		var = np.var(steering_angles)

		if not math.isnan(entropy) and not math.isnan(mutinf):
			number_samples += 1
			entropy_sum += entropy
			mutinf_sum += mutinf
			#var_sum += var
			#went_sum += weighted_entropy
			if len(mutinf_buffer) < 100:
				entropy_buffer.append(entropy)
				vratio_buffer.append(vratio)
				mutinf_buffer.append(mutinf)
				var_buffer.append(var)
				went_buffer.append(weighted_entropy)
			else:
				entropy_buffer.pop(0)
				entropy_buffer.append(entropy)
				vratio_buffer.pop(0)
				vratio_buffer.append(vratio)
				mutinf_buffer.pop(0)
				mutinf_buffer.append(mutinf)
				var_buffer.append(var)
				var_buffer.pop(0)
				var_buffer.append(var)
				went_buffer.pop(0)
				went_buffer.append(weighted_entropy)
			av_mutinf = sum(mutinf_buffer[-10:])/len(mutinf_buffer[-10:])
			if len(avmutinf_buffer) < 100:
				avmutinf_buffer.append(av_mutinf)
			else:
				avmutinf_buffer.pop(0)
				avmutinf_buffer.append(av_mutinf)
			av_entropy = entropy_sum/number_samples
			#av_var = var_sum/number_samples
			#av_went = went_sum/number_samples
			
			if entropy > highest_entropy:
				highest_entropy = entropy
			if mutinf > highest_mutinf:
				highest_mutinf = mutinf
			if var > highest_var:
				highest_var = var
			if var > 0.01:
				print("VAR WARNING / ",end="")
			if mutinf > 0.3:
				print("MUTINF WARNING / ",end="")
			if entropy > 3:
				print("ENTROPY WARNING / ", end="")
			if vratio > 0.7:
				print("VRATIO WARNING / ", end = "")
			if weighted_entropy > 0.25:
				print("WEIGHTENTROPY WARNING / ", end= "")



			if weighted_entropy > highest_went:
				highest_went = weighted_entropy
			vratio_sum += vratio
			av_vratio = vratio_sum/number_samples
			if vratio > highest_vratio:
				highest_vratio = vratio
			print("Prediction: " + str(prediction))
			print("Entropy: " + str(entropy))
			print("Max entropy: "+ str(highest_entropy))
			print("Av entropy: " +str(av_entropy))
			print("Weighted Entropy: " + str(weighted_entropy))
			print("Max weighted entropy: "+ str(highest_went))
			print("Av weighted entropy: " +str(av_went))
			print("Var ratio: " + str(vratio))
			print("Max var ratio: "+ str(highest_vratio))
			print("Av var ratio: " +str(av_vratio))
			print("Mutual inf: " + str(mutinf))
			print("Max mutual inf: "+ str(highest_mutinf))
			print("Av mutual inf: " +str(av_mutinf))
			print("Variance: ",str(var))
			print("Max var: "+ str(highest_var))
			print("Av var: " +str(av_var))
			print()

		if avmutinf_buffer[-1] > 0.6:
			print("WARN, high uncertainty")

		thresh = 1
		'''
		if went_buffer[-1] > thresh and went_buffer[-2] > thresh and went_buffer[-3] > thresh and went_buffer[-4] > thresh and went_buffer[-5] > thresh:
			print("ALERT, LONG TIME HIGH VALUES")
		'''
		if plotter == 3:
			plt.clf()
			xs = list(range(len(mutinf_buffer)))
			ys_line = ([0.61]*len(mutinf_buffer))
			'''
			ys1 = entropy_buffer
			ys2 = vratio_buffer
			'''
			ys3 = mutinf_buffer
			#ys4 = var_buffer
			#ys5 = went_buffer
			ys6 = avmutinf_buffer
			'''
			ax1.clear()
			ax1.plot(xs,ys1)
			ax1.set_ylabel('Predictive Entropy')
			'''
			plt.plot(xs,ys3)
			plt.plot(xs,ys6)
			plt.plot(xs,ys_line,'r')
			'''
			ax2.clear()
			ax2.plot(xs,ys2)
			ax2.set_ylabel('Variation Ratio')
			ax3.clear()
			ax3.plot(xs,ys3)
			ax3.set_ylabel('Mutual Information')
			'''
			plt.legend(['mutual information','moving average mutual information'],loc='upper left')
			'''
			ax4.clear()
			ax4.plot(xs,ys4)
			ax5.clear()
			ax5.plot(xs,ys5)
			'''
			plt.draw()
			plt.pause(0.001)
			#plt.pause(0.05)
			#plt.show(block=False)
			plotter = 1
		else:
			plotter = plotter + 1

	except Exception as e:
		print(e)

	send_control(float(prediction),0.05)

@sio.on('connect')
def connect(sid, environ):
	print("connect ", sid)
	send_control(0,0)

def send_control(steering_angle,throttle=0.2):
	try:
		sio.emit("steer",data={'steering_angle':steering_angle.__str__(),'throttle':throttle.__str__()},skip_sid=True)
	except Exception as e:
		print(e)

if __name__=="__main__":
	get_custom_objects().update({'atan':atan})
	model = keras.models.load_model(sys.argv[1])
	predict_stochastic = K.function([model.layers[0].input,K.learning_phase()],[model.layers[-1].output])
	with open('stats_im.txt','rb') as infile:
		image_mean = pickle.load(infile)
	with open('stats_is.txt','rb') as infile:
		image_std = pickle.load(infile)
	app = socketio.Middleware(sio,app)
	eventlet.wsgi.server(eventlet.listen(('',4567)),app)

