import sys
import os
import sys
sys.path.append('../..')
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

if not len(sys.argv) == 5:
    print('Usage: ./get_exp.py index model_path input_path output_path')
    sys.exit(0)

model_name = sys.argv[2]
bayes_model = PosteriorModel(model_name)

X = np.load(sys.argv[3])

bayes_model.set_weights(bayes_model.sample())
exp = eps_LRP(bayes_model,X)
y = np.argmax(bayes_model._predict(np.asarray([X])))

np.save(f'{sys.argv[4]}/e{sys.argv[1]}.npy',exp,False)
np.save(f'{sys.argv[4]}/ye{sys.argv[1]}.npy',y,False)
