import sys
import os
import sys
sys.path.append('../..')
sys.path.append('../shap')
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

import shap


if not len(sys.argv) == 6:
    print('Usage: ./get_mnist_exp.py index model_path input_path output_path background_path')
    sys.exit(0)

model_name = sys.argv[2]
bayes_model = PosteriorModel(model_name)

X = np.load(sys.argv[3])

bayes_model.set_weights(bayes_model.sample())
model = bayes_model.model
y = np.argmax(np.array(bayes_model._predict(X)).flatten())

background = np.load(sys.argv[5])

e = shap.DeepExplainer(model,background)
shap_values = e.shap_values(X)

np.save(f'{sys.argv[4]}/e{sys.argv[1]}_{y}.npy',shap_values,False)
