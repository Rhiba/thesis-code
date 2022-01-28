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
sys.path.append('LayerwiseRelevancePropagation/src')
from lrp import RelevancePropagation

if not len(sys.argv) == 6:
    print('Usage: ./get_exp.py index model_path input_path output_path grayscale')
    sys.exit(0)

model_name = sys.argv[2]
bayes_model = PosteriorModel(model_name)

X = np.load(sys.argv[3])

bayes_model.set_weights(bayes_model.sample())
y = np.argmax(np.array(bayes_model._predict(X)).flatten())

lrp = RelevancePropagation(0.01,'max',bool(sys.argv[5]),X.shape,bayes_model.model)
exp = lrp.run(X.reshape(*X.shape[1:]))

np.save(f'{sys.argv[4]}/e{sys.argv[1]}_{y}.npy',exp,False)
