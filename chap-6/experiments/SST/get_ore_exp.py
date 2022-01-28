import sys
import os
import sys
sys.path.append('../../')
sys.path.append('../../utils/')
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
from load_data import load_data, load_embedding_function

from embedding import Embedding
from scipy.spatial import ConvexHull
from operator import itemgetter

sys.path.append('../../abduction_algorithms')
#from abduction_algorithms import knn_smallest_explanation_linf
from abduction_algorithms_HS_cost import knn_smallest_explanation_linf_with_cost, Entails

import itertools

sys.path.append('./../../../Marabou/')
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
from hitman_cost import HitmanCost

import json


if not len(sys.argv) == 9 and not len(sys.argv) == 10:
    print('Usage: ./get_ore_exp.py index model_path X_path.npy X_text.json output_path maxlen emb_dims k [ksize]')
    sys.exit(0)

model_name = sys.argv[2]
index = sys.argv[1]
X_path = sys.argv[3]
X_text_path = sys.argv[4]
out_path = sys.argv[5]
maxlen = int(sys.argv[6])
emb_dims = int(sys.argv[7])
knn = int(sys.argv[8])
window_size = emb_dims
if len(sys.argv) == 10:
    ksize = int(sys.argv[9])
else:
    ksize = None

#X_train, y_train, X_test, y_test, text_train, text_test = load_data('SST',maxlen,emb_dims,'../../training_data/','../../embeddings/',True)
embedding, word2index, index2word, index2embedding = load_embedding_function('SST',maxlen,emb_dims,'../../embeddings/')
shape = (1,maxlen*emb_dims) if ksize == None else (1,ksize,ksize,emb_dims)
embedding = lambda W: np.array([index2embedding[word2index[w]] for w in W]).reshape(*shape)

with open(X_text_path) as json_file:
    data = json.load(json_file)
    text_list = data['text']


input_ = text_list[:maxlen] + ['<PAD>']*(maxlen-len(text_list))
padded_input = input_
X = np.load(X_path).reshape(*shape)


bayes_model = PosteriorModel(model_name)

tmp_weights = bayes_model.sample()

bayes_model.set_weights(tmp_weights)
print(bayes_model.model.summary())
print(X.shape)
y = np.argmax(bayes_model._predict(X))

act_y = np.load(os.path.join(out_path,'y_act_tmp.npy'))

model = bayes_model.model

__Embedding = Embedding(word2index, index2word, index2embedding)
nearest_neighbors, eq_convex_hull, vertices_convex_hull = [], [], []
for i in padded_input:
    tmp = __Embedding.nearest_neighbors(i, knn, method='l2')
    nearest_neighbors += [[index2embedding[word2index[w]] for w in tmp[0]]]
    eq_convex_hull += [[eq.tolist() for eq in ConvexHull(nearest_neighbors[-1]).equations]]
    vertices_convex_hull += [[eq.tolist() for eq in ConvexHull(nearest_neighbors[-1]).vertices]]

for i in range(len(vertices_convex_hull)):
    vertices_convex_hull[i] = list(itemgetter(*vertices_convex_hull[i])(nearest_neighbors[i]))
vertices_convex_hull = np.array(vertices_convex_hull)
minmax_input_bounds = [[np.min(v, axis=0), np.max(v, axis=0)] for v in vertices_convex_hull]


for n in range(maxlen):
    if ksize:
        xx = X.reshape(1,maxlen*emb_dims)
        xx = xx[:,n*window_size:(n+1)*window_size]
    else:
        xx = np.array(X[:,n*window_size:(n+1)*window_size])
    for i, eq in enumerate(eq_convex_hull[n]):
        w,b = np.array(eq[:-1]).reshape(emb_dims,1), eq[-1]
        dp = np.dot(xx,np.array(w)) + b
        assert dp <= 1e-3, print("The convex hull is NOT consistent! Error at equation {} (input {}), result of xW<=b is zero or negative, {}".format(i, n, dp))
print("The convex hull is consistent: each embedded point belongs to one of the respective {} facets equations".format(sum([len(eq) for eq in eq_convex_hull])))


excl_list = ['<PAD>']
uniform = False
word_cost_func = dict()
for word in padded_input:
    if not word in word_cost_func.keys():
        # if its a pad or not in the embedding, give it a high cost
        #if word == '<PAD>' or word2index[word] <= 2 or word == 'arliss' or word == 'howard':
        if word in excl_list:
            print()
            print('*****************')
            print('excluding:',word)
            print('*****************')
            print()
            if uniform:
                word_cost_func[word] = 1
            else:
                word_cost_func[word] = 100
        else:
            word_cost_func[word] = 1
            

input_shape = X.shape
prediction = model.predict(X)
input_ = X.flatten().tolist()
y_hat = np.argmax(prediction)
np.save(os.path.join(out_path,f'ye{index}.json'),y_hat)
c_hat = np.max(prediction)
HS_maxlen = 1e7
verbose = False
print("Classifiation for the input is {} (confidence {})".format(y_hat, c_hat))

frozen_graph_prefix = f'tmp_model_{index}.pb'
filename = frozen_graph_prefix
model_without_softmax = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
tf.saved_model.save(model_without_softmax, frozen_graph_prefix)
output_constraints = [y_hat, (1 if y_hat==0 else 0), 1e-3]


fgsm_args = None
adv_args = None
adv_sims = None

weights_softmax = model.layers[-1].get_weights()
h, exec_time, GAMMA = knn_smallest_explanation_linf_with_cost(model, filename, padded_input, X, word_cost_func, minmax_input_bounds, y_hat, [eq_convex_hull, minmax_input_bounds], output_constraints, window_size, weights_softmax,
                                                    adv_attacks=False, adv_args=adv_args, sims=adv_sims, randomize_pickfalselits=False, HS_maxlen=HS_maxlen, verbose=verbose)

print("Minimum Size Explanation found {} (size {})".format(h, len(h)/window_size))
print("Complementary set of Minimum Size Explanation is {}".format([i for i in range(maxlen*emb_dims) if i not in h]))
print("Execution Time: {}".format(exec_time))



word_ids = [int(i/emb_dims) for idx,i in enumerate(h) if idx % emb_dims == 0 ]
sorted(word_ids)
words_out = [w for idx,w in enumerate(padded_input) if idx in word_ids]
min_expls = [words_out]

verbose = False
min_length = int(len(h)/window_size)
min_cost = sum([word_cost_func[w] for w in words_out])
convex_hull_constraints = [eq_convex_hull, minmax_input_bounds]

count = 0
for comb in itertools.combinations(list(range(maxlen)), min_length):
    count += 1
    h = []
    for c in comb:
        h += list(range(c*emb_dims,(c*emb_dims)+emb_dims))
        
    network = Marabou.read_tf(filename, modelType='savedModel_v2', savedModelTags=['serving_default'])
    res = Entails(h, network, X.flatten().tolist(), convex_hull_constraints, output_constraints, weights_softmax, window_size, 'knn-linf', verbose)
    if len(res) == 0:
        word_ids = [int(i/emb_dims) for idx,i in enumerate(h) if idx % emb_dims == 0 ]
        words_out = [w for idx,w in enumerate(padded_input) if idx in word_ids]
        cost = sum([word_cost_func[w] for w in words_out])
        if cost <= min_cost:
            min_expls.append(words_out)


print(min_expls)
output = dict()
output['expls'] = min_expls
with open(os.path.join(out_path,f'e{index}.json'),'w') as json_out:
    json.dump(output,json_out)
