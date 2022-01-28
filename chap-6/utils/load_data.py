from pandas import read_csv
import string
import numpy as np
from glove_utils import load_embedding, pad_sequences

def load_data(dataset='sst',maxlen=10,emb_dims=5,data_folder='../../training_data/',embedding_folder='../../embeddings/',with_text=False):
    dataset = dataset.lower()
    if dataset == 'sst':
        X_train = read_csv(data_folder+'SST_2/training/SST_2__FULL.csv', sep=',',header=None).values
        X_test = read_csv(data_folder+'SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
        y_train, y_test = [], []
        for i in range(len(X_train)):
            r, s = X_train[i]  # review, score (comma separated in the original file)
            X_train[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
            y_train.append((0 if s.strip()=='negative' else 1))
        for i in range(len(X_test)):
            r, s = X_test[i]
            X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
            y_test.append((0 if s.strip()=='negative' else 1))
        X_train, X_test = X_train[:,0], X_test[:,0]
        n = -1  # you may want to take just some samples (-1 to take them all)
        X_train = X_train[:n]
        X_test = X_test[:n]
        y_train = y_train[:n]
        y_test = y_test[:n]

        if with_text:
            return_text_train = X_train.copy()
            return_text_test = X_test.copy()
        # Select the embedding
        EMBEDDING_FILENAME = embedding_folder+'custom-embedding-SST.{}d.txt'.format(emb_dims)
        word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)

        X_train = [[index2embedding[word2index[x]] for x in xx] for xx in X_train]        
        X_train = np.asarray(pad_sequences(X_train, maxlen=maxlen, emb_size=emb_dims))
        X_train = X_train.astype("float32").reshape(-1,maxlen*emb_dims)


        X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
        X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
        X_test = X_test.astype("float32").reshape(-1,maxlen*emb_dims)

        if with_text:
            return X_train,y_train,X_test,y_test,return_text_train,return_text_test
        return X_train,y_train,X_test,y_test

    elif dataset == 'imbd':
        print('Not implemented.') 
        return None
    elif dataset == 'twitter':
        print('Not implemented.') 
        return None
    else:
        print('Supported datasets: SST, IMDB, Twitter.')
        return None


def load_embedding_function(dataset='sst',maxlen=10,emb_dims=5,embedding_folder='../../embeddings/'):
    dataset = dataset.lower()
    if dataset == 'sst':
        filename = embedding_folder+f'custom-embedding-SST.{emb_dims}d.txt'
        word2index, index2word, index2embedding = load_embedding(filename)
        embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(1,maxlen*emb_dims)
        return embedding, word2index, index2word, index2embedding
    elif dataset == 'imbd':
        print('Not implemented.') 
        return None
    elif dataset == 'twitter':
        print('Not implemented.') 
        return None
    else:
        print('Supported datasets: SST, IMDB, Twitter.')
        return None
