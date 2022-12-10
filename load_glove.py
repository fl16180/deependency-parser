'''
Code referenced from
        https://discuss.pytorch.org/t/how-to-download-and-use-glove-vectors/54847/2
        
There isn't an 'official' tutorial on using GloVe embeddings in Pytorch 
so this was the closest resource I could find.

I further modified the code to use numpy binaries instead of bcolz
which does not install on my system, and numpy is perfectly efficient anyway.
'''
import os
import pickle
import numpy as np
from collections import defaultdict


def preprocess_glove(glove_path):
    ''' This parses the GloVe text file which contains each word and the embedding,
        into a far more efficient compressed array with the vectors, and a
        separate dictionary-to-index lookup.
    '''
    words = []
    word2idx = {}
    vectors = np.zeros([400000, 50])

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for idx, l in enumerate(f):
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx

            vect = np.array(line[1:]).astype(float)
            vectors[idx, :] = vect
        
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))
    np.save(f'{glove_path}/6B.50.npy', vectors, allow_pickle=True)

    
def load_glove_dict(glove_path):
    ''' This loads the compressed vectors and dictionary for usage
    '''
    vectors = np.load(f'{glove_path}/6B.50.npy', allow_pickle=True)
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove

    
if __name__ == '__main__':

    preprocess_glove('./glove.6B')
