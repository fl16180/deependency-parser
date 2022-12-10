import numpy as np
from collections import Counter


class Vocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        
    def build(self, sequences):
        # build vocab from sequence
        vocab = Counter([x for inst in sequences for x in inst])
        vocab = {v: i for i, v in enumerate(
            sorted([word for (word, val) in vocab.items() if val >= self.min_freq]))
        }
        self.vocab = vocab
        n = len(vocab.keys())
        # add unknown and padding tokens in
        self.vocab['<UNK>'] = n
        self.vocab['<PAD>'] = n + 1
        return self
        
    def transform(self, sequence):
        # use unknown token if word not in vocab
        mapping = [self.vocab.get(x, self.vocab['<UNK>']) for x in sequence]
        return mapping

    def __len__(self):
        return len(self.vocab)


class TwoStack:
    def __init__(self):
        self.stack = []
        
    def add(self, x):
        self.stack.append(x)
        
    def view1(self):
        if len(self.stack) == 0:
            return None
        return self.stack[-1]
    
    def view2(self):
        if len(self.stack) <= 1:
            return None
        return self.stack[-2]

    def view3(self):
        if len(self.stack) <= 2:
            return None
        return self.stack[-3]
    
    def pop1(self):
        if len(self.stack) == 0:
            return None
        return self.stack.pop(-1)
        
    def pop2(self):
        if len(self.stack) <= 1:
            return None
        return self.stack.pop(-2)

    def __bool__(self):
        return len(self.stack) > 0

    def __len__(self):
        return len(self.stack)
    
    def __str__(self):
        return self.stack.__str__()


class Buffer:
    def __init__(self, n):
        self.array = [x + 1 for x in range(n)]

    def get(self, i):
        if len(self.array) < i + 1:
            return None
        return self.array[i]

    def pop(self, i):
        return self.array.pop(i)

    def __len__(self):
        return len(self.array)


class RelationGraph:
    def __init__(self, n):
        self.mat = np.zeros((n + 1, n + 1))
        self.dep_to_idx = {}
        self.idx_to_dep = {}
        self.true_graph = False
        
    def set_true_labels(self, head, deprel):
        # lookup for deprel entry to matrix value
        self.dep_to_idx = {
            dep: i for i, dep in enumerate(set(deprel))
        }
        # lookup for matrix value to deprel type
        self.idx_to_dep = {
            v: k for k, v in self.dep_to_idx.items()
        }

        for child, parent in enumerate(head):
            # self.mat[parent, child + 1] = self.dep_to_idx[deprel[child]]
            self.mat[parent, child + 1] = 1
            
        self.true_graph = True
    
    def add_relation(self, from_idx, to_idx, deprel=1):
        if self.true_graph:
            raise Exception('Cannot modify true labeled graph')
        self.mat[from_idx, to_idx] = deprel
    
    def contains(self, from_idx, to_idx):
        return self.mat[from_idx, to_idx] > 0
    
    def get_children(self, idx):
        return list(np.where(self.mat[idx] > 0)[0])
    
    def get_parent(self, idx):
        return np.where(self.mat[:, idx] > 0)[0][0]
    
    def __str__(self):
        out = '{'
        for a, b in zip(*np.where(self.mat > 0)):
            out += f'{a}->{b},'
        out += '}'
        return out
