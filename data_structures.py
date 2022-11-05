import numpy as np


class TwoStack:
    def __init__(self):
        self.stack = ['root']
        
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


class RelationGraph:
    def __init__(self, sentence, head, deprel):
        self.sentence = sentence
        self.head = head
        self.deprel = deprel
        
        # lookup for deprel entry to matrix value
        self.dep_to_idx = {
            dep: i for i, dep in enumerate(set(deprel))
        }
        # lookup for matrix value to deprel type
        self.idx_to_dep = {
            v: k for k, v in self.dep_to_idx.items()
        }

        n = len(sentence)
        self.mat = np.zeros((n + 1, n + 1))        # 0th entry is root
        for child, parent in enumerate(head):
            self.mat[parent, child + 1] = self.dep_to_idx[deprel[child]]     
    
    def contains(self, from_idx, to_idx):
        return self.mat[from_idx, to_idx] > 0
    
    def get_children(self, idx):
        return list(np.where(self.mat[idx] > 0)[0])
    
    def get_parent(self, idx):
        return np.where(self.mat[:, idx] > 0)[0][0]
    
    def get_word(self, idx):
        return self.sentence[idx]
    


# stack = TwoStack()
# word_list = df.iloc[0]['sentence']
# graph = RelationGraph(df.iloc[0]['sentence'], df.iloc[0]['head'], df.iloc[0]['deprel'])