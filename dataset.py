import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset

from data_structures import TwoStack, RelationGraph


def read_syntax_tree(lines):
    sentences = []
    heads = []
    deprels = []
    
    count_word = False
    for l in lines:
        if l[:6] == '# text':
            count_word = True
            sentence = []
            head = []
            deprel = []
        elif count_word:
            if l[0] != '\n':
                cells = l.split('\t')
                
                # skip inferred words not part of the original sentence
                if '.1' in str(cells[0]):
                    continue
                    
                sentence.append(cells[1])
                head.append(int(cells[6]))
                deprel.append(cells[7])
            else:
                count_word = False
                sentences.append(sentence)
                heads.append(head)
                deprels.append(deprel)
        else:
            continue
    return pd.DataFrame(
        {'sentence': sentences, 'head': heads, 'deprel': deprels}
    )


def read_ud_file(fname='en_ewt-ud-train.conllu'):
    with open(fname) as f:
        lines = f.readlines()
    df = read_syntax_tree(lines)
    return df


class UDDataset(Dataset):

    def __init__(self, language, split='train'):
        self.language = language
        self.split = split

        if self.split == 'train':
            self.df = read_ud_file('en_ewt-ud-train.conllu')
        elif self.split == 'dev':
            self.df = read_ud_file('en_ewt-ud-dev.conllu')
        elif self.split == 'test':
            print('not implemented yet')

        stack = TwoStack()
        word_list = self.df.iloc[0]['sentence']
        graph = RelationGraph(self.df.iloc[0]['sentence'], self.df.iloc[0]['head'], self.df.iloc[0]['deprel'])

        self.states = []
        self.targets = []

    def __getitem__(self, idx):
        return self.states[idx], self.targets[idx]


    def __len__(self):
        return len(self.states)
