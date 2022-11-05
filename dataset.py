from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset

from utils import unravel_sentence

DATA_DIR = Path('/Users/fredlu/Documents/hw')


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
                if '.1' in cells[0]:
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

    def __init__(self, language='en', split='train'):
        if language != 'en':
            raise NotImplementedError
        self.language = language
        self.split = split

        print('reading data')
        if self.split == 'train':
            self.df = read_ud_file(DATA_DIR / f'{language}_ewt-ud-train.conllu')
        elif self.split == 'dev':
            self.df = read_ud_file(DATA_DIR / f'{language}_ewt-ud-dev.conllu')
        elif self.split == 'test':
            raise NotImplementedError

        print('unraveling data')

        n_failed_unravel = 0
        self.states = []
        self.targets = []
        for row in self.df.iterrows():
            # print(row)
            try:
                new_states, new_targets = unravel_sentence(
                    row[1]['sentence'],
                    row[1]['head'],
                    row[1]['deprel']
                )
                self.states.extend(new_states)
                self.targets.extend(new_targets)
            except Exception as e:
                n_failed_unravel += 1
                # print(e, 'on ', row[0])
        print('failed unravels: ', n_failed_unravel)

    def __getitem__(self, idx):
        return self.states[idx], self.targets[idx]

    def __len__(self):
        return len(self.states)


if __name__ == '__main__':
    dataset = UDDataset(language='en', split='train')
    print(dataset[0], dataset[1])
    print(dataset[100])
    print(len(dataset))
