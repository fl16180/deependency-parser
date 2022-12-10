from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset

from utils import read_syntax_tree, unravel_sentence

DATA_DIR = Path('./data')


def read_ud_file(fname='en_ewt-ud-train.conllu'):
    with open(fname) as f:
        lines = f.readlines()
    df = read_syntax_tree(lines)
    return df


class UDDataset(Dataset):

    def __init__(self, language='en_ewt', split='train', vocab=None):
        self.split = split
        header = language

        print('reading data')
        if self.split == 'train':
            self.df = read_ud_file(DATA_DIR / f'{header}-ud-train.conllu')
        elif self.split == 'dev':
            self.df = read_ud_file(DATA_DIR / f'{header}-ud-dev.conllu')
        elif self.split == 'test':
            self.df = read_ud_file(DATA_DIR / f'{header}-ud-test.conllu')

        print('unraveling data')

        n_failed_unravel = 0
        self.states = []
        self.contexts = []
        self.targets = []
        for row in self.df.iterrows():
            try:
                new_states, new_targets = unravel_sentence(
                    row[1]['sentence'],
                    row[1]['head'],
                    row[1]['deprel']
                )
                n_copy = len(new_targets)
                new_context = [
                    (row[1]['sentence'], row[1]['pos'],
                     row[1]['head'], row[1]['deprel'])
                    for _ in range(n_copy)
                ]
                self.contexts.extend(new_context)
                self.states.extend(new_states)
                self.targets.extend(new_targets)
            except Exception as e:
                n_failed_unravel += 1
        print('failed unravels: ', n_failed_unravel)

    def __getitem__(self, idx):
        return self.states[idx], self.contexts[idx], self.targets[idx]

    def __len__(self):
        return len(self.states)


if __name__ == '__main__':
    dataset = UDDataset(language='en', split='train')
    print(dataset[0], dataset[1])
    print(dataset[100])
    print(len(dataset))
