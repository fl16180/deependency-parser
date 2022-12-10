from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_structures import Vocab

from dataset import read_ud_file, UDDataset


VERSION = 'FULL'


def get_word(idx, sentence, vocab):
    if idx is None or idx == 0:
        word = '<PAD>'
    else:
        word = sentence[idx - 1]
    word_idx = vocab.vocab.get(word, vocab.vocab['<UNK>'])
    return word_idx


def get_pos(idx, pos_seq, vocab):
    if idx is None or idx == 0:
        pos = '<PAD>'
    else:
        pos = pos_seq[idx - 1]
    pos_idx = vocab.vocab.get(pos, vocab.vocab['<UNK>'])
    return pos_idx


def base_featurize(data, word_vocab, **kwargs):
    ''' baseline features (3 words) '''
    states, context, labels = zip(*data)
    
    labels = np.array(labels)
    target = torch.zeros(len(labels))
    target[labels == 'left-arc'] = 1
    target[labels == 'right-arc'] = 2

    features = torch.zeros((len(data), 3))

    # for each state config
    for i in range(len(data)):

        # get top words of stack and buffer
        idx1 = states[i][0].view1()
        idx2 = states[i][0].view2()
        idx3 = states[i][1].get(0)

        # convert word index to embedding word index
        word1 = get_word(idx1, context[i][0], word_vocab)
        word2 = get_word(idx2, context[i][0], word_vocab)
        word3 = get_word(idx3, context[i][0], word_vocab)

        features[i, :] = torch.tensor([word1, word2, word3])

    return features.long(), target.long()


def part_featurize(data, word_vocab, pos_vocab):
    ''' more words features '''
    states, context, labels = zip(*data)
    
    labels = np.array(labels)
    target = torch.zeros(len(labels))
    target[labels == 'left-arc'] = 1
    target[labels == 'right-arc'] = 2

    features = torch.zeros((len(data), 6))
    for i in range(len(data)):

        idx1 = states[i][0].view1()
        idx2 = states[i][0].view2()
        idx3 = states[i][0].view3()
        idx4 = states[i][1].get(0)
        idx5 = states[i][1].get(1)
        idx6 = states[i][1].get(2)

        word1 = get_word(idx1, context[i][0], word_vocab)
        word2 = get_word(idx2, context[i][0], word_vocab)
        word3 = get_word(idx3, context[i][0], word_vocab)
        word4 = get_word(idx4, context[i][0], word_vocab)
        word5 = get_word(idx5, context[i][0], word_vocab)
        word6 = get_word(idx6, context[i][0], word_vocab)

        features[i, :] = torch.tensor([
            word1, word2, word3, word4, word5, word6]
        )
    return features.long(), target.long()


def full_featurize(data, word_vocab, pos_vocab):
    ''' more words + POS features '''
    states, context, labels = zip(*data)
    
    labels = np.array(labels)
    target = torch.zeros(len(labels))
    target[labels == 'left-arc'] = 1
    target[labels == 'right-arc'] = 2

    features = torch.zeros((len(data), 12))
    for i in range(len(data)):

        idx1 = states[i][0].view1()
        idx2 = states[i][0].view2()
        idx3 = states[i][0].view3()
        idx4 = states[i][1].get(0)
        idx5 = states[i][1].get(1)
        idx6 = states[i][1].get(2)

        word1 = get_word(idx1, context[i][0], word_vocab)
        word2 = get_word(idx2, context[i][0], word_vocab)
        word3 = get_word(idx3, context[i][0], word_vocab)
        word4 = get_word(idx4, context[i][0], word_vocab)
        word5 = get_word(idx5, context[i][0], word_vocab)
        word6 = get_word(idx6, context[i][0], word_vocab)

        pos1 = get_pos(idx1, context[i][1], pos_vocab)
        pos2 = get_pos(idx2, context[i][1], pos_vocab)
        pos3 = get_pos(idx3, context[i][1], pos_vocab)
        pos4 = get_pos(idx4, context[i][1], pos_vocab)
        pos5 = get_pos(idx5, context[i][1], pos_vocab)
        pos6 = get_pos(idx6, context[i][1], pos_vocab)

        features[i, :] = torch.tensor([
            word1, word2, word3, word4, word5, word6,
            pos1, pos2, pos3, pos4, pos5, pos6]
        )
    return features.long(), target.long()


def train(model, train_loader, optimizer, **kwargs):
    ''' train model '''
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for data, target in train_loader:
        
        out = model(data)
        loss = loss_fn(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_model(model, data_loader, **kwargs):
    ''' Compute transition accuracy on data_loader'''
    model.eval()
    with torch.no_grad():
        preds = []
        truth = []
        for data, target in data_loader:
            out = model(data)
            out = torch.softmax(out, 1).argmax(1)
            preds.append(out.detach())
            truth.append(target)

        preds = torch.cat(preds)
        truth = torch.cat(truth)

    return torch.sum(preds == truth) / len(preds)


def eval_graph(model, word_vocab, pos_vocab, eval_set, language):
    ''' Predicts full dependency parse for each sentence in real time 
    
    Follows the unraveling procedure, but using the NN oracle for each transition
    prediction.
    '''
    model.eval()

    from dataset import DATA_DIR, read_ud_file
    from data_structures import TwoStack, Buffer, RelationGraph

    df = read_ud_file(DATA_DIR / f'{language}-ud-{eval_set}.conllu')
    
    graph_accs = []
    for row in df.iterrows():
        sentence = row[1]['sentence']
        pos_seq = row[1]['pos']

        stack = TwoStack()
        buffer = Buffer(len(sentence))
        graph = RelationGraph(len(sentence))

        stack.add(0)

        while stack:
            # only one item on stack (root)
            if len(stack) == 1:
                if len(buffer) > 0:
                    stack.add(buffer.pop(0))
                else:
                    stack.pop1()
            # two items on stack at end, go ahead and join root
            elif len(stack) == 2 and len(buffer) == 0:
                graph.add_relation(stack.view2(), stack.view1())
                stack.pop2()
            else:
                # compile features
                idx1 = stack.view1()
                idx2 = stack.view2()
                idx3 = stack.view3()
                idx4 = buffer.get(0)
                idx5 = buffer.get(1)
                idx6 = buffer.get(2)

                word1 = get_word(idx1, sentence, word_vocab)
                word2 = get_word(idx2, sentence, word_vocab)
                word3 = get_word(idx3, sentence, word_vocab)
                word4 = get_word(idx4, sentence, word_vocab)
                word5 = get_word(idx5, sentence, word_vocab)
                word6 = get_word(idx6, sentence, word_vocab)

                pos1 = get_pos(idx1, pos_seq, pos_vocab)
                pos2 = get_pos(idx2, pos_seq, pos_vocab)
                pos3 = get_pos(idx3, pos_seq, pos_vocab)
                pos4 = get_pos(idx4, pos_seq, pos_vocab)
                pos5 = get_pos(idx5, pos_seq, pos_vocab)
                pos6 = get_pos(idx6, pos_seq, pos_vocab)

                if VERSION == 'FULL':
                    features = torch.tensor([
                        word1, word2, word3, word4, word5, word6,
                        pos1, pos2, pos3, pos4, pos5, pos6]
                    ).view(1, 12)
                elif VERSION == 'PART':
                    features = torch.tensor([
                        word1, word2, word3, word4, word5, word6]
                    ).view(1, 6)
                else:
                    features = torch.tensor([
                        word1, word2, word4]
                    ).view(1, 3)

                out = torch.softmax(model(features), 1).flatten()
                top1 = out.topk(2)[1][0].item()
                top2 = out.topk(2)[1][1].item()

                # can't shift, require left or right arc
                if len(buffer) == 0 and top1 == 0:
                    action = top2
                # only two left on stack, so can't left-arc until buffer empty
                elif len(stack) == 2 and len(buffer) != 0 and top1 == 1:
                    action = top2
                else:
                    action = top1

                if action == 0:
                    stack.add(buffer.pop(0))
                elif action == 1:
                    graph.add_relation(stack.view1(), stack.view2())
                    stack.pop2()
                else:
                    graph.add_relation(stack.view2(), stack.view1())
                    stack.pop1()

        mat = torch.tensor(graph.mat)
        preds = torch.zeros(len(row[1]['head']))
        for ix in range(len(preds)):
            nonzero = torch.where(mat[:, ix+1] != 0)[0]
            if len(nonzero) == 0:
                preds[ix] = -1
            else:
                preds[ix] = nonzero.item()

        acc = (torch.sum(preds == torch.tensor(row[1]['head'])) / len(preds)).item()
        graph_accs.append(acc)
    return np.mean(graph_accs)



class BaseModel(nn.Module):
    def __init__(self, vocab, embed_dim=50, n_class=3):
        super().__init__()

        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.fnn = nn.Sequential(
            nn.Linear(3*embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, n_class)
        )

    def forward(self, X):
        X = self.embed(X)
        X = X.view(X.shape[0], -1)
        X = self.fnn(X)
        return X


class PartModel(nn.Module):
    def __init__(self, vocab, pos_vocab, embed_dim=50, n_class=3):
        super().__init__()

        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.embed = init_glove(self.embed, vocab)

        self.fnn = nn.Sequential(
            nn.Linear(6*embed_dim, 2 * embed_dim),
            nn.Tanh(),
            nn.Linear(2 * embed_dim, 2 * embed_dim)
        )

    def forward(self, X):
        X_word = X
        X_word = self.embed(X_word)
        X_word = X_word.view(X_word.shape[0], -1)

        X = self.fnn(X_word)
        return X


class FullModel(nn.Module):
    def __init__(self, vocab, pos_vocab, embed_dim=50, n_class=3):
        super().__init__()

        self.embed = nn.Embedding(len(vocab), embed_dim)
        self.embed = init_glove(self.embed, vocab)
        self.pos_embed = nn.Embedding(len(pos_vocab), embed_dim)

        self.fnn = nn.Sequential(
            nn.Linear(12*embed_dim, 2 * embed_dim),
            nn.Tanh(),
            nn.Linear(2 * embed_dim, 2 * embed_dim)
        )

    def forward(self, X):
        X_word = X[:, :6]
        X_word = self.embed(X_word)
        X_word = X_word.view(X_word.shape[0], -1)

        X_pos = X[:, 6:]
        X_pos = self.pos_embed(X_pos)
        X_pos = X_pos.view(X_pos.shape[0], -1)

        X = torch.cat([X_word, X_pos], 1)
        X = self.fnn(X)
        return X


def init_glove(emb_layer, vocab):
    from load_glove import load_glove_dict
    word2emb = load_glove_dict('/Users/fredlu/Documents/hw/glove.6B')

    for word, idx in vocab.vocab.items():
        if word in word2emb:
            vec = word2emb[word]
            emb_layer.weight.data[idx] = torch.tensor(vec)
    return emb_layer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_word_freq', type=int, default=20)
    parser.add_argument('--eval', default='dev', choices=['dev', 'test'])
    parser.add_argument('--lang', default='en_ewt')
    args = parser.parse_args()

    # generate vocab from train set
    DATA_DIR = Path('./data')
    df = read_ud_file(DATA_DIR / f'{args.lang}-ud-train.conllu')

    word_vocab = Vocab(min_freq=args.min_word_freq).build(df['sentence'].tolist())
    pos_vocab = Vocab(min_freq=1).build(df['pos'].tolist())
    deprel_vocab = Vocab(min_freq=1).build(df['deprel'].tolist())


    train_ds = UDDataset(language=args.lang, split='train')
    test_ds = UDDataset(language=args.lang, split=args.eval)

    if VERSION == 'BASE':
        model = BaseModel(word_vocab, 50, 3)
        featurizer = base_featurize
    elif VERSION == 'PART':
        model = PartModel(word_vocab, pos_vocab, 50, 3)
        featurizer = part_featurize
    else:
        model = FullModel(word_vocab, pos_vocab, 50, 3)
        featurizer = full_featurize

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=256,
        collate_fn=partial(featurizer, word_vocab=word_vocab, pos_vocab=pos_vocab))
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256,
        collate_fn=partial(featurizer, word_vocab=word_vocab, pos_vocab=pos_vocab))


    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        train(model, train_loader, opt,
            word_vocab=word_vocab,
            pos_vocab=pos_vocab,
            deprel_vocab=deprel_vocab)

        acc = eval_model(model, test_loader)
        print('transition acc: ', acc)

        parse_acc = eval_graph(model, word_vocab, pos_vocab, args.eval, args.lang)
        print('parse acc: ', parse_acc)
