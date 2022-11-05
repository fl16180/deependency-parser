import numpy as np
import pandas as pd


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


def load_ud_file(fname='en_ewt-ud-train.conllu'):
    with open(fname) as f:
        lines = f.readlines()
    df = read_syntax_tree(lines)
    return df
