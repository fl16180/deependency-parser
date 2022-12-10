import numpy as np
import pandas as pd
import copy

from data_structures import *


def word_map(sentence):
    idx_to_word = {0: 'root'}
    for i, word in enumerate(sentence):
        idx_to_word[i + 1] = word
    return idx_to_word


def view_states(sentence, states, targets):
    idx_to_word = word_map(sentence)
    for state, target in zip(states, targets):
        stack, buffer, graph = state
        stack_str = [idx_to_word[x] for x in stack.stack]
        buffer_str = [idx_to_word[x] for x in buffer]
        print(stack_str, buffer_str, target)
    return


def read_syntax_tree(lines):
    sentences = []
    poses = []
    heads = []
    deprels = []
    
    count_word = False
    for l in lines:
        if l[:6] == '# text':
            count_word = True
            sentence = []
            pos = []
            head = []
            deprel = []
        elif count_word:
            if l[0] != '\n':
                cells = l.split('\t')
                
                # skip inferred words not part of the original sentence
                if '.1' in cells[0]:
                    continue
                if '-' in cells[0]:
                    continue

                try:
                    int(cells[6])
                except:
                    from pdb import set_trace; set_trace()
                    
                sentence.append(cells[1])
                pos.append(cells[4])
                head.append(int(cells[6]))
                deprel.append(cells[7])
            else:
                count_word = False
                sentences.append(sentence)
                poses.append(pos)
                heads.append(head)
                deprels.append(deprel)
        else:
            continue
    return pd.DataFrame(
        {'sentence': sentences, 'pos': poses, 'head': heads, 'deprel': deprels}
    )


def unravel_sentence(sentence, head, deprel):
    ''' Key function for unraveling a labeled sentence into transitions
    '''
    true_graph = RelationGraph(len(sentence))
    true_graph.set_true_labels(head, deprel)

    stack = TwoStack()
    word_list = Buffer(len(sentence))
    graph = RelationGraph(len(sentence))

    train_states = []
    train_targets = []

    # add root and first token
    stack.add(0)

    while stack:
        state = (copy.deepcopy(stack), copy.deepcopy(word_list), copy.deepcopy(graph))

        if len(stack) == 1:
            if len(word_list) > 0:
                stack.add(word_list.pop(0))
                target = 'shift'
            else:
                stack.pop1()
                target = 'done'
        else:
            # check for left-arc
            if true_graph.contains(stack.view1(), stack.view2()):
                graph.add_relation(stack.view1(), stack.view2())
                stack.pop2()
                target = 'left-arc'

            elif true_graph.contains(stack.view2(), stack.view1()) and \
                    all([graph.contains(stack.view1(), child) for child in true_graph.get_children(stack.view1())]):
                graph.add_relation(stack.view2(), stack.view1())
                stack.pop1()
                target = 'right-arc'

            else:
                stack.add(word_list.pop(0))
                target = 'shift'

        train_states.append(state)
        train_targets.append(target)

    return train_states, train_targets
