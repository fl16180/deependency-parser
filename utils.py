import numpy as np
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



def unravel_sentence(sentence, head, deprel):

    true_graph = RelationGraph(len(sentence))
    true_graph.set_true_labels(head, deprel)

    stack = TwoStack()
    word_list = [x + 1 for x in range(len(sentence))]
    graph = RelationGraph(len(sentence))

    idx_to_word = word_map(sentence)

    train_states = []
    train_targets = []

    # add root and first token
    stack.add(0)

    while stack:
        state = (copy.deepcopy(stack), copy.copy(word_list), copy.deepcopy(graph))

        print(stack, word_list, graph)

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
