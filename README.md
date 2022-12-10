Here is the code implementing neural dependency parsing.


Experiment.py is the main file for training. See model and featurizing code within.

load_glove.py is for loading pretrained embeddings

data_structures.py implements the Stack, Buffer, Relational Graph, and Vocab

utils.py contains important functions for generating the transition dataset (read_syntax_tree, unravel_sentence)
