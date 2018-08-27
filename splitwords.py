#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import logging

from model import ZModel
from config import config

def get_processing_word(vocab_words=None,lowercase=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = "$NUM$"
        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words["$UNK$"]
        # 3.  word id
        return word

    return f

def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    return np.load(filename)["embeddings"]

def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    result = dict()
    with open(filename,encoding="utf-8") as file:
        for idx, word in enumerate(file):
            word = word.strip()
            result[word] = idx
    return result

# directory for training outputs
if not os.path.exists('results/crf/'):
    os.makedirs('results/crf/')

# load vocabs
vocab_words = load_vocab('data/words.txt')
vocab_tags  = load_vocab('data/tags.txt')

# get processing functions
processing_word = get_processing_word(vocab_words, lowercase=True)

# get pre trained embeddings
embeddings = get_trimmed_glove_vectors("data/meta.npz")

# get logger
logger = get_logger("results/crf/log.txt")

# # build model
model = ZModel(config, embeddings, ntags=len(vocab_tags),logger=logger)
model.build()

sess = model.init_tf_sess()