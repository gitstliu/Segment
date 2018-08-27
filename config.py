#!/usr/bin/env python
# -*- coding: utf-8 -*-

class config():
    train_embeddings = False
    hidden_size = 300
    crf = True # if crf, training is 1.7x slower
    model_output = "results/crf/model.weights/"