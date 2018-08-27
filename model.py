#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

def pad_sequences(sequences, pad_tok):
    max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]
    return sequence_padded, sequence_length

class ZModel(object):
    def __init__(self, config, embeddings, ntags, logger):
        """
        :param config: 高参
        :param embeddings: embedding层
        :param ntags: tag的数量 e.g. B-ORG, I-PER....
        :param logger: logger instance
        """
        self.config = config
        self.embeddings = embeddings
        self.ntags = ntags

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")  # batch size, max length of sentence in batch
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")  # shape = batch size
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        word_ids, sequence_lengths = pad_sequences(words,0)
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }
        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels
        if lr is not None:
            feed[self.lr] = lr
        if dropout is not None:
            feed[self.dropout] = dropout
        return feed, sequence_lengths

    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32, trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("bi-lstm"):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, self.word_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2 * self.config.hidden_size, self.ntags], dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())
            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])

    def add_pred_op(self):
        """
        Adds labels_pred to self
        """
        if not self.config.crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

    def add_loss_op(self):
        """
        Adds loss to self
        """
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

    def predict_batch(self, sess, words):
        """
        Args:
            sess: a tensorflow session
            words: list of sentences
        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params],feed_dict=fd)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(logit, transition_params)
                viterbi_sequences += [viterbi_sequence]
            return viterbi_sequences, sequence_lengths
        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)
            return labels_pred, sequence_lengths


    def init_tf_sess(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, self.config.model_output)
        return sess

    def interactive_shell(self, processing_word, sentence,sess):
        starttime = time.time()
        try:
                words_raw = list(sentence)
                wordsmeta = map(processing_word, words_raw)
                words = list(wordsmeta)
                pred_ids, _ = self.predict_batch(sess, [words])

                taglist = list(pred_ids[0])
                length = len(taglist)
                head = 0
                results = list()
                for i in range(length):
                    if taglist[i] == 0 or taglist[i] == 2:
                        results.append(sentence[head:i + 1])
                        head = i + 1

                return results
        except EOFError:
                print("Closing session.")
                sess.close()
        endtime = time.time()
        print("totaltime", endtime - starttime)