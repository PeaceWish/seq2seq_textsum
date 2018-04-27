#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/27 12:09
# @Author  : Zhangyuheng
# @File    : seq2seqModel.py

import tensorflow as tf


class Seq2seqModel:
    def __init__(self,
                 rnn_size,                  # rnn的单元数
                 num_layers,                # rnn的层数
                 embedding_size,            # embedding的维度
                 learning_rate,             # 学习率
                 word_to_id,                # 单词（汉字分词）转换为id
                 mode,                      # 模式
                 use_attention,             # 是否使用attention
                 beam_search,               # 是否使用beam search
                 beam_size,                 # beam 的大小
                 max_gradient_norm=5.0):
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.word_to_id = word_to_id
        self.vocab_size = len(self.word_to_id)
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm
        print("===========开始构建model===========")
        # 开始设置placeholder
        # 考虑到处理效率，每次处理一批句子，其大小为batch_size
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        # encoder输入 [batch_size, encoder_max_size] 对于不足max_size的使用0 padding
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        # encoder输入长度 列表表示每一个输入的实际长度
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        # dropout 百分比
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
        # decoder输出 [batch_size, decoder_max_size]
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        # decoder输入长度 列表表示每一个输入的实际长度
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        print("===========1 input_embedding===========")
        with tf.variable_scope('input_embedding'):
            # embedding [vocab_size, embedding_size]
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            # encoder_inputs_embedded [batch_size, encoder_max_size, embedding_size]
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)

        print("===========2 encoder===================")
        with tf.variable_scope('encoder'):
            # 创建LSTMCell两层+dropout
            encoder_cell = self._create_lstm_cell()
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                               encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)

        print("===========3 decoder====================")
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size,
                                                                memory=encoder_outputs,
                                                                memory_sequence_length=self.encoder_inputs_length)
        # 问题：为什么attention在decoder上，要传encoder_inputs_length

        decoder_cell = self._create_rnn_cell()
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                           attention_layer_size=self.rnn_size, name='Attention_Wrapper')
        decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(
            cell_state=encoder_state)
        output_layer = tf.layers.Dense(self.vocab_size,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        # 训练模式 需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        if self.mode == 'train':

            pass
        # 评价模式 不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        elif self.mode == 'eval':
            pass
        # 预测模式 只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        elif self.mode == 'predict':
            pass

    def _create_lstm_cell(self):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            return tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
        return tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
