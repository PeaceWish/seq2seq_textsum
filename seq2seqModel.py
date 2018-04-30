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

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32,
                                     name='masks')
        self.saver = tf.train.Saver(tf.global_variables()) # 保存所有的全局变量
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
        # 我们通过source_sequence_length 保证注意机制的权重有适当的规范化（只在 non-padding的位置）

        decoder_cell = self._create_rnn_cell()
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                           attention_layer_size=self.rnn_size, name='Attention_Wrapper')
        decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(
            cell_state=encoder_state)
        output_layer = tf.layers.Dense(self.vocab_size,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        # 训练模式 需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        if self.mode == 'train':
            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<go>']), ending], 1)
            decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
            # 训练阶段，使用TrainingHelper+BasicDecoder的组合，这一般是固定的，当然也可以自己定义Helper类，实现自己的功能
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                sequence_length=self.decoder_targets_length,
                                                                time_major=False, name='training_helper')
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                               initial_state=decoder_initial_state,
                                                               output_layer=output_layer)
            # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
            # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=self.max_target_sequence_length)
            # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            # self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
            # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志

            # logits：尺寸[batch_size, decoder_targets_length, vocab_size]
            # targets：尺寸[batch_size, decoder_targets_length]，不用做one_hot。
            # weights：[batch_size, decoder_targets_length]，即mask，滤去padding的loss计算，使loss计算更准确。
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                         targets=self.decoder_targets,
                                                         weights=self.mask)
            # Training summary for the current batch_loss
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()
            optimizer = tf.train.AdamOptimizer(self.learing_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
        # 预测模式 只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        elif self.mode == 'predict':
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<go>']
            end_token = self.word_to_idx['<eos>']
            # decoder阶段根据是否使用beam_search决定不同的组合，
            # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
            # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
            if self.beam_search:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                         start_tokens=start_tokens, end_token=end_token,
                                                                         initial_state=decoder_initial_state,
                                                                         beam_width=self.beam_size,
                                                                         output_layer=output_layer)
            else:
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                           start_tokens=start_tokens,
                                                                           end_token=end_token)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                    initial_state=decoder_initial_state,
                                                                    output_layer=output_layer)
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                      maximum_iterations=10)
            # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
            # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]
            # sample_id: [batch_size, decoder_targets_length], tf.int32

            # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
            # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
            # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
            # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
            if self.beam_search:
                self.decoder_predict_decode = decoder_outputs.predicted_ids
            else:
                self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)

        # 评价模式 不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        # 一般用于评价一个模型的效果，使用和train阶段不一样的数据，但是一般不适用于seq2seq模型
        elif self.mode == 'eval':
            pass

    def _create_lstm_cell(self):
        def single_rnn_cell():
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            return tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
        return tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
