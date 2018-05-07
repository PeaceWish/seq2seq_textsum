import tensorflow as tf


class BasicModel(object):
    def __init__(self, vocab_size, embedding_size, num_units, num_layers, max_target_sequence_length, batch_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_units = num_units
        self.num_layers = num_layers
        self.max_target_sequence_length = max_target_sequence_length
        self.batch_size = batch_size

    def _build_model(self):
        pass

    def _get_lstm_cell(self, num_units):
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell


class TrainModel(BasicModel):
    def __init__(self, vocab_size, embedding_size, num_units, num_layers, max_target_sequence_length, batch_size,
                 max_gradient_norm,
                 learning_rate):
        BasicModel.__init__(self, vocab_size, embedding_size, num_units, num_layers, max_target_sequence_length,
                            batch_size)
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate
        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.decoder_outputs = tf.placeholder(tf.int32, [None, None], name='decoder_outputs')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')

        # embedding 是可训练变量
        embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
        encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])
        # 执行encoder 操作
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                          sequence_length=self.source_sequence_length,
                                                          dtype=tf.float32)
        # 由于是文摘任务，可以使用同一个embedding
        decoder_outputs_embedded = tf.nn.embedding_lookup(embedding, self.decoder_outputs)
        # Build RNN cell with Attention
        decoder_cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])
        # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.num_units, encoder_output,
                                                                memory_sequence_length=self.source_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=self.num_units)
        # Helper
        training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_outputs_embedded, self.target_sequence_length,
                                                            time_major=False)
        # FC
        projection_layer = tf.layers.Dense(self.vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        # Decoder
        decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size).clone(
            cell_state=encoder_state)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                           initial_state=decoder_initial_state,
                                                           output_layer=projection_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=self.max_target_sequence_length)
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32,
                                 name="masks")
        # 损失函数 Loss
        self.train_loss = tf.contrib.seq2seq.sequence_loss(training_logits, self.decoder_outputs, masks)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.train_loss, trainable_params)
        # 梯度
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_params))

    def train(self, sess, batch):
        decoder_targets, encoder_inputs, decoder_targets_length, encoder_inputs_length = batch
        # 对于训练阶段，需要执行self.train_op, self.loss, 二个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: encoder_inputs,
                     self.source_sequence_length: encoder_inputs_length,
                     self.decoder_outputs: decoder_targets,
                     self.target_sequence_length: decoder_targets_length}
        _, _loss = sess.run([self.train_op, self.train_loss], feed_dict=feed_dict)
        return _loss


class InferenceModel(BasicModel):
    def __init__(self, vocab_size, embedding_size, num_units, num_layers, max_target_sequence_length, batch_size,
                 beam_size, segment_to_int, infer_mode='greedy'):
        BasicModel.__init__(self, vocab_size, embedding_size, num_units, num_layers, max_target_sequence_length,
                            batch_size)
        self.beam_size = beam_size
        self.segment_to_int = segment_to_int
        self.infer_mode = infer_mode
        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')  # infer输入
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        # embedding 是可训练变量
        embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
        encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])
        # 执行encoder 操作
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                          sequence_length=self.source_sequence_length,
                                                          dtype=tf.float32)
        # FC
        projection_layer = tf.layers.Dense(self.vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        if self.infer_mode == 'beam_search':
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(self.source_sequence_length,
                                                                  multiplier=self.beam_size)
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=self.beam_size)
            beam_attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                         memory=tiled_encoder_outputs,
                                                                         memory_sequence_length=tiled_sequence_length)
            beam_decoder_cell = tf.contrib.rnn.MultiRNNCell(
                [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])
            beam_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=beam_decoder_cell,
                                                                    attention_mechanism=beam_attention_mechanism,
                                                                    attention_layer_size=self.num_units)
            start_tokens = tf.tile(tf.constant([self.segment_to_int['<GO>']], dtype=tf.int32), [self.batch_size],
                                   name='start_token')
            end_token = self.segment_to_int['<EOS>']
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.beam_size)
            decoder_initial_state = beam_decoder_cell.zero_state(dtype=tf.float32,
                                                                 batch_size=self.batch_size * self.beam_size).clone(
                cell_state=tiled_encoder_final_state)
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=beam_decoder_cell,
                                                                      embedding=embedding,
                                                                      start_tokens=start_tokens,
                                                                      end_token=end_token,
                                                                      initial_state=decoder_initial_state,
                                                                      beam_width=self.beam_size,
                                                                      output_layer=projection_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                maximum_iterations=self.max_target_sequence_length)
            self.predict = predicting_decoder_output.predicted_ids

        else:
            # Helper
            start_tokens = tf.tile(tf.constant([self.segment_to_int['<GO>']], dtype=tf.int32), [self.batch_size],
                                   name='start_tokens')
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens, self.segment_to_int['<EOS>'])
            # Decoder
            decoder_cell = tf.contrib.rnn.MultiRNNCell(
                [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units, memory=encoder_output,
                                                                    memory_sequence_length=self.source_sequence_length)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.num_units)
            decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(
                cell_state=encoder_state)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state,
                                                                 output_layer=projection_layer)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                maximum_iterations=self.max_target_sequence_length)
            self.predict = predicting_decoder_output.sample_id

    def infer(self, sess, infer_batch):
        encoder_inputs, encoder_inputs_length = infer_batch
        # infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: encoder_inputs,
                     self.source_sequence_length: encoder_inputs_length}
        _predict = sess.run([self.predict], feed_dict=feed_dict)
        return _predict
