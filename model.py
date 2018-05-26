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
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name='decoder_inputs')
        self.decoder_outputs = tf.placeholder(tf.int32, [None, None], name='decoder_outputs')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')

        # embedding 是可训练变量
        embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
        # FC
        projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False, name="output_projection")

        # 构建encoder-decoder
        #  创建encoder
        encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])
        #  执行encoder 操作
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                          sequence_length=self.source_sequence_length,
                                                          dtype=tf.float32)

        #  创建decoder(多层rnn+attention)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                memory=encoder_output,
                                                                memory_sequence_length=self.source_sequence_length)
        decoder_cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                           attention_mechanism=attention_mechanism,
                                                           attention_layer_size=self.num_units)
        decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size).clone(
            cell_state=encoder_state)

        decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.decoder_inputs)

        # Helper
        training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded, self.target_sequence_length,
                                                            time_major=False)
        # Decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                           initial_state=decoder_initial_state,
                                                           output_layer=projection_layer)
        # Dynamic decoding
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                                                decoder=training_decoder,
                                                                maximum_iterations=self.max_target_sequence_length)

        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        # Loss
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_outputs, logits=training_logits)
        target_weights = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length,
                                          dtype=training_logits.dtype)
        self.train_loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.train_loss, trainable_params)
        # 梯度
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # Optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_params))

    def train(self, sess, batch):
        pad_source_inputs_batch, pad_target_inputs_batch, pad_target_outputs_batch, source_lengths, targets_lengths = batch
        # 对于训练阶段，需要执行self.train_op, self.loss, 二个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: pad_source_inputs_batch,
                     self.source_sequence_length: source_lengths,
                     self.decoder_inputs: pad_target_inputs_batch,
                     self.decoder_outputs: pad_target_outputs_batch,
                     self.target_sequence_length: targets_lengths}
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
        # FC
        projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False, name="output_projection")

        # 构建encoder-decoder
        #  创建encoder
        encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])
        #  执行encoder 操作
        encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                          sequence_length=self.source_sequence_length,
                                                          dtype=tf.float32)

        #  创建decoder(多层rnn+attention)
        tgt_sos_id = self.segment_to_int['<GO>']
        tgt_eos_id = self.segment_to_int['<EOS>']

        if self.infer_mode == 'beam_search':
            memory = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=self.beam_size)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(self.source_sequence_length,
                                                                   multiplier=self.beam_size)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.beam_size)
        else:
            memory = encoder_output
            source_sequence_length = self.source_sequence_length

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                memory=memory,
                                                                memory_sequence_length=source_sequence_length)
        decoder_cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(self.num_units) for _ in range(self.num_layers)])

        alignment_history = (self.infer_mode != 'beam_search')

        cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=self.num_units,
                                                   alignment_history=alignment_history)

        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if self.infer_mode == 'beam_search':
            decoder_initial_state = cell.zero_state(self.batch_size * self.beam_size, dtype=tf.float32).clone(
                cell_state=encoder_state)
            # decoder
            infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell,
                                                                 embedding=embedding,
                                                                 start_tokens=start_tokens,
                                                                 end_token=end_token,
                                                                 initial_state=decoder_initial_state,
                                                                 beam_width=self.beam_size,
                                                                 output_layer=projection_layer)
        else:
            decoder_initial_state = cell.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            # helper
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens, end_token)
            # decoder
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state,
                                                            output_layer=projection_layer)

        # Dynamic decoding
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                                                                 infer_decoder,
                                                                 maximum_iterations=self.max_target_sequence_length)
        if self.infer_mode == 'beam_search':
            self.predict = predicting_decoder_output.predicted_ids
        else:
            self.logits = predicting_decoder_output.rnn_output
            self.predict = predicting_decoder_output.sample_id

    def infer(self, sess, infer_batch):
        encoder_inputs, encoder_inputs_length = infer_batch
        # infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: encoder_inputs,
                     self.source_sequence_length: encoder_inputs_length}
        _predict = sess.run([self.predict], feed_dict=feed_dict)
        return _predict


