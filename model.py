__author__ = 'Rahul'

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


class SiameseRNN(object):
    def __init__(self, args):
        self.lstm_size = lstm_size
        self.num_of_layers = num_of_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        # self.num_of_chars = num_of_chars
        self.max_doc_length = max_doc_length
        self.max_grad_norm = max_grad_norm

    def get_hidden_states(self):
        self.inputs_1 = tf.placeholder(tf.float32, [self.batch_size, self.max_doc_length, 300])
        self.inputs_2 = tf.placeholder(tf.float32, [self.batch_size, self.max_doc_length, 300])

        self.inputs_1_sequence_lengths = tf.placeholder(tf.int32, [self.batch_size])
        self.inputs_2_sequence_lengths = tf.placeholder(tf.int32, [self.batch_size])

        self.labels = tf.placeholder(tf.int32, [self.batch_size])

        lstm = rnn_cell.BasicLSTMCell(self.lstm_size)
        stacked_lstm = rnn_cell.MultiRNNCell([lstm] * self.num_of_layers)

        self.state_1 = stacked_lstm.zero_state(self.batch_size, tf.float32)
        self.state_2 = stacked_lstm.zero_state(self.batch_size, tf.float32)

        #embeddings = tf.get_variable(name='embedding', shape=[self.num_of_chars, self.lstm_size])

        inputs_1 = [tf.squeeze(input_, [1]) for input_ in
                         tf.split(1, self.max_doc_length, self.inputs_1)]

        inputs_2 = [tf.squeeze(input_, [1]) for input_ in
                         tf.split(1, self.max_doc_length, self.inputs_2)]

        with tf.variable_scope('RNN'):
            outputs_1, self.state_1 = rnn.rnn(stacked_lstm, inputs=inputs_1,
                                              initial_state=self.state_1, sequence_length=self.inputs_1_sequence_lengths)
            tf.get_variable_scope().reuse_variables()
            outputs_2, self.state_2 = rnn.rnn(stacked_lstm, inputs=inputs_2,
                                              initial_state=self.state_2, sequence_length=self.inputs_2_sequence_lengths)

    def get_logits(self):
        weights = tf.Variable(tf.truncated_normal([2 * self.num_of_layers * self.lstm_size, 2]))
        biases = tf.Variable(tf.truncated_normal([2]))

        # concat_states = tf.concat(1, [self.state_1, self.state_2])

        distances = tf.abs(tf.sub(self.state_1, self.state_2))

        self.logits = tf.nn.xw_plus_b(distances, weights, biases)

    def train(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels)
        self.loss = tf.reduce_mean(cross_entropy)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)