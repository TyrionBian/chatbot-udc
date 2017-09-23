"""
Model to predict the matching score between query and responses.

"""
import numpy as np
import tensorflow as tf

class Ranker:
    """ A retrieval-based chatbot.
    Architecture: LSTM Encoder/Encoder.
    """

    def __init__(self, args, is_training):
        """
        Args:
            args: 模型的超参数。
        """
        print("模型初始化...")

        self.is_training = is_training

        self.args = args
        self.dtype = tf.float32

        self.optOp = None    # 用于训练阶段
        self.outputs = None  # 用于测试阶段

        # 搭建模型的computational graph
        self.buildNetwork()

    def buildNetwork(self):
        """ 搭建模型的computational graph
        """
        # 定义 rnn cell
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(
                self.args.hiddenSize,
            )
            if self.is_training:
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    output_keep_prob=self.args.dropout
                )
            return cell

        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.numLayers)],
        )

        # Network input (placeholders)
        with tf.name_scope('placeholder_query'):
            self.query_seqs  = tf.placeholder(tf.int32, [None, None], name='query')
            self.query_length  = tf.placeholder(tf.int32, [None], name='query_length')

        with tf.name_scope('placeholder_response'):
            self.response_seqs = tf.placeholder(tf.int32, [None, None], name='response')
            self.response_length = tf.placeholder(tf.int32, [None], name='response_length')

        with tf.name_scope('placeholder_labels'):
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
            self.targets = tf.placeholder(tf.int32, [None], name='targets')

        with tf.name_scope('embedding_layer'):
            self.embedding = tf.get_variable('embedding',
             [self.args.vocabularySize, self.args.embeddingSize])
            self.embed_query = tf.nn.embedding_lookup(self.embedding, self.query_seqs)
            self.embed_response = tf.nn.embedding_lookup(self.embedding, self.response_seqs)
            if self.is_training and self.args.dropout > 0:
                self.embed_query = tf.nn.dropout(self.embed_query, keep_prob = self.args.dropout)
                self.embed_response = tf.nn.dropout(self.embed_response, keep_prob = self.args.dropout)

        query_output, query_final_state = tf.nn.dynamic_rnn(
            cell = encoder_cell,
            inputs = self.embed_query,
            sequence_length = self.query_length,
            time_major = False,
            dtype=tf.float32)

        response_output, response_final_state = tf.nn.dynamic_rnn(
            cell = encoder_cell,
            inputs = self.embed_response,
            sequence_length = self.response_length,
            time_major = False,
            dtype=tf.float32)

        with tf.variable_scope('bilinar_regression'):
             W = tf.get_variable("bilinear_W",
                    shape=[self.args.hiddenSize, self.args.hiddenSize],
                           initializer=tf.truncated_normal_initializer())


        if self.is_training:
            # 训练阶段, 使用minibatch内其他样本的response作为negative response
            response_final_state = tf.matmul(response_final_state[-1].h, W)
            logits = tf.matmul(
                a = query_final_state[-1].h, b = response_final_state,
                transpose_b = True)
            self.losses = tf.losses.softmax_cross_entropy(
                onehot_labels = self.labels,
                logits = logits)
            self.mean_loss = tf.reduce_mean(self.losses, name="mean_loss")
            train_loss_summary = tf.summary.scalar('loss', self.mean_loss)
            self.training_summaries = tf.summary.merge(
                                     inputs = [train_loss_summary], name='train_monitor')

            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.mean_loss)

        else:
            # 测试阶段，每一个样本的negative response是固定的
            # [batch_size x 20, rnn_dim]
            response_final_state = tf.matmul(response_final_state[-1].h, W)
            query_final_state = tf.reshape(
                    tf.tile(query_final_state[-1].h, [1, 20]),
                    [-1, self.args.hiddenSize])
            # [batch_size, batch_size x 20]
            logits = tf.reduce_sum(
                    tf.multiply(
                        x = query_final_state,
                        y = response_final_state),
                    axis = 1,
                    keep_dims = True)
            logits = tf.reshape(logits, [-1, 20])
            # top_k percentage
            self.response_top_1 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(
                        predictions = logits,
                        targets = self.targets,
                        k = 1,
                        name = 'prediction_in_top_1'),
                    dtype = tf.float32))
            self.response_top_3 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(
                        predictions = logits,
                        targets = self.targets,
                        k = 3,
                        name = 'prediction_in_top_3'),
                    dtype = tf.float32))
            self.response_top_5 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(
                        predictions = logits,
                        targets = self.targets,
                        k = 5,
                        name = 'prediction_in_top_5'),
                    dtype = tf.float32))

            top1_summary = tf.summary.scalar('valid_top1_of20', self.response_top_1)
            top3_summary = tf.summary.scalar('valid_top3_of20', self.response_top_3)
            top5_summary = tf.summary.scalar('valid_top5_of20', self.response_top_5)
            self.evaluation_summaries = tf.summary.merge(
                                     inputs = [top1_summary, top3_summary, top5_summary],
                                     name='eval_monitor')

            self.outputs = (self.response_top_1,
                self.response_top_3, self.response_top_5, logits)
    def step(self, batch):
        """ Forward/training step operation.
        """
        def zero_initial_state(batch_size, embed_dim, num_layers):
            return tuple(
                [(np.zeros((batch_size, embed_dim)),
                np.zeros((batch_size, embed_dim)))
            for _ in range(num_layers)])

        # Feed the dictionary
        feedDict = {}
        ops = None

        feedDict[self.query_seqs] = batch.query_seqs
        feedDict[self.query_length] = batch.query_length
        feedDict[self.response_seqs] = batch.response_seqs
        feedDict[self.response_length] = batch.response_length

        if self.is_training:  # Training
            ops = (self.optOp, self.mean_loss, self.training_summaries)
            feedDict[self.labels] = np.eye(len(batch.query_seqs))
        else: # Testing or Validating
            ops = (self.outputs, self.evaluation_summaries)
            feedDict[self.targets] = np.zeros((len(batch.query_seqs))).astype(int)
        # Return one pass operator
        return ops, feedDict


