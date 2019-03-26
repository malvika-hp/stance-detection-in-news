######
# basic BOW model with architecture extendable to more complex LSTM models which use both headings and bodies separately.
######
import tensorflow as tf
import numpy as np
import random

from our_model_config import OurModel
from our_util import Progbar, minibatches, get_performance, softmax

class LSTMCondModel(OurModel):

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.headings_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.h_max_len), name = "headings")
        self.bodies_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.b_max_len), name = "bodies")
        
        #self.lex_headings_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.b_max_len), name = "lex_headings")
        self.lex_bodies_placeholder = tf.placeholder(tf.int64, shape=(None, self.config.b_max_len), name = "lex_bodies")
        
        self.headings_lengths_placeholder = tf.placeholder(tf.int64, shape=(None), name = "headings_lengths")
        self.bodies_lengths_placeholder = tf.placeholder(tf.int64, shape=(None), name = "bodies_lengths")
        
        #self.lex_headings_lengths_placeholder = tf.placeholder(tf.int64, shape=(None), name = "lex_headings_lengths_placeholder")
        self.lex_bodies_lengths_placeholder = tf.placeholder(tf.int64, shape=(None), name = "lex_bodies_lengths_placeholder")
        
        self.labels_placeholder = tf.placeholder(tf.int64, shape=(None), name = "labels")
        self.dropout_placeholder = tf.placeholder(tf.float64, name = 'dropout')

    def create_feed_dict(self, headings_batch, bodies_batch, headings_lengths_batch, bodies_lengths_batch, lex_bodies_batch, lex_bodies_lengths_batch, labels_batch=None, dropout = 1.0):
        """Creates the feed_dict for the model.
        """
        # print("############################   Create feed dict ###########################")
        # print("body:   ",bodies_batch)
        # print("lex:   ",lex_bodies_batch)
        #
        # print("body len:   ",len(bodies_lengths_batch))
        # print("lex len:   ",len(lex_bodies_lengths_batch))

        feed_dict = {
            self.headings_placeholder: headings_batch,
            self.bodies_placeholder: bodies_batch,
            self.headings_lengths_placeholder: headings_lengths_batch,
            self.bodies_lengths_placeholder: bodies_lengths_batch,
            # self.lex_headings_placeholder: lex_headings_batch,
            # self.lex_headings_lengths_placeholder: lex_headings_lengths_batch,
            self.lex_bodies_placeholder: lex_bodies_batch,
            self.lex_bodies_lengths_placeholder: lex_bodies_lengths_batch
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def add_embedding(self, option = 'Constant'):
        """Adds an embedding layer that maps from input tokens (integers) to vectors for both the headings and bodies:

        Returns:
            embeddings_headings: tf.Tensor of shape (None, h_max_len, embed_size)
            embeddings_bodies: tf.Tensor of shape (None, b_max_len, embed_size)
        """
        if option == 'Constant':
            embeddings_headings_temp = tf.nn.embedding_lookup(params = tf.constant(self.config.pretrained_embeddings), ids = self.headings_placeholder)
            embeddings_bodies_temp   = tf.nn.embedding_lookup(params = tf.constant(self.config.pretrained_embeddings), ids = self.bodies_placeholder)
        elif option == 'Variable':
            embeddings_headings_temp = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.headings_placeholder)
            embeddings_bodies_temp   = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.bodies_placeholder)
            #embeddings_lex_headings_temp = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.lex_headings_placeholder)
            embeddings_lex_bodies_temp = tf.nn.embedding_lookup(params = tf.Variable(self.config.pretrained_embeddings), ids = self.lex_bodies_placeholder)
        
        embeddings_headings = tf.reshape(embeddings_headings_temp, shape = (-1, self.config.h_max_len, self.config.embed_size))
        embeddings_bodies = tf.reshape(embeddings_bodies_temp, shape = (-1, self.config.b_max_len, self.config.embed_size))
        return embeddings_headings, embeddings_bodies, embeddings_lex_bodies_temp

    def add_prediction_op(self):

        with tf.variable_scope('head'):

            # LSTM that handles the headers
            cell_h = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_h = tf.nn.rnn_cell.DropoutWrapper(cell_h, output_keep_prob = self.dropout_placeholder)

            cell_h_back = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_h_back = tf.nn.rnn_cell.DropoutWrapper(cell_h_back, output_keep_prob = self.dropout_placeholder)


            theInitializer = tf.contrib.layers.xavier_initializer(uniform = True, dtype = tf.float64)

            # x = self.inputs_placeholder
            x_header, x_body, x_lex_body = self.add_embedding(option = self.config.trainable_embeddings)
            # print('Predict op: x', x)
            #rnnOutput_h = tf.nn.dynamic_rnn(cell_h, inputs = x_header, dtype = tf.float64, sequence_length = self.headings_lengths_placeholder) #MODIF

            (rnnOutput_h, rnnState_h) = tf.nn.bidirectional_dynamic_rnn(cell_h,cell_h_back, inputs = x_header, dtype = tf.float64, sequence_length = self.headings_lengths_placeholder) #MODIF

            #rnnOutput_h = tf.reshape(rnnOutput_h, shape = (-1, self.config.hidden_size))
            #print("rnn h out fw 0 shape ", rnnOutput_h_fw[0].shape)
            #print("rnn h out fw 1 shape ", rnnOutput_h_bw[1].shape)

            Y_fw = tf.slice(rnnOutput_h[0], begin = [0, 0, 0], size = [-1, self.config.attention_length, -1])
            Y_bw = tf.slice(rnnOutput_h[1], begin = [0, 0, 0], size = [-1, self.config.attention_length, -1])

            Y = tf.add(Y_fw, Y_bw)


        # with tf.variable_scope('lex_head'):
        #     lex_cell_h = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
        #     lex_cell_h = tf.nn.rnn_cell.DropoutWrapper(lex_cell_h, output_keep_prob = self.dropout_placeholder)
        #
        #     lex_cell_h_back = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
        #     lex_cell_h_back = tf.nn.rnn_cell.DropoutWrapper(lex_cell_h_back, output_keep_prob = self.dropout_placeholder)
        #
        #     (lex_rnnOutput_h, lex_rnnState_h) = tf.nn.bidirectional_dynamic_rnn(cell_h,cell_h_back, inputs = x_header, dtype = tf.float64, sequence_length = self.headings_lengths_placeholder) #MODIF
        #
        #     Y_lex_head_fw = tf.slice(lex_rnnOutput_h[0], begin = [0, 0, 0], size = [-1, self.config.attention_length, -1])
        #     Y_lex_head_bw = tf.slice(lex_rnnOutput_h[1], begin = [0, 0, 0], size = [-1, self.config.attention_length, -1])
        #
        #     Y_lex_head = tf.add(Y_lex_head_fw, Y_lex_head_bw)
        #
        #
        #     Y = tf.add(Y_head, Y_lex_head)

        with tf.variable_scope('body'):
            # LSTM that handles the bodies
            cell_b = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_b = tf.nn.rnn_cell.DropoutWrapper(cell_b, output_keep_prob = self.dropout_placeholder)

            cell_b_back = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            cell_b_back = tf.nn.rnn_cell.DropoutWrapper(cell_b_back, output_keep_prob = self.dropout_placeholder)

            U_b = tf.get_variable(name = 'U_b', shape = (self.config.hidden_size, self.config.n_classes), initializer = theInitializer, dtype = tf.float64)
            b_b = tf.get_variable(name = 'b_b', shape = (self.config.n_classes), initializer = theInitializer, dtype = tf.float64)

            rnnOutput_b, rnnState_b = tf.nn.bidirectional_dynamic_rnn(cell_b, cell_b_back, inputs = x_body, dtype = tf.float64, initial_state_fw = rnnState_h[0], initial_state_bw = rnnState_h[1],sequence_length = self.bodies_lengths_placeholder)
            
            #print("Body shape.   ", x_body.shape)
            #print rnnOutput_b[1]
            #h_N = rnnOutput_b[1][1]
            
            # Removed becuse of including sentiment
            # h_N = tf.multiply(rnnState_b[0][1], rnnState_b[1][1]) # batch_size, cell.state_size
            body_last_state = tf.add(rnnState_b[0][1], rnnState_b[1][1])

        with tf.variable_scope('lex_body'):

            lex_cell_b = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            lex_cell_b = tf.nn.rnn_cell.DropoutWrapper(lex_cell_b, output_keep_prob = self.dropout_placeholder)

            lex_cell_b_back = tf.nn.rnn_cell.BasicLSTMCell(num_units = self.config.hidden_size)
            lex_cell_b_back = tf.nn.rnn_cell.DropoutWrapper(lex_cell_b_back, output_keep_prob = self.dropout_placeholder)



            lex_rnnOutput_b, lex_rnnState_b = tf.nn.bidirectional_dynamic_rnn(lex_cell_b,lex_cell_b_back, inputs = x_lex_body, dtype = tf.float64, sequence_length = self.lex_bodies_lengths_placeholder)



            lex_last_state = tf.add(lex_rnnState_b[0][1], lex_rnnState_b[1][1]) # batch_size, cell.state_size

            h_N = tf.add(body_last_state, lex_last_state)


        ## ATTENTION!
        W_y = tf.get_variable(name = 'Wy', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        W_h = tf.get_variable(name = 'Wh', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        w = tf.get_variable(name = 'w', shape = (self.config.hidden_size, 1), initializer = theInitializer, dtype = tf.float64)
        W_p = tf.get_variable(name = 'Wo', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)
        W_x = tf.get_variable(name = 'Wx', shape = (self.config.hidden_size, self.config.hidden_size), initializer = theInitializer, dtype = tf.float64)

        M_1 = tf.reshape(tf.matmul(tf.reshape(Y, shape = (-1, self.config.hidden_size)), W_y), shape = (-1, self.config.attention_length, self.config.hidden_size))
        #print M_1
        M_2 = tf.expand_dims(tf.matmul(h_N, W_h), axis = 1)
        M = tf.tanh(M_1 + M_2)
        alpha = tf.reshape(tf.nn.softmax(tf.matmul(tf.reshape(M, shape = (-1, self.config.hidden_size)), w)), shape = (-1, self.config.attention_length))

        r = tf.squeeze(tf.matmul(tf.transpose(tf.expand_dims(alpha, 2), perm = [0, 2, 1]), Y))
        h_star = tf.tanh(tf.matmul(r, W_p) + tf.matmul(h_N, W_x))

        # Compute predictions
        preds = tf.matmul(h_star, U_b) + b_b # batch_size, n_classes
        return preds


    def train_on_batch(self, sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch, lex_bodies_batch, lex_bodies_len):
        """Perform one step of gradient descent on the provided batch of data.
        Args:
            sess: tf.Session()
            headings_batch: np.ndarray of shape (n_samples, n_features)
            bodies_batch: np.ndarray of shape (n_samples, n_features)
            headings_lengths_batch: np.ndarray of shape (n_samples, 1)
            bodies_lengths_batch: np.ndarray of shape (n_samples, 1)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        feed = self.create_feed_dict(h_batch, b_batch, h_len_batch, b_len_batch, lex_bodies_batch, lex_bodies_len, y_batch, dropout = self.config.dropout)
        # print('feed', feed)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        ## for debugging / testing
        if (np.isnan(loss)):
            print('headings', h_batch)
            print('bodies', b_batch)
            print('nh_len', h_len_batch)
            print('b_len', b_len_batch)
            print('labels', y_batch)
            assert(False)
        return loss

    def predict_on_batch(self, sess, h_batch, b_batch, h_len_batch, b_len_batch, lex_bodies_batch, lex_bodies_len):
        """Make predictions for the provided batch of data
        Args:
            sess: tf.Session()
            headings_batch: np.ndarray of shape (n_samples, n_features)
            bodies_batch: np.ndarray of shape (n_samples, n_features)
            headings_lengths_batch: np.ndarray of shape (n_samples, 1)
            bodies_lengths_batch: np.ndarray of shape (n_samples, 1)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(h_batch, b_batch, h_len_batch, b_len_batch, lex_bodies_batch, lex_bodies_len)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_epoch(self, sess, h_np, b_np, h_len, b_len, y, lex_bodies_batch, lex_bodies_len):
        # prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses = []
        # shuffle
        ind = range(self.config.num_samples)
        random.shuffle(ind)
        # sizes
        batch_start = 0
        batch_end = 0       
        N = self.config.batch_size
        num_batches = self.config.num_samples / N
        # run batches
        for i in range(num_batches):
            batch_start = (i*N)
            batch_end = (i+1)*N
            indices = ind[batch_start:batch_end]
            h_batch = h_np[indices,:]
            b_batch = b_np[indices,:]
            h_len_batch = h_len[indices]
            b_len_batch = b_len[indices]

            # l_h_batch = lex_headings_batch[indices,:]
            # l_h_len = lex_headings_len[indices]
            l_b_batch = lex_bodies_batch[indices,:]
            l_b_len = lex_bodies_len[indices]

            y_batch = y[indices]
            loss = self.train_on_batch(sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch, l_b_batch, l_b_len)
            losses.append(loss)
            if (i % (1 + num_batches/10)) == 0:
                print('batch: ', i, ', loss: ', loss)
        # run last smaller batch
        if (batch_end < self.config.num_samples):
            indices = ind[batch_end:]
            h_batch = h_np[indices,:]
            b_batch = b_np[indices,:]
            h_len_batch = h_len[indices]
            b_len_batch = b_len[indices]
            # l_h_batch = lex_headings_batch[indices,:]
            # l_h_len = lex_headings_len[indices]
            l_b_batch = lex_bodies_batch[indices, :]
            l_b_len = lex_bodies_len[indices]
            y_batch = y[indices]
            # loss
            loss = self.train_on_batch(sess, h_batch, b_batch, h_len_batch, b_len_batch, y_batch, l_b_batch, l_b_len)
            losses.append(loss)
        return losses

    def fit(self, sess, h_np, b_np, h_len, b_len, y, dev_h, dev_b, dev_h_len, dev_b_len, dev_y, lex_train_bodies_batch, lex_train_bodies_len, lex_dev_bodies_batch, lex_dev_bodies_len): #M
        #losses = []
        losses_epochs = [] #M
        dev_performances_epochs = [] # M
        dev_predictions_epochs = [] #M
        dev_predicted_classes_epochs = [] #M

        for epoch in range(self.config.n_epochs):
            print('-------new epoch---------')
            loss = self.run_epoch(sess, h_np, b_np, h_len, b_len, y, lex_train_bodies_batch, lex_train_bodies_len)

            # Computing predictions #MODIF
            dev_predictions = self.predict_on_batch(sess, dev_h, dev_b, dev_h_len, dev_b_len, lex_dev_bodies_batch, lex_dev_bodies_len)

            # Computing development performance #MODIF
            dev_predictions = softmax(np.array(dev_predictions))
            dev_predicted_classes = np.argmax(dev_predictions, axis = 1)
            dev_performance = get_performance(dev_predicted_classes, dev_y, n_classes = 4)

            # Adding to global outputs #MODIF
            dev_predictions_epochs.append(dev_predictions)
            dev_predicted_classes_epochs.append(dev_predicted_classes)
            dev_performances_epochs.append(dev_performance)
            losses_epochs.append(loss)

            print("prediction epochs: ", dev_performances_epochs)
            print("predicted classes epochs: ", dev_predicted_classes_epochs)
            print("performances epochs: ", dev_performances_epochs)
            print('EPOCH: ', epoch, ', LOSS: ', np.mean(loss))

        return losses_epochs, dev_performances_epochs, dev_predicted_classes_epochs, dev_predictions_epochs

    def __init__(self, config):
        self.config = config
        self.headings_placeholder = None
        self.bodies_placeholder = None
        self.headings_lengths_placeholder = None
        self.bodies_lengths_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None
        self.build()
