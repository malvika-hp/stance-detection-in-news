#!/usr/bin/env python2
# -*- coding: utf-8 -*-
######
# Execution script for the conditional LSTM with attention
# Based on starter code from PS3-CS224n
######
## General libraries
import tensorflow as tf
import numpy as np
import random

## Our Own Code
from BILSTM_conditional_sentiment import LSTMCondModel
from run_text_processing import save_data_pickle, get_data, get_lexicon_data, get_lexicon_data_bilstm_sentiment
from our_util import Progbar, minibatches, pack_labels, split_data, softmax, get_performance, convertOutputs, downsample_label, split_indices

def run_save_data_pickle(): ## Needs NLTK to be installed!
    save_data_pickle(outfilename = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
                    embedding_type = 'twitter.27B.50d',
                    parserOption = 'nltk')

def run_lstm_conditional(config, split = True, outputpath = '../../xp', final = False):
    ## Get data
    config, data_dict = get_data(config, 
            filename_embeddings = '/../../glove/glove.twitter.27B.50d.txt',
            pickle_path = '/../../glove/twitter50d_h_ids_b_ids_pickle.p',
            concat = False)
    
    config1, lex_data_dict = get_lexicon_data_bilstm_sentiment(config,
                            filename_embeddings = '/../../glove/glove.twitter.27B.50d.txt',
                            pickle_path = 'lexiconp.p',
                            concat = False)
    # print(data_dict_l)


    ## pass data into local namespace:
    y = data_dict['y']
    h = data_dict['h_np']
    b = data_dict['b_np']
    h_len = data_dict['h_seqlen']
    b_len = data_dict['b_seqlen']
    
    # Do shortening of dataset ## affects number of samples and max_len.
    if config.num_samples  is not None:
        ## Random seed
        np.random.seed(1)
        ind = range(np.shape(h)[0])
        random.shuffle(ind)
        indices = ind[0:config.num_samples ]
        h = h[indices,:]
        b = b[indices,:]
        h_len = h_len[indices]
        b_len = b_len[indices]
        y = y[indices]

    # Truncate headlines and bodies
    if config.h_max_len is not None:
        h_max_len = config.h_max_len
        if np.shape(h)[1] > h_max_len:
            h = h[:, 0:h_max_len]
        h_len = np.minimum(h_len, h_max_len)

    if config.b_max_len is not None:
        b_max_len = config.b_max_len
        if np.shape(b)[1] > b_max_len:
            b = b[:, 0:b_max_len]
        b_len = np.minimum(b_len, b_max_len)


    ####################### LEX DATA #######################

    #lex_y = lex_data_dict['y']
    #lex_h = lex_data_dict['h_np']
    #lex_h_len = lex_data_dict['h_seqlen']

    lex_b = lex_data_dict['h_b_np']
    lex_b_len = lex_data_dict['seqlen']




    # lex_y = data_dict_l['y']
    # lex_h_b_np = data_dict_l['h_b_np']
    # lex_seqlen = data_dict_l['seqlen']

    # lex_data = pack_labels(lex_h_b_np, lex_y, lex_seqlen)
    # if config1.num_samples is not None:
    #     lex_num_samples = config1.num_samples
    #     lex_data = lex_data[0:num_samples - 1]
    # lex_train_data, lex_dev_data, lex_test_data, lex_train_indices, lex_dev_indices, lex_test_indices = split_data(lex_data, prop_train = 0.6, prop_dev = 0.2, seed = 56) 
    # config1.num_samples = len(lex_train_indices)
    # config1.max_length = 75

    # print("lex_train_data", lex_train_data[0])

    if split:
        # Split data
        train_indices, dev_indices, test_indices = split_indices(np.shape(h)[0])
        # Divide data
        train_h = h[train_indices,:]
        train_b = b[train_indices,:]
        train_h_len = h_len[train_indices]
        train_b_len = b_len[train_indices]
        train_y = y[train_indices]


        ####################### LEX DATA #######################

        #lex_train_h = lex_h[train_indices,:]
        #lex_train_h_len = lex_h_len[train_indices]

        lex_train_b = lex_b[train_indices,:]
        lex_train_b_len = lex_b_len[train_indices]

        # test
        dev_h = h[dev_indices,:]
        dev_b = b[dev_indices,:]
        dev_h_len = h_len[dev_indices]
        dev_b_len = b_len[dev_indices]
        dev_y = y[dev_indices]

        ################## LEX ########################

        #lex_dev_h = lex_h[dev_indices,:]
        #lex_dev_h_len = lex_h_len[dev_indices]

        lex_dev_b = lex_b[dev_indices,:]
        lex_dev_b_len = lex_b_len[dev_indices]

        if final:
            # Combine train and dev
            train_dev_indices = train_indices + dev_indices
            train_h = h[train_dev_indices,:]
            train_b = b[train_dev_indices,:]
            train_h_len = h_len[train_dev_indices]
            train_b_len = b_len[train_dev_indices]
            train_y = y[train_dev_indices]

            ################## LEX  train ########################
            #lex_train_h = lex_h[train_dev_indices,:]
            #lex_train_h_len = lex_h_len[train_dev_indices]

            lex_train_b = lex_b[train_dev_indices,:]
            lex_train_b_len = lex_b_len[train_dev_indices]

            # Set dev to test
            dev_h = h[test_indices,:]
            dev_b = b[test_indices,:]
            dev_h_len = h_len[test_indices]
            dev_b_len = b_len[test_indices]
            dev_y = y[test_indices]

            ################### LEX dev ######################

            #lex_dev_h = lex_h[test_indices,:]
            #lex_dev_h_len = lex_h_len[test_indices]

            lex_dev_b = lex_b[test_indices,:]
            lex_dev_b_len = lex_b_len[test_indices]
      
    ## Passing parameter_dict to config settings
    ## Changes to config  based on data shape
    assert(np.shape(train_h)[0] == np.shape(train_b)[0] == np.shape(train_y)[0] == np.shape(train_h_len)[0] == np.shape(train_b_len)[0] == np.shape(lex_train_b_len)[0])
    config.num_samples = np.shape(train_h)[0]
    config.h_max_len = np.shape(train_h)[1]
    config.b_max_len = np.shape(train_b)[1]
    
    ## Start Tensorflow!
    print('Starting TensorFlow operations')
    print 'With hidden layers: ', config.n_layers ## hidden layer?
    with tf.Graph().as_default():
        tf.set_random_seed(1)
        model = LSTMCondModel(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init) 
            losses_ep, dev_performances_ep, dev_predicted_classes_ep, dev_predictions_ep = model.fit(session, train_h, train_b, train_h_len, train_b_len, train_y, dev_h, dev_b, dev_h_len, dev_b_len, dev_y, lex_train_b, lex_train_b_len, lex_dev_b, lex_dev_b_len) #M

    # Write results to csv
    convertOutputs(outputpath, config, losses_ep, dev_performances_ep)

    print('Losses ', losses_ep)
    print('Dev Performance ', dev_performances_ep) #M
    return losses_ep, dev_predicted_classes_ep, dev_performances_ep #M

## for debugging
if __name__ == "__main__":
    print('Doing something!')
    run_save_data_pickle()
    losses, dev_predicted_classes, dev_performance = run_bow(num_samples = 1028)
    print('Execution Complete')
