#!/usr/bin/env python2
# -*- coding: utf-8 -*-

######
# Call all models with different hyperparameters
######

# standard libs
import numpy as np

# our code imports
from execute_bow_config import run_bow
from execute_lstm_config import run_lstm
from execute_lstm_attention import run_lstm_attention
from execute_lstm_conditional_baseline import run_lstm_conditional as run_lstm_conditional_baseline
from execute_lstm_config_baseline import run_lstm as run_lstm_baseline
from execute_lstm_sentiment_config import run_lstm as run_lstm_sentiment
from execute_bilstm_conditional_sentiment import run_lstm_conditional as run_bilstm_conditional_sentiment
from execute_bilstm_conditional import run_lstm_conditional as run_bilstm_conditional

### Parameter Overview:
class Config:
  """Holds model hyperparams and data information.
  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation. Use self.config.? instead of Config.?
  """
  ### Parameter Overview:
  ## For all models:
  # main params,
  n_epochs = 2
  lr = 0.001
  batch_size = 128 
  n_classes = 4 
  hidden_size = 100
  n_layers = 0
  xp = None
  model = None

  ## Determined at data loading:
  embed_size = None  # not passed to config - assigned in get_data
  vocab_size = None  # not passed to config - assigned in get_data
  pretrained_embeddings = []  # not passed to config   - assigned in get_data
  num_samples = None  # only indirectly passed to comfig, If defined, shortens the dataset, Otherwise determined at data loading,
  downsample = False

  ## LSTM specific:
  # main params 
  dropout  = 0.8  ## Attention: this is the keep_prob! # not assigned to BOW
  # extra_hidden_size = None
  trainable_embeddings = 'Variable'
  max_length = None  # indirectly passed to config in LSTM, If defined, truncates sequences, Otherwise determined at data loading
  attention_length = 15

  ## BOW specific:
  # main params
  hidden_next = 0.6  # defines the number of hidden units in next layer
  # Determined at data loading:
  h_max_len = None  # not passed to config
  b_max_len = None  # not passed to config


########################################################################################################
############################################# BASELINE MODEL EXECUTION ################################
########################################################################################################

def run_bow_with_parameters(args):

  # Final test st
  np.random.seed(1)
  config = Config()
  config.n_layers = 1
  config.xp = 'final_test'
  config.model = 'bow'
  config.lr = 0.005
  config.trainable_embeddings = 'Variable'
  config.b_max_len = 600
  config.n_epochs = 40
  result = run_bow(config, final = True)

def run_lstm_with_parameters(args):
  # Final test
  np.random.seed(1)
  config0 = Config()
  config0.max_length = 75
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 1
  config0.n_layers = 2
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.lr = 0.001
  # config0.num_samples = 100
  config0.xp = 'final_test'
  config0.model = 'lstm_basic'
  result = run_lstm_baseline(config0, final = True)

def run_lstm_conditional_baseline_with_parameters(args):
  # To be defined - parameter saving not ready
  np.random.seed(1)
  config0 = Config()
  # print('Running run_lstm_with_parameters')
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 10
  config0.n_layers = 1
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.n_layers = 2
  config0.lr = 0.001
  # config0.num_samples = 100
  config0.b_max_len = 75
  config0.attention_length = 15
  config0.xp = 'final_test'
  config0.model = 'conditional_lstm'
  # print 'config0' + str(config0.__dict__)
  result0 = run_lstm_conditional_baseline(config0, final = True)

########################################################################################################
################################### IMPROVMENT OVER BASELINE EXECUTIONS ################################
########################################################################################################


def run_lstm_sentiment_with_parameters(args):
  # Final test
  np.random.seed(1)
  config0 = Config()
  config0.max_length = 75
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 10
  config0.n_layers = 2
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.lr = 0.001
  # config0.num_samples = 100
  config0.xp = 'final_test'
  config0.model = 'lstm_basic_sentiment'
  result = run_lstm_sentiment(config0, final = True)

def run_bilstm_conditional_with_parameters(args):
  np.random.seed(1)
  config0 = Config()
  # print('Running run_lstm_with_parameters')
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 1
  config0.n_layers = 1
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.n_layers = 2
  config0.lr = 0.001
  # config0.num_samples = 100
  config0.b_max_len = 75
  config0.attention_length = 15
  config0.xp = 'final_test'
  config0.model = 'conditional_bilstm'
  # print 'config0' + str(config0.__dict__)
  result0 = run_bilstm_conditional(config0, final = True)

def run_bilstm_conditional_sentiment_with_parameters(args):
  np.random.seed(1)
  config0 = Config()
  # print('Running run_lstm_with_parameters')
  config0.trainable_embeddings = 'Variable'
  config0.hidden_size = 100
  config0.n_epochs = 10
  config0.n_layers = 1
  config0.batch_size = 128
  config0.dropout = 0.8
  config0.n_layers = 2
  config0.lr = 0.001
  # config0.num_samples = 100
  config0.b_max_len = 300
  config0.attention_length = 15
  config0.xp = 'final_test'
  config0.model = 'conditional_bilstm_sentiment'
  # print 'config0' + str(config0.__dict__)
  result0 = run_bilstm_conditional_sentiment(config0, final = True)

if __name__ == "__main__":
  print("-- Running Test Script --")
  #print("-- Start BOW Experiments --")
  #run_bow_with_parameters('')

  # print("-- Start LSTM Basic Experiment --")
  # run_lstm_with_parameters('')

  # print("-- Start LSTM Conditional Baseline Experiments --")
  # run_lstm_conditional_baseline_with_parameters('')

  # print("-- Start LSTM Basic Sentiment Experiment --")
  # run_lstm_sentiment_with_parameters('')

  # print("-- Start BILSTM Conditional Sentiment Experiments --")
  # run_bilstm_conditional_sentiment_with_parameters('')

  print("-- Start BILSTM Conditional Experiments --")
  run_bilstm_conditional_with_parameters('')


  # print("-- Finished Test Script --")