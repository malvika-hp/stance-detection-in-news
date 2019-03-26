CSE 538 Stance Detection in News


#############################  Original Source ############################# 

    https://github.com/ourownstory/stance_detection

############################# Files Modified ############################# 
    1. execute_lstm_config.py - Renamed to execute_lstm_config_baseline.py 
    2. execute_lstm_conditional.py - Renamed to execute_lstm_conditional_baseline.py
    3. run_text_processing - Added new methods for new models
        - lwords2ids_vects
        - get_lexicon_data
        - lconcatConvert_np
        - save_lexicon_data
    4. our_util.py - Added new methods for new model
        - minibatch_lstm_sentiment
        - minibatches_lstm_sentiment
        - get_minibatches_lstm_sentiment

############################# Models  #################################

Model Files:
    our_model_config.py
        contains abstract model class to be extended by other models. Is based off of the model classes used in the course assignments.
    bow_model_config.py
        Bag of words model class that extends our_model_config.py
    basicLSTM_model_config_baseline.py
        model class for the basic LSTM model that operates on the concatenated input
    basicLSTM_sentiment_model_config
        model class for the basic LSTM with sentiment model that operates on the concatenated input
    LSTM_conditional_baseline.py
        model class for the LSTM with attention and conditional encoding
    BILSTM_conditional.py
        model class for the biLSTM model with attention and conditional encoding
    BILSTM_conditional_sentiment.py
        model class for the biLSTM model with attention, conditional encoding and sentiment

############################# Model executor classes used in execution script #################################

Model Execution Files
    execute_bow_config
        script that executes a single experiment of the bag of words model for a given set of parameters
    execute_lstm_config_baseline.py
        script that executes a single experiment of the basic LSTM baseline model for a given set of parameters
    execute_lstm_conditional_baseline.py
        script that executes a single experiment of the LSTM model with conditional encoding and attention for a given set of parameters for a given set of parameters
    execute_lstm_sentiment_config.py
        script that executes a single experiment of the basic LSTM with sentiment model for a given set of parameters
    execute_bilstm_conditional.py
        script that executes a single experiment of the biLSTM model with conditional encoding and attention for a given set of parameters for a given set of parameters
    execute_bilstm_conditional_sentiment.py
        script that executes a single experiment of the biLSTM model with conditional encoding, attention and sentiment for a given set of parameters for a given set of parameters

############################# Utility Files #################################

Utility Files
    our_util.py
        Utility functions for use in other files. Based on the example of the util.py files provided in course assignments.
    run_text_processing.py
        File that performas tokenization, loads the data, etc

############################# File for executing the models #################################
Runtime scripts
    test_script6.py
        Allows the user to define a set of experiments for any of the models described above.
    - To run any model just uncommnet following the comments written over the method calls
        
        For example: On running python test_script6.py If test_script6.py content is 

                print("-- Start BILSTM Conditional Experiments --")
                run_bilstm_conditional_with_parameters('')

        Then the script will execute biLSTM conditional model 

############################# System Requirements ##########################

    Python     - Python 2.7.15 :: Anaconda, Inc.
    TensorFlow - 1.12.0
    Nltk       - 3.3








