#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reads a pickle file with training data, uses this data to train a RNN and stores the resulting model.
"""

import os
import argparse
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

from keras.preprocessing.sequence import pad_sequences
from keras.engine.sequential import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, TimeDistributed, SimpleRNN, Bidirectional
from keras.metrics import Accuracy, Precision, Recall, TruePositives, FalsePositives, FalseNegatives,\
    TrueNegatives

# configuration of the training parameters
BATCH_SIZE = 1000
EPOCH_NUM = 150


def load_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses the data from the given pickle file in a way that it can be fed into the RNN.

    :param data_path: the training or test data as a pickle file
    :return: x and y array that can be fed into the RNN
    """

    with open(data_path, 'rb') as pickle_data:
        data = pickle.load(pickle_data)
    amount_slices = len(data[0])
    slices_lengths = [len(byte_slice) for byte_slice in data[0]]
    max_slice_length = max(slices_lengths)

    # load the byte values
    # appends zeros to slices that do not have the maximum length
    byte_slices = pad_sequences(data[0], maxlen=max_slice_length, dtype='int32', padding='post', value=0)
    # add one to each byte value; this is necessary as we do not want any zeros to cancel out values in the RNN
    byte_slices += 1

    # load the function starts
    # do the same padding as above
    function_starts = pad_sequences(data[1], maxlen=max_slice_length, dtype='int32', padding='post', value=0)
    # create a 2d-vector per byte that determines if the byte marks a function start
    function_start_vectors = np.zeros((amount_slices, max_slice_length, 2), dtype=function_starts.dtype)
    for byte_id in range(amount_slices):
        function_start_vectors[byte_id, np.arange(max_slice_length), function_starts[byte_id]] = 1

    return byte_slices, function_start_vectors


def train_model(training_data: Tuple[np.ndarray, np.ndarray]) -> Sequential:
    """
    Trains the RNN with the given training data.

    :param training_data: training data as returned by load_data
    :return: the trained model
    """

    # defining the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=257, output_dim=16, input_length=len(training_data[0][0])))
    model.add(Bidirectional(SimpleRNN(units=8, activation='relu', dropout=0.5, return_sequences=True)))
    model.add(TimeDistributed(Dense(2, activation='softmax'), input_shape=(len(training_data[0][0]), 8)))
    # compile the model
    model.compile('adam', 'categorical_crossentropy', metrics=[Accuracy(),
                                                               Precision(name='precision'),
                                                               Recall(name='recall'),
                                                               TruePositives(name='tp'),
                                                               FalsePositives(name='fp'),
                                                               FalseNegatives(name='fn'),
                                                               TrueNegatives(name='tn')])

    # train the model
    model.fit(training_data[0], training_data[1], batch_size=BATCH_SIZE, epochs=EPOCH_NUM)

    # log the results for each epoch
    logger = logging.getLogger('__main__')
    for idx, epoch in enumerate(zip(model.history.history['loss'], model.history.history['accuracy'],
                                    model.history.history['precision'], model.history.history['recall'])):
        logger.info(f"Epoch {idx}: loss: {epoch[0]}, accuracy {epoch[1]}, precision {epoch[2]}, recall {epoch[3]}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Reads a pickle file with training data, uses this data to train a RNN"
                                                 " and stores the resulting model.")
    parser.add_argument('training_data',
                        metavar='TRAINING_DATA',
                        help="Path to the pickle file that contains the training data.")
    parser.add_argument('model',
                        metavar='MODEL',
                        help="Path to the file that should store the resulting model.")

    args = parser.parse_args()

    # check if the pickle training file exists
    training_data_path = Path(args.training_data)
    if not training_data_path.exists():
        raise FileNotFoundError(f"The given file {args.training_data} was not found.")

    # check if the model is stored in a valid directory
    model_path = Path(args.model)
    if not model_path.parent.exists():
        raise ValueError(f"The path where the model should be stored '{model_path.parent}' does not exist.")
    if not os.access(model_path.parent.absolute(), os.W_OK):
        raise ValueError(f"The path where the model should be stored '{model_path.parent}' is not writable.")

    # load the training data so that it can be presented to the RNN
    training_data = load_data(training_data_path)

    # train and store the model
    model = train_model(training_data)
    model.save(model_path)


if __name__ == '__main__':
    main()
