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
import tensorflow as tf
from pathlib import Path
from typing import Tuple
from keras.metrics import BinaryAccuracy, Precision, Recall, TruePositives, FalsePositives, FalseNegatives,\
    TrueNegatives
from keras.preprocessing.sequence import pad_sequences
from keras.engine.sequential import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, TimeDistributed, SimpleRNN, Bidirectional

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

    return byte_slices, function_starts


def compute_initial_bias(ground_truth: np.ndarray) -> float:
    """
    Computes the initial bias.
    We use the formula described here:
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias

    :param ground_truth: ground truth data that will be used for training
    :returns: the initial bias
    """

    pos = np.count_nonzero(ground_truth)
    neg = np.count_nonzero(ground_truth == 0)
    return np.log([pos/neg])


def train_model(training_data: Tuple[np.ndarray, np.ndarray]) -> Sequential:
    """
    Trains the RNN with the given training data.

    :param training_data: training data as returned by load_data
    :return: the trained model
    """

    # compute the initial bias
    output_bias = tf.keras.initializers.Constant(compute_initial_bias(training_data[1]))

    # defining the model architecture
    model = Sequential()
    model.add(Embedding(input_dim=257, output_dim=16, input_length=len(training_data[0][0])))
    model.add(Bidirectional(SimpleRNN(units=8, activation='relu', dropout=0.5, return_sequences=True)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid', bias_initializer=output_bias),
                              input_shape=(len(training_data[0][0]), 8)))
    # compile the model
    model.compile('adam', 'binary_crossentropy', metrics=[BinaryAccuracy(),
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
    for idx, epoch in enumerate(zip(model.history.history['loss'], model.history.history['binary_accuracy'],
                                    model.history.history['precision'], model.history.history['recall'])):
        logger.info(f"Epoch {idx}: loss: {epoch[0]}, binary_accuracy {epoch[1]}, precision {epoch[2]}, "
                    f"recall {epoch[3]}")

    return model


def compute_best_threshold(model: Sequential, evaluation_data: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Computes the threshold that returns the best F1-score.

    :param model: the trained RNN
    :param evaluation_data: bytes and function starts as returned by load_data
    :return: the optimal threshold for the given model and evaluation data
    """

    step_width = 1

    prediction = model.predict(evaluation_data[0])
    threshold_values = {}
    for i in range(0, 100, step_width):
        threshold = i / 100
        threshold_values[threshold] = evaluate(prediction, threshold, evaluation_data)

    return max(threshold_values, key=lambda k: threshold_values[k][2])


def evaluate(prediction: np.ndarray, threshold: float, evaluation_data: Tuple[np.ndarray, np.ndarray]) \
        -> Tuple[float, float, float, int, int, int, int]:
    """
    Evaluates the given data against the prediction of the given model.

    :param prediction: array that holds the probabilities per byte
    :param threshold: threshold to decide if a byte starts a function
    :param evaluation_data: bytes and function starts as returned by load_data
    :return: precision, recall, f1
    """

    # use the given threshold to decide which bytes start a function
    original_shape = prediction.shape
    prediction = np.where(prediction > threshold, 1, 0).reshape(original_shape[0], original_shape[1])
    ground_truth = evaluation_data[1]

    amount_positives = np.count_nonzero(ground_truth)
    amount_negatives = ground_truth.shape[0] * ground_truth.shape[1] - amount_positives

    false_negatives = np.where((ground_truth - prediction) == 1)[0].shape[0]
    true_positives = amount_positives - false_negatives
    false_positives = np.count_nonzero(prediction) - true_positives
    true_negatives = amount_negatives - false_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, true_positives, true_negatives, false_positives, false_negatives


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

    # check if the training file exists
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
