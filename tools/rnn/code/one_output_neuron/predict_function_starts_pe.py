#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uses a trained RNN model to extract function starts of a given PE binary.

PLEASE MAKE SURE TO USE A PYTHON VERSION NEWER OR EQUAL TO 3.7 AS THIS SCRIPT RELIES ON DICTS BEING ORDERED
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Union

import pefile
from keras.models import load_model, Sequential
from keras.preprocessing.sequence import pad_sequences

# import the utils.py script from the parent directory
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import create_memory_map, store_prediction_results, print_prediction_results, SLICE_SIZE


# threshold which is used to decide whether a byte starts a function or not
# if the RNN predicts a probability above the threshold, we consider the byte a function start
THRESHOLD = 0.38
# size of the batches that are propagated to the RNN
BATCH_SIZE = 1000


def prepare_section_data(section: Dict[int, Tuple[int, int]]) -> np.ndarray:
    """
    Prepares the given section data to be fed into the RNN.

    :param section: memory mapped representation of the bytes within the section
    :return: numpy array with byte slices that can be fed into the RNN
    """

    # get the byte values
    section_bytes = [byte[0] for byte in section.values()]
    # split the bytes into slices
    section_bytes_slices = [section_bytes[i:i + SLICE_SIZE] for i in range(0, len(section_bytes), SLICE_SIZE)]
    # add padding to the slices and convert them to numpy arrays
    prepared_data = pad_sequences(section_bytes_slices, maxlen=SLICE_SIZE, dtype='int32', padding='post', value=0)
    # cancel out zero values
    prepared_data += 1

    return prepared_data


def mark_function_starts(section: Dict[int, Tuple[int, int]], function_starts: np.ndarray):
    """
    Marks the function starts in the memory representation of the section.

    :param section: memory mapped representation of the bytes within the section
    :param function_starts: numpy array that contains the predicted function starts
    """

    # concatenate all slices
    function_starts = np.concatenate(function_starts)
    # collect the indices of all bytes that where predicted to be a function start
    function_start_indices = np.where(function_starts > THRESHOLD)[0]
    # mark all predicted function starts in the section
    section_offset = list(section)[0]
    for function_start_index in function_start_indices:
        current_index = section_offset + function_start_index
        section[current_index] = (section[current_index][0], 1)


def map_probability(section: Dict[int, Tuple[int, int]], probability: np.ndarray) -> List[Tuple[int, float]]:
    """
    Takes the class probability for each byte index in the given sections and maps it to the corresponding bytes.

    :param section: memory mapped representation of the bytes within the section
    :param probability: an array of slices that hold the probability of a function starts per byte
    :return: list of tuples that hold the VA and the class probability
    """

    mapped_probabilities = []

    virtual_addresses = iter(list(section))
    for byte_slice in probability:
        for probability in byte_slice:
            function_start = probability[0]
            # We do not need very precise probabilities. Therefore, we can round the probability.
            function_start = round(float(function_start), 5)
            # We may have more probabilities than we have bytes in the section due to padding. Therefore, we can
            # break once we stored the probabilities for each address.
            try:
                current_va = virtual_addresses.__next__()
            except StopIteration:
                break
            mapped_probabilities.append((current_va, function_start))

    return mapped_probabilities


def predict_function_starts(pe: pefile.PE, model: Sequential, include_probabilities: bool = False) \
        -> Union[List[Tuple[int, int]], Tuple[List[Tuple[int, int]], List[Tuple[int, float]]]]:
    """
    Predicts the function starts in the given binary using the given model.

    :param pe: the loaded PE file
    :param model: the model that should be used for the prediction
    :param include_probabilities: if true, also returns the probability for each byte
    :return: a list of functions starts with their virtual address and file offset (and optionally a list with the
    probability per address)
    """
    
    # create a memory mapped representation for all sections of the PE file
    memory_map = create_memory_map(pe, include_all_sections=True)
    # iterate over all sections and extract the function starts
    function_starts_va = []
    probabilities = []
    for section in memory_map:
        # verify that the section contains at least one byte
        if not section:
            continue
        prepared_data = prepare_section_data(section)
        prediction = model.predict(prepared_data, batch_size=BATCH_SIZE)
        if include_probabilities:
            probabilities += map_probability(section, prediction)
        mark_function_starts(section, prediction)
        function_starts_va += [key for key in section.keys() if section[key][1] == 1]
    # collect the file offsets for the function starts
    function_starts_fo = []
    for function_start in function_starts_va:
        function_starts_fo.append(pe.get_offset_from_rva(function_start - pe.OPTIONAL_HEADER.ImageBase))

    # combine virtual addresses and file offsets
    extracted_function_starts = list(zip(function_starts_va, function_starts_fo))

    if include_probabilities:
        return extracted_function_starts, probabilities
    return extracted_function_starts


def main():
    parser = argparse.ArgumentParser(description="Uses a trained RNN model to extract function starts of a given "
                                                 "PE binary.")
    parser.add_argument('model', metavar='MODEL',
                        help="The model that should be used for the extraction.")
    parser.add_argument('binary',
                        metavar='PE_BINARY',
                        help="The PE binary for which the functions starts should be extracted.")
    parser.add_argument('--output',
                        help="Optional output file where the predicted virtual addresses should be stored."
                             "Each line will contain one predicted address.")
    parser.add_argument('--include-probabilities',
                        dest='include_probabilities',
                        default=False,
                        action='store_true',
                        help="If passed, the output will include the probabilities for each detected function.")
    parser.add_argument('--silent',
                        dest='silent',
                        default=False,
                        action='store_true',
                        help="If passed, the results won't be printed to stdout.")

    args = parser.parse_args()

    # check if the binary exists and is a valid file
    binary_path = Path(args.binary)
    if not binary_path.exists():
        raise FileNotFoundError(f"The given PE binary {args.binary} was not found.")

    # check if the model exists, if a model is given
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"The given model {args.model} was not found.")
    # load the model
    model = load_model(model_path)

    # check if the binary is a valid PE
    try:
        pe = pefile.PE(binary_path)
    except pefile.PEFormatError:
        raise ValueError(f"Unable to load the given PE file {binary_path}. Make sure the file is a valid PE file.")

    # check if an output file is given and if it exists, if it is a directory
    store_output = False
    if args.output:
        store_output = True
        output_file = Path(args.output)
        if output_file.exists() and output_file.is_dir():
            raise ValueError(f"The given output file {args.output} is a directory."
                             f"The output file must not be a directory")
        output_file.parent.mkdir(exist_ok=True, parents=True)

    if args.include_probabilities:
        function_starts, probabilities = predict_function_starts(pe, model, include_probabilities=True)
        if not args.silent:
            print_prediction_results(function_starts, probabilities)
    else:
        function_starts = predict_function_starts(pe, model)
        if not args.silent:
            print_prediction_results(function_starts)

    # store the predicted function starts, if an output file was given
    if store_output:
        store_prediction_results(function_starts, output_file)


if __name__ == '__main__':
    main()
