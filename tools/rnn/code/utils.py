#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the RNN.
"""


import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Union

import pefile

# the maximum size of each slice in the training dataset
SLICE_SIZE = 1000


def store_training_data(training_data: Tuple[List[List[int]], List[List[int]]], output_path: Path):
    """
    Stores the given training data in a pickle file.

    :param training_data: the training data
    :param output_path: path of the pickle output file
    """

    with open(output_path, 'wb') as output_file:
        pickle.dump(training_data, output_file)


def memory_map_to_rnn_struct(memory_map: List[Dict[int, Tuple[int, int]]]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Converts the memory map into data structures that are required by the RNN.

    :param memory_map: the memory map that should be converted
    :returns: two lists, one contains the byte values, one the function start markings
    """

    byte_values_slices = []
    function_starts_slices = []
    for mm in memory_map:
        # 1. extract the values and store them in a list
        bytes_with_function_start = list(mm.values())
        # if the section does not contain any functions starts, skip
        if not bytes_with_function_start:
            continue
        # 2. separate the byte values and the function start markings
        byte_values, function_starts = zip(*bytes_with_function_start)
        # 3. convert the tuples into lists
        byte_values = list(byte_values)
        function_starts = list(function_starts)
        # 4. split the lists into slices
        byte_values_slices += [byte_values[i:i + SLICE_SIZE] for i in range(0, len(byte_values), SLICE_SIZE)]
        function_starts_slices += [function_starts[i:i + SLICE_SIZE] for i in
                                   range(0, len(function_starts), SLICE_SIZE)]

    return byte_values_slices, function_starts_slices


def mark_function_starts(memory_map: List[Dict[int, Tuple[int, int]]], ground_truth: List[int]) -> None:
    """
    Reads the ground truth data and marks the function starts in the memory map.

    :param memory_map: a list of dictionaries with a memory mapped representation of all relevant sections
    :param ground_truth: a list containing function start addresses
    """

    for function_start in ground_truth:
        for section in memory_map:
            if function_start in section.keys():
                section[function_start] = (section[function_start][0], 1)


def load_ground_truth(gt: Path) -> List[int]:
    """
    Loads the function start addresses from the ground truth file.
    Does not check if the ground truth file exists.

    :param gt: the file that contains the ground truth information
    :return: a list containing the virtual addresses of the function starts
    """
    gt_list = []

    with open(gt, 'r') as gt_data:
        for line in gt_data.readlines():
            function_start = int(line.split()[0], base=16)
            gt_list.append(function_start)

    return gt_list


def store_prediction_results(prediction: List[Tuple[int, int]], output: Path) -> None:
    """
    Stores the virtual addresses of a prediction in a file.

    :param prediction: a list containing the virtual addresses and file offsets of predicted function starts
    :param output: the file that should contain the results
    """
    output = Path(output)
    with open(output, 'w') as output_file:
        output_file.writelines([f"{address[0]:x}\n" for address in prediction])


def print_prediction_results(prediction_results: List[Tuple[int, int]],
                             probabilities: Union[List[Tuple[int, float, float]], List[Tuple[int, float]]] = None)\
        -> None:
    """
    Prints the prediction results.

    :param prediction_results: a list containing the virtual addresses and file offsets of predicted function starts
    :param probabilities: a list containing the class probabilities, either for one or two classes depending on the RNN
    architecture
    """
    if probabilities:
        # if probabilities contains tuples of length 3, the two output neurons architecture was used for the prediction
        if len(probabilities[0]) == 3:
            # convert the probabilities into a dict
            probabilities = {entry[0]: (entry[1], entry[2]) for entry in probabilities}
            output_lines = [f"{prediction[0]:<15x}; "
                            f"is_fs:{probabilities[prediction[0]][1]:8f}, "
                            f"is_no_fs:{probabilities[prediction[0]][0]:8f}"
                            for prediction in prediction_results]
        # if probabilities contains tuples of length 2, the one output neuron architecture was used for the prediction
        else:
            # convert the probabilities into a dict
            probabilities = {entry[0]: entry[1] for entry in probabilities}
            output_lines = [f"{prediction[0]:<15x}; "
                            f"is_fs:{probabilities[prediction[0]]:8f}"
                            for prediction in prediction_results]
    else:
        output_lines = [f"{prediction[0]:x}" for prediction in prediction_results]

    # print the results
    for line in output_lines:
        print(line)


def create_memory_map(pe: pefile.PE, include_all_sections: bool = False) -> List[Dict[int, Tuple[int, int]]]:
    """
    Creates a memory representation of all relevant sections in the PE file.
    Each section is represented by a dictionary.
    The key of the dictionary represents a virtual address (VA). The VA is the sum of the ImageBase and the
    RVA.
    The value of the dictionary is a tuple that holds the value of the byte at the VA and a 0 or 1 whether the
    byte marks a function start. This functions only puts a 0 into the function start field. You need to use another
    function to mark the function starts.

    Example that shows the content within the first executable section at address 0x140001000:
    dict[0][0x140001000] = (0x55, 0)

    :param pe: the loaded PE file
    :param include_all_sections: include all sections instead of executable sections only
    :return: a list of dictionaries with a memory mapped representation of all relevant sections
    """
    memory_map = []
    pe_image_base = pe.OPTIONAL_HEADER.ImageBase

    # get the relevant section
    if include_all_sections:
        sections = pe.sections
    else:
        sections = get_executable_sections(pe)

    # create a memory mapped representation per section
    for i in range(len(sections)):
        section = sections[i]
        # get all bytes within the section
        section_bytes = section.get_data()
        # compute the virtual address of the beginning of the section
        section_va = pe_image_base + section.get_VirtualAddress_adj()
        # add a memory mapped representation for this section and add the values
        memory_map.append({})
        for j in range(len(section_bytes)):
            memory_map[i][section_va + j] = (section_bytes[j], 0)

    return memory_map


def get_executable_sections(pe: pefile.PE) -> List[pefile.SectionStructure]:
    """
    Returns all sections of the given PE that are marked as executable.

    :param pe: the loaded PE file
    :return: list of all executable sections
    """

    return [section for section in pe.sections if section.IMAGE_SCN_MEM_EXECUTE]


def compute_p_r_f(amount_tp: int, amount_fp: int, amount_fn: int) -> Tuple[float, float, float]:
    """
    Compute precision, recall and F1-Measure for the given values.

    :param amount_tp: the amount of true positives
    :param amount_fp: the amount of false positives
    :param amount_fn: the amount of false negatives
    """

    try:
        precision = amount_tp / (amount_tp + amount_fp)
        recall = amount_tp / (amount_tp + amount_fn)
        f_measure = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        raise ZeroDivisionError(f'TP: {amount_tp}, FP: {amount_fp}, FN: {amount_fn}')
    return precision, recall, f_measure
