#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reads a PE file and the corresponding function boundaries and writes a pickle file that can be used to train the RNN.

The input file containing the function boundaries as virtual addresses must have the following format:
<first_function_start_va>
<second_function_start_va>
...

The following format is also allowed:
<first_function_start_va> <first_function_end_va>
<second_function_start_va> <second_function_end_va>
...

The virtual addresses must be given as a hex value without any prefix, e.g. 140001000.

The pickle output file contains a python tuple with the following structure:
([byte_sequence_1,...,byte_sequence_n], [ground_truth_1,...,ground_truth_n])
Each byte sequence contains y values between 0 and 255 representing one byte.
Each ground truth list contains y values between 0 and 1 with a 1 representing a function start at this position.

This script does only extract data from sections that are executable.
"""

import pickle

import pefile
import argparse
from typing import List, Tuple
from pathlib import Path

# import the utils.py script from the parent directory
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import load_ground_truth, create_memory_map, mark_function_starts, memory_map_to_rnn_struct, \
    store_training_data


def extract_training_data(pe: pefile.PE, ground_truth: Path) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extracts the byte values from the PE's executable sections, maps them with the ground truth data, and returns them
    in a format that can be directly used to train the RNN.
    :param pe: the PE file
    :param ground_truth: path to the file that contains the ground truth data
    :return: a tuple of byte and ground truth slices
    """

    # get a memory mapped representation of the executable sections and mark the function starts
    memory_map = create_memory_map(pe)
    ground_truth = load_ground_truth(ground_truth)
    mark_function_starts(memory_map, ground_truth)

    # convert the memory map into data structures that are required by the RNN
    return memory_map_to_rnn_struct(memory_map)


def extract_and_store(pe_file: Path, gt_file: Path, output_file: Path):
    """
    Extracts and stores the training data for the given PE and ground truth file combination.

    :param pe_file: path to the PE file
    :param gt_file: path to the ground truth file
    :param output_file: path to the output file
    """

    try:
        pe = pefile.PE(pe_file)
    except pefile.PEFormatError:
        raise ValueError(f"The given file {pe_file} is not a PE file.")

    training_data = extract_training_data(pe, gt_file)
    store_training_data(training_data, output_file)


def single_extraction(args: argparse.Namespace):
    """
    Performs the training data extraction in single mode, i.e. for one PE and ground truth file combination.
    :param args: arguments passed in the command
    """

    # check if the files exist
    pe_path = Path(args.pe)
    if not pe_path.exists():
        raise FileNotFoundError(f"The given file {args.pe} was not found.")

    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        raise FileNotFoundError(f"The given file {args.ground_truth} was not found.")

    # create the parent directories for the output file if they do not exist
    output_file = Path(args.output_file)
    if output_file.is_dir():
        raise ValueError(f"The outputfile {output_file} is a directory.")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extract_and_store(pe_path, gt_path, output_file)


def batch_extraction(args: argparse.Namespace):
    """
    Performs the training data extraction for batch mode, i.e. a directory of PEs and a directory of ground truth files.
    :param args: arguments passed in the command
    """

    # check if the files exist and are valid directories
    pe_directory = Path(args.pe_directory)
    if not pe_directory.exists():
        raise FileNotFoundError(f"The given directory {args.pe_directory} does not exist.")
    if not pe_directory.is_dir():
        raise ValueError(f"{args.pe_directory} must be a directory that contains the PE files.")

    gt_directory = Path(args.gt_directory)
    if not gt_directory.exists():
        raise FileNotFoundError(f"The given directory {args.gt_directory} does not exist.")
    if not gt_directory.is_dir():
        raise ValueError(f"{args.gt_directory} must be a directory.")

    # create the output directory if it does not exist
    output_directory = Path(args.output_directory)
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"{args.output_directory} must be a directory.")
    output_directory.mkdir(parents=True, exist_ok=True)

    # collect all PEs and ground truth files
    pe_files = [pe_file for pe_file in pe_directory.iterdir()]
    gt_files = [gt_file for gt_file in gt_directory.iterdir()]

    # map paths of the PEs and ground truth files to their filenames
    pe_files_map = {pe_file.name: pe_file for pe_file in pe_files}
    gt_files_map = {gt_file.name: gt_file for gt_file in gt_files}
    gt_file_names = list(gt_files_map)

    # extract the training data for each valid mapping
    for pe_file_name in pe_files_map.keys():
        # check if their exist a ground truth file for the current PE
        if pe_file_name in gt_file_names:
            print(f"Extracting data for {pe_file_name}")
            # compute path for the output file
            output_file = output_directory / f"{pe_file_name}.pkl"
            extract_and_store(pe_files_map[pe_file_name], gt_files_map[pe_file_name], output_file)


def merge(args: argparse.Namespace):
    """
    Merges several pickle files containing training data into a single pickle file that can be used for training.
    :param args: arguments passed in the command
    """

    # check if the input directory exists and is a valid directory
    input_directory = Path(args.input_dir)
    if not input_directory.exists():
        raise FileNotFoundError(f"The given directory {args.input_dir} does not exist.")
    if not input_directory.is_dir():
        raise ValueError(f"{args.input_dir} must be a directory.")

    output_file = Path(args.output_file)
    if output_file.exists() and output_file.is_dir():
        raise ValueError(f"{args.output_file} is a directory. The output file must not be a directory.")

    resulting_training_data = ([], [])

    # iterate over all files in the input directory and collect the training data from the pickle files
    for file in input_directory.iterdir():
        print(f"Processing input file {file.name}")
        try:
            with open(file, 'rb') as pickle_data:
                data = pickle.load(pickle_data)
                resulting_training_data = (resulting_training_data[0] + data[0], resulting_training_data[1] + data[1])
        except pickle.UnpicklingError:
            continue

    store_training_data(resulting_training_data, output_file)


def main():
    parser = argparse.ArgumentParser(description="Extracts RNN training data from PE files and corresponding ground"
                                                 " truth files.")
    subparsers = parser.add_subparsers()

    # add parser for single extraction mode
    single_parser = subparsers.add_parser('single',
                                          description="Extracts the training data for a single PE and ground truth "
                                                      "file.")
    single_parser.add_argument('pe',
                               metavar='PE',
                               help="Path to the PE file.")
    single_parser.add_argument('ground_truth',
                               metavar='GROUND_TRUTH',
                               help="Path to the file that contains the ground truth data. The file must contain one "
                                    "line per function with the virtual address of the function start and optionally "
                                    "the virtual address of the end of the function separated by a comma.")
    single_parser.add_argument('output_file',
                               metavar='OUTPUT',
                               help="Path to the pickle file that should hold the training data.")
    single_parser.set_defaults(func=single_extraction)

    # add parser for batch extraction mode
    batch_parser = subparsers.add_parser('batch',
                                         description="Extracts the training data for multiple files. "
                                                     "The PE files and the ground truth data lie in different "
                                                     "directories. "
                                                     "Corresponding PE files and ground truth files must have "
                                                     "the same file name. "
                                                     "Each PE and ground truth file combination results in an pickle "
                                                     "output file that is stored in the output directory.")
    batch_parser.add_argument('pe_directory',
                              metavar='PE_DIRECTORY',
                              help="Directory that contains the PE files.")
    batch_parser.add_argument('gt_directory',
                              metavar='GT_DIRECTORY',
                              help="Directory that contains the ground truth files.")
    batch_parser.add_argument('output_directory',
                              metavar='OUTPUT_DIRECTORY',
                              help="Directory that will contain the pickle output files.")
    batch_parser.set_defaults(func=batch_extraction)

    # add parser to merge several pickle files that contain training data
    merge_parser = subparsers.add_parser('merge',
                                         description="Merges a set of pickle training data files into a single pickle "
                                                     "file.")
    merge_parser.add_argument('input_dir',
                              metavar='INPUT_DIR',
                              help="Directory that contains the pickle training data files.")
    merge_parser.add_argument('output_file',
                              metavar='OUTPUT_FILE',
                              help="Path to the file that should contain the resulting pickle training file.")
    merge_parser.set_defaults(func=merge)

    # parse arguments and call the corresponding function
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
