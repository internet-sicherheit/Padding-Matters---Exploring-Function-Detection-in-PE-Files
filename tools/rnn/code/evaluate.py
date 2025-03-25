#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the predictions of the RNN against a given ground truth.
"""

import argparse
from pathlib import Path

from utils import load_ground_truth


def main():
    parser = argparse.ArgumentParser(description="Evaluate the predictions of the RNN against a given ground truth.")

    parser.add_argument('prediction',
                        metavar='PREDICTION_RESULTS',
                        help="Path to the file that contains the prediction results.")
    parser.add_argument('gt',
                        metavar='GROUND_TRUTH',
                        help="Path to the file that contains the ground truth.")

    args = parser.parse_args()

    # check if the file with the prediction results exists
    prediction_path = Path(args.prediction)
    if not prediction_path.exists():
        raise FileNotFoundError(f"The given file {args.prediction} does not exist.")

    # check if the ground truth file exists
    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise FileNotFoundError(f"The given file {args.gt} does not exist.")

    # extract the function starts from the prediction results
    predicted_function_starts = load_ground_truth(prediction_path)
    predicted_function_starts = set(predicted_function_starts)
    # extract the function starts from the ground truth file
    gt_function_starts = load_ground_truth(gt_path)
    gt_function_starts = set(gt_function_starts)

    # compute true positives, false positives, and false negatives
    true_positives = predicted_function_starts & gt_function_starts
    false_positives = predicted_function_starts - true_positives
    false_negatives = gt_function_starts - true_positives

    # TODO: add error handling for divide by zero
    # check if true positives exists, otherwise no computation can be performed
    if len(true_positives) == 0:
        raise ZeroDivisionError("There exists no true positives. Therefore, the results cannot be computed.")

    # compute precision, recall, f1-score
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    f1_score = 2 * precision * recall / (precision + recall)

    # print the results
    print(f"TP: {len(true_positives)}, FP: {len(false_positives)}, FN: {len(false_negatives)}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1-score : {f1_score}")


if __name__ == '__main__':
    main()
