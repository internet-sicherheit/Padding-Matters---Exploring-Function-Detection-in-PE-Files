# RNN

This folder contains a reimplementation of [Shin et al.'s bidirectional recurrent neural network](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/shin).

Each directory contains a `READMDE.md` file that provides further information.

### [Code](code)
This directory contains the Python code necessary to prepare, train and use the RNN.

### [Models](models)
This directory contains the trained models that can be used to extract function starts.

### [Examples](examples)
This directory contains example files to test the code.

## Requirements
All experiments were performed using Python version `3.8.2`.
Please make sure to use a Python version between `3.7` and `3.10` as this is required by TensorFlow.
All necessary packages are documented in [`requirements.txt`](requirements.txt).
We recommend using a Python virtual environment.

### Installation of packages
```commandline
$ pip install -r requirements.txt
```

## Reproduction
An extended example on how to use this reimplementation to reproduce the results in our paper can be found in [`reproduction.md`](reproduction.md#reproduction).

The folds used in the cross-validation experiment can be found [here](folds.md).