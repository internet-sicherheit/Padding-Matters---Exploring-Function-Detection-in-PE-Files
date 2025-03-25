# Code

This directory contains the Python code necessary to prepare, train and use the RNN.

## Pipeline
There are four key steps to reproduce the experiments from our paper:

1. [Prepare Training Data](#prepare-training-data)
2. [Train a Model](#train-a-model)
3. [Predict Function Starts](#predict-function-starts)
4. [Evaluate Prediction Results](#evaluate-prediction-results)

An extended example on how to use the pipeline to reproduce the results in our paper can be found [here](../reproduction.md). 

### Prepare Training Data
In order to train an RNN model, you need to first prepare the training data.
Therefore, you need Portable Executable (PE) binaries and the corresponding ground truth data.

The ground truth data must be a file that contains the virtual addresses of the function starts.
Each function has its own line that inlcudes the virtual address of the function starts and optionally, separated by a space, the virtual address of the function end.
An example of a ground truth file can be found [here](../examples/example-gt-file).

```commandline
$ ./preparation/prepare_training_data.py -h
usage: prepare_training_data.py [-h] {single,batch,merge} ...

Extracts RNN training data from PE files and corresponding ground truth files.

positional arguments:
  {single,batch,merge}

optional arguments:
  -h, --help            show this help message and exit
```

To prepare the training data for a single PE and ground truth combination, you can use the following code:
```commandline
$ ./preparation/prepare_training_data.py single -h
usage: prepare_training_data.py single [-h] PE GROUND_TRUTH OUTPUT

Extracts the training data for a single PE and ground truth file.

positional arguments:
  PE            Path to the PE file.
  GROUND_TRUTH  Path to the file that contains the ground truth data. The file must contain one line per function with the virtual address of the function start and optionally the virtual address of the end of the function separated by
                a comma.
  OUTPUT        Path to the pickle file that should hold the training data.

optional arguments:
  -h, --help    show this help message and exit
  
 $ ./preparation/prepare_training_data.py single ../examples/example-pe-file ../examples/example-gt-file example-training-data.pkl
```
The script produces a pickle file that can be used to train the RNN.

To prepare the training data for multiple PEs and ground truth files, you can use the `batch` mode of the [`prepare_training_data.py`](preparation/prepare_training_data.py) script.
```commandline
$ ./preparation/prepare_training_data.py batch -h
usage: prepare_training_data.py batch [-h] PE_DIRECTORY GT_DIRECTORY OUTPUT_DIRECTORY

Extracts the training data for multiple files. The PE files and the ground truth data lie in different directories. Corresponding PE files and ground truth files must have the same file name. Each PE and ground truth file combination
results in an pickle output file that is stored in the output directory.

positional arguments:
  PE_DIRECTORY      Directory that contains the PE files.
  GT_DIRECTORY      Directory that contains the ground truth files.
  OUTPUT_DIRECTORY  Directory that will contain the pickle output files.

optional arguments:
  -h, --help        show this help message and exit
```

To merge multiple pickle files into a single pickle file, you can use the `merge` mode of the [`prepare_training_data.py`](preparation/prepare_training_data.py) script.
```commandline
$ ./preparation/prepare_training_data.py merge -h
usage: prepare_training_data.py merge [-h] INPUT_DIR OUTPUT_FILE

Merges a set of pickle training data files into a single pickle file.

positional arguments:
  INPUT_DIR    Directory that contains the pickle training data files.
  OUTPUT_FILE  Path to the file that should contain the resulting pickle training file.

optional arguments:
  -h, --help   show this help message and exit
```

### Train a Model
Given a pickle training data file, you can train an RNN model.
We provide a script to train both RNN architectures described in our paper.
Both scripts share the same calling conventions.
In the examples, we will only refer to the one output neuron architecture.

```commandline
$ ./one_output_neuron/train_rnn.py -h
2023-03-13 15:50:22.128804: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-03-13 15:50:22.128885: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
usage: train_rnn.py [-h] TRAINING_DATA MODEL

Reads a pickle file with training data, uses this data to train a RNN and stores the resulting model.

positional arguments:
  TRAINING_DATA  Path to the pickle file that contains the training data.
  MODEL          Path to the file that should store the resulting model.

optional arguments:
  -h, --help     show this help message and exit
```
```commandline
$ ./one_output_neuron/train_rnn.py ../examples/example-training-data-file.pkl example_model.h5
2023-03-13 15:51:03.958918: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-03-13 15:51:03.959022: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-03-13 15:51:08.055846: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2023-03-13 15:51:08.055925: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-03-13 15:51:08.055985: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fedora): /proc/driver/nvidia/version does not exist
2023-03-13 15:51:08.057793: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/150
1/1 [==============================] - 10s 10s/step - loss: 0.0343 - binary_accuracy: 0.9945 - precision: 0.0000e+00 - recall: 0.0000e+00 - tp: 0.0000e+00 - fp: 0.0000e+00 - fn: 210.0000 - tn: 37790.0000
...
Epoch 150/150
1/1 [==============================] - 1s 575ms/step - loss: 0.0101 - binary_accuracy: 0.9964 - precision: 0.9412 - recall: 0.3810 - tp: 80.0000 - fp: 5.0000 - fn: 130.0000 - tn: 37785.0000
```

The script produces a H5 file that contains the model architecture and trained weights.
Please note that the training file used in the example above is only for testing the functionality of the script and should not be used to train an actual model. 

### Predict Function Starts
Using a trained model, you can utilize the RNN to predict function starts.
We provide a script to predict function starts for both RNN architectures described in our paper.
Both scripts share the same calling conventions.
In the examples, we will only refer to the one output neuron architecture.

```commandline
$ ./one_output_neuron/predict_function_starts_pe.py -h
2023-03-13 15:58:27.399481: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-03-13 15:58:27.399574: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
usage: predict_function_starts_pe.py [-h] [--output OUTPUT] [--include-probabilities] [--silent] MODEL PE_BINARY

Uses a trained RNN model to extract function starts of a given PE binary.

positional arguments:
  MODEL                 The model that should be used for the extraction.
  PE_BINARY             The PE binary for which the functions starts should be extracted.

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Optional output file where the predicted virtual addresses should be stored.Each line will contain one predicted address.
  --include-probabilities
                        If passed, the output will include the probabilities for each detected function.
  --silent              If passed, the results won't be printed to stdout.
```
```commandline
$ ./one_output_neuron/predict_function_starts_pe.py ../models/one-output-neuron/pe-x86-64.h5 ../examples/example-pe-file --output prediction_results
2023-03-13 16:00:01.373173: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-03-13 16:00:01.373233: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-03-13 16:00:05.189542: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2023-03-13 16:00:05.189584: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-03-13 16:00:05.189625: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fedora): /proc/driver/nvidia/version does not exist
2023-03-13 16:00:05.189934: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
140001014
140001020
...
14000a1ee
14000a210
```

NOTE: For large samples, you may need to adjust the batch size in the prediction script.
Depending on your system, the process might otherwise be killed.
During the prediction, the batch size does not influence the quality of the prediction, only the processing time is influenced.

To change the batch size for the one output neuron architecture, you need to change the value in line 30 in the sript [`predict_function_starts_pe.py`](one_output_neuron/predict_function_starts_pe.py).

To change the batch size for the two output neurons architecture, you need to change the value in line 27 in the sript [`predict_function_starts_pe.py`](two_output_neurons/predict_function_starts_pe.py).

### Evaluate Prediction Results
To evaluate the prediction results against the ground truth, you can use the script [evaluate.py](evaluate.py).

```commandline
$ ./evaluate.py -h
usage: evaluate.py [-h] PREDICTION_RESULTS GROUND_TRUTH

Evaluate the predictions of the RNN against a given ground truth.

positional arguments:
  PREDICTION_RESULTS  Path to the file that contains the prediction results.
  GROUND_TRUTH        Path to the file that contains the ground truth.

optional arguments:
  -h, --help          show this help message and exit
```
```commandline
$ ./evaluate.py prediction_results ../examples/example-gt-file 
TP: 201, FP: 2, FN: 9
Precision: 0.9901477832512315
Recall   : 0.9571428571428572
F1-score : 0.9733656174334142
```