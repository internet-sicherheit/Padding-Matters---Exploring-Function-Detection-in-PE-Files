# Examples

This directory contains example files that can be used to test the functionality of the code.

### [Example PE file](example-pe-file)
An example 64-bit PE file.
The PE file can be used with the script [`prepare_training_data.py`](../code/preparation/prepare_training_data.py) to extract training data that can be used to train the RNN.
The PE file can also be used to extract the function starts using the RNN. 

### [Example ground truth file](example-gt-file)
An example ground truth file that contains the function starts and function ends for the [`example-pe-file`](example-pe-file).
This ground truth file can be used with the script [`prepare_training_data.py`](../code/preparation/prepare_training_data.py) to extract training data that can be used to train the RNN.
The ground truth file can also be used with the script [`evaluate.py`](../code/evaluate.py) to evaluate the prediction results of the RNN.

### [Example training data](example-training-data-file.pkl)
A pickle file that contains training data that can be used to train the RNN.

### [Example mode](example-model.h5)
A trained model that can be used to extract function starts.