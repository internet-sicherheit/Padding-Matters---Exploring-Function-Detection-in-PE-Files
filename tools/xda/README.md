# XDA

This folder provides additional training data and models for [XDA: Accurate, Robust Disassembly with Transfer Learning](https://www.ndss-symposium.org/ndss-paper/xda-accurate-robust-disassembly-with-transfer-learning/) based on the [provided artifacts](https://github.com/CUMLSec/XDA).


## Preparation

### Download training data and models

Due to the large data volume, we are providing the new models separately. 
Run the script [download_models.sh](download_models.sh) to download the training data and models.


### Training data

The folder [`data-src`](data-src/) contains the training data used to retrain the function boundary detection part of XDA.
The folder [`data-src/funcbound-retrained`](data-src/funcbound-retrained/) contains the training data for the updated label encoding.
The folder [`data-src/funcbound-retrained-rp`](data-src/funcbound-retrained-rp/) contains the training data for the updated label encoding and the randomized padding bytes.

To select the 10% for the training data, we used the [folds from the cross-validation of the RNN](../rnn/folds.md).
We used fold 1 for the training data and fold 3 for validation.

We used the script [preprocess.sh](https://github.com/CUMLSec/XDA/blob/main/scripts/finetune/preprocess.sh) to preprocess the training data. The preprocessed training data is stored in [`data-bin`](data-bin/).

### Models

For reproduction of the original experiments, we used the pretrained and finetuned models provided by the authors: https://github.com/CUMLSec/XDA/tree/main?tab=readme-ov-file#preparation

Our updated finetuned model containing the new label encoding is stored in [`checkpoints/funcbound-retrained`](checkpoints/funcbound-retrained/) once downloaded as described [here](#download-training-data-and-models).
Our updated finetuned model containing the new label encoding and the random padding is stored in [`checkpoints/funcbound-retrained-rp`](checkpoints/funcbound-retrained-rp/) once downloaded as described [here](#download-training-data-and-models).

## Usage

To utilize the training data and models, please follow the instructions provided by the XDA authors to install XDA: https://github.com/CUMLSec/XDA?tab=readme-ov-file#installation

Once you have a working version of XDA installed, you can place the downloaded models in the appropriate directories and use them.

Based on the original implementation, we created the script [`predict_boundaries.py`](predict_boundaries.py) and [`predict_starts_only.py`](predict_starts_only.py) to predict the function boundaries/starts of x64 PE files.
The scripts are designed to run the prediction for a whole directory of x64 PE files.
Place the scripts in the [`scripts/play`](https://github.com/CUMLSec/XDA/tree/main/scripts/play) folder and adjust the output directory and model path before running them.

> **_Note_**: Please keep in mind that for the new models, the function end describes the last byte belonging to the function, not the first byte that does not belong to the function, as was the case with the original models.