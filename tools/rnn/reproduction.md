# Reproduction

This file contains a description of the necessary steps to reproduce essential experiments in our paper.

Please make sure to fulfill the requirements described [here](README.md#requirements).

## Train a Model
In this section we describe the necessary steps to train a model for a 32-bit one output neuron RNN.
We will use the [BAP/ByteWeight dataset](../../dataset/byteweight/) to train a model.

### Prepare Training Data
Prepare training data in batch mode.
```commandline
$ ./code/preparation/prepare_training_data.py batch ../../dataset/byteweight/binaries/normal-padding/x86/ ../../dataset/byteweight/ground-truth/x86 training_data
Extracting data for msvs_whatever_32_O1_7z
Extracting data for msvs_whatever_32_O1_Client7z
Extracting data for msvs_whatever_32_O1_SfxSetup
Extracting data for msvs_whatever_32_O1_gvim
Extracting data for msvs_whatever_32_O1_hidapi
...
Extracting data for msvs_whatever_32_Ox_readmsg
Extracting data for msvs_whatever_32_Ox_smtpsend
Extracting data for msvs_whatever_32_Ox_vim
```

Now, the directory training_data contains several pickle files.
```commandline
$ ls training_data/
msvs_whatever_32_O1_7z.pkl         msvs_whatever_32_O1_puttytel.pkl   msvs_whatever_32_O2_plink.pkl     msvs_whatever_32_Od_gvim.pkl       msvs_whatever_32_Od_smtpsend.pkl   msvs_whatever_32_Ox_puttygen.pkl
msvs_whatever_32_O1_Client7z.pkl   msvs_whatever_32_O1_readmsg.pkl    msvs_whatever_32_O2_pscp.pkl      msvs_whatever_32_Od_hidapi.pkl     msvs_whatever_32_Od_vim.pkl        msvs_whatever_32_Ox_putty.pkl
msvs_whatever_32_O1_gvim.pkl       msvs_whatever_32_O1_SfxSetup.pkl   msvs_whatever_32_O2_psftp.pkl     msvs_whatever_32_Od_libsodium.pkl  msvs_whatever_32_Ox_7z.pkl         msvs_whatever_32_Ox_puttytel.pkl
msvs_whatever_32_O1_hidapi.pkl     msvs_whatever_32_O1_smtpsend.pkl   msvs_whatever_32_O2_puttygen.pkl  msvs_whatever_32_Od_pageant.pkl    msvs_whatever_32_Ox_Client7z.pkl   msvs_whatever_32_Ox_readmsg.pkl
msvs_whatever_32_O1_libsodium.pkl  msvs_whatever_32_O1_vim.pkl        msvs_whatever_32_O2_putty.pkl     msvs_whatever_32_Od_pbc.pkl        msvs_whatever_32_Ox_gvim.pkl       msvs_whatever_32_Ox_SfxSetup.pkl
msvs_whatever_32_O1_pageant.pkl    msvs_whatever_32_O2_7z.pkl         msvs_whatever_32_O2_puttytel.pkl  msvs_whatever_32_Od_plink.pkl      msvs_whatever_32_Ox_hidapi.pkl     msvs_whatever_32_Ox_smtpsend.pkl
msvs_whatever_32_O1_pbc.pkl        msvs_whatever_32_O2_Client7z.pkl   msvs_whatever_32_O2_readmsg.pkl   msvs_whatever_32_Od_pscp.pkl       msvs_whatever_32_Ox_libsodium.pkl  msvs_whatever_32_Ox_vim.pkl
msvs_whatever_32_O1_plink.pkl      msvs_whatever_32_O2_gvim.pkl       msvs_whatever_32_O2_SfxSetup.pkl  msvs_whatever_32_Od_psftp.pkl      msvs_whatever_32_Ox_pageant.pkl
msvs_whatever_32_O1_pscp.pkl       msvs_whatever_32_O2_hidapi.pkl     msvs_whatever_32_O2_smtpsend.pkl  msvs_whatever_32_Od_puttygen.pkl   msvs_whatever_32_Ox_pbc.pkl
msvs_whatever_32_O1_psftp.pkl      msvs_whatever_32_O2_libsodium.pkl  msvs_whatever_32_O2_vim.pkl       msvs_whatever_32_Od_putty.pkl      msvs_whatever_32_Ox_plink.pkl
msvs_whatever_32_O1_puttygen.pkl   msvs_whatever_32_O2_pageant.pkl    msvs_whatever_32_Od_7z.pkl        msvs_whatever_32_Od_puttytel.pkl   msvs_whatever_32_Ox_pscp.pkl
msvs_whatever_32_O1_putty.pkl      msvs_whatever_32_O2_pbc.pkl        msvs_whatever_32_Od_Client7z.pkl  msvs_whatever_32_Od_readmsg.pkl    msvs_whatever_32_Ox_psftp.pkl
```

In the next step, we merge all the pickle files.
```commandline
$ ./code/preparation/prepare_training_data.py merge training_data/ training_data.pkl
Processing input file msvs_whatever_32_O1_7z.pkl
Processing input file msvs_whatever_32_O1_Client7z.pkl
Processing input file msvs_whatever_32_O1_SfxSetup.pkl
Processing input file msvs_whatever_32_O1_gvim.pkl
Processing input file msvs_whatever_32_O1_hidapi.pkl
...
Processing input file msvs_whatever_32_Ox_readmsg.pkl
Processing input file msvs_whatever_32_Ox_smtpsend.pkl
Processing input file msvs_whatever_32_Ox_vim.pkl
```

### Use the training data to train a model
The previously generated pickle file can now be used to train an RNN model.
```commandline
$ ./code/one_output_neuron/train_rnn.py training_data.pkl pe-x86.h5
2023-03-13 19:34:24.026567: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-03-13 19:34:24.026588: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-03-13 19:34:28.857533: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2023-03-13 19:34:28.857557: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-03-13 19:34:28.857575: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fedora): /proc/driver/nvidia/version does not exist
2023-03-13 19:34:28.857790: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/150
30/30 [==============================] - 25s 753ms/step - loss: 0.0210 - binary_accuracy: 0.9968 - precision: 0.0000e+00 - recall: 0.0000e+00 - tp: 0.0000e+00 - fp: 0.0000e+00 - fn: 92899.0000 - tn: 28970104.0000
Epoch 2/150
30/30 [==============================] - 22s 747ms/step - loss: 0.0193 - binary_accuracy: 0.9968 - precision: 0.0000e+00 - recall: 0.0000e+00 - tp: 0.0000e+00 - fp: 0.0000e+00 - fn: 92899.0000 - tn: 28970100.0000
Epoch 3/150
30/30 [==============================] - 22s 725ms/step - loss: 0.0158 - binary_accuracy: 0.9968 - precision: 0.0000e+00 - recall: 0.0000e+00 - tp: 0.0000e+00 - fp: 0.0000e+00 - fn: 92899.0000 - tn: 28970102.0000
Epoch 4/150
30/30 [==============================] - 22s 725ms/step - loss: 0.0108 - binary_accuracy: 0.9971 - precision: 0.9698 - recall: 0.0935 - tp: 8688.0000 - fp: 271.0000 - fn: 84211.0000 - tn: 28969828.0000
Epoch 5/150
30/30 [==============================] - 22s 723ms/step - loss: 0.0070 - binary_accuracy: 0.9982 - precision: 0.9062 - recall: 0.4998 - tp: 46428.0000 - fp: 4806.0000 - fn: 46471.0000 - tn: 28965292.0000
Epoch 6/150
30/30 [==============================] - 22s 719ms/step - loss: 0.0054 - binary_accuracy: 0.9985 - precision: 0.9170 - recall: 0.5932 - tp: 55110.0000 - fp: 4985.0000 - fn: 37789.0000 - tn: 28965116.0000
Epoch 7/150
30/30 [==============================] - 22s 731ms/step - loss: 0.0044 - binary_accuracy: 0.9988 - precision: 0.9403 - recall: 0.6593 - tp: 61249.0000 - fp: 3889.0000 - fn: 31650.0000 - tn: 28966212.0000
...
Epoch 147/150
30/30 [==============================] - 22s 740ms/step - loss: 8.2282e-04 - binary_accuracy: 0.9997 - precision: 0.9774 - recall: 0.9418 - tp: 87491.0000 - fp: 2019.0000 - fn: 5408.0000 - tn: 28968082.0000
Epoch 148/150
30/30 [==============================] - 22s 739ms/step - loss: 8.3301e-04 - binary_accuracy: 0.9997 - precision: 0.9772 - recall: 0.9423 - tp: 87541.0000 - fp: 2047.0000 - fn: 5358.0000 - tn: 28968056.0000
Epoch 149/150
30/30 [==============================] - 22s 738ms/step - loss: 8.2125e-04 - binary_accuracy: 0.9997 - precision: 0.9765 - recall: 0.9429 - tp: 87593.0000 - fp: 2105.0000 - fn: 5306.0000 - tn: 28967998.0000
Epoch 150/150
30/30 [==============================] - 23s 754ms/step - loss: 8.2345e-04 - binary_accuracy: 0.9997 - precision: 0.9767 - recall: 0.9423 - tp: 87540.0000 - fp: 2087.0000 - fn: 5359.0000 - tn: 28968016.0000
```

## Predict Function Starts
Once a model is trained, it can be used to predict function starts.
In this example, we use a model, trained using the ByteWeight dataset, to predict the function starts in `cryptor.exe`, an executable of the Conti ransomware.

The Conti dataset can be unzipped using the following command in the dataset directory:
```commandline
$ ./unzip_conti_dataset.sh 
Archive:  datasets/conti/binaries.zip
   creating: datasets/conti/binaries/
   creating: datasets/conti/binaries/random-padding/
   creating: datasets/conti/binaries/random-padding/x86/
  inflating: datasets/conti/binaries/random-padding/x86/cryptor.exe  
   creating: datasets/conti/binaries/random-padding/x86-64/
  inflating: datasets/conti/binaries/random-padding/x86-64/cryptor.exe  
   creating: datasets/conti/binaries/normal-padding/
   creating: datasets/conti/binaries/normal-padding/x86/
  inflating: datasets/conti/binaries/normal-padding/x86/cryptor.exe  
   creating: datasets/conti/binaries/normal-padding/x86-64/
  inflating: datasets/conti/binaries/normal-padding/x86-64/cryptor.exe
```

Now, we can predict the function starts in `cryptor.exe`.
```commandline
$ ./code/one_output_neuron/predict_function_starts_pe.py models/one-output-neuron/pe-x86.h5 ../../datasets/conti/binaries/normal-padding/x86/cryptor.exe --output conti_prediction_results
2023-03-13 20:52:20.836423: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2023-03-13 20:52:20.836449: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2023-03-13 20:52:21.950510: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2023-03-13 20:52:21.950535: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2023-03-13 20:52:21.950553: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (fedora): /proc/driver/nvidia/version does not exist
2023-03-13 20:52:21.950718: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
401010
401740
...
42daa0
42db20
42dc90
``` 

NOTE: For large samples, you may need to adjust the batch size in the prediction script.
Depending on your system, the process might otherwise be killed.
During the prediction, the batch size does not influence the quality of the prediction, only the processing time is influenced.
You can find the instructions on how to change the batch size [here](code/README.md#predict-function-starts).

## Evaluate
To evaluate the prediction results against ground truth, we use the script [`evaluate.py`](code/evaluate.py).
```commandline
$ ./code/evaluate.py conti_prediction_results ../../datasets/conti/ground-truth/x86/cryptor.exe 
TP: 640, FP: 19, FN: 82
Precision: 0.9711684370257967
Recall   : 0.8864265927977839
F1-score : 0.9268645908761767
```

You can see that the evaluation results match the results presented in our paper in Figure 6.