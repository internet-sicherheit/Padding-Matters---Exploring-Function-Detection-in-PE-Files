# Models

This directory contains trained models that where used in the experiments described in the paper.
We provide models for both architectures of the RNN, i.e. the original architecture with two output neurons and the modified architecture with one output neuron.


We provide the models in the following structure:
```
├── cross-validation
│   ├── x86
│   └── x86-64
├── pe-x86-64.h5
└── pe-x86.h5
```

For both RNN architectures, we provide one model for 32-bit (`pe-x86.h5`) and one model for 64-bit (`pe-x86-64.h5`) binaries.
Additionally, we provide ten models for each architecture in the subdirectory `cross-validation` that were used for the cross validation.
The suffix in the filename describes which fold was excluded during the training and used to evaluate the model.

For the RNN with one output neuron, we further provide two models in the `random-padding` subdiretory.
These models were trained with binaries in which the padding between functions was replaced by random bytes.