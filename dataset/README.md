# Datasets

This directory contains the datasets (BAP/ByteWeight and `FuncPEval`) that were used in our paper.

Each dataset is given with the following structure:
```
├── binaries
│   ├── random-padding
│   │   ├── x86
│   │   └── x86-64
│   └── normal-padding
│       ├── x86
│       └── x86-64
└── ground-truth
    ├── x86
    └── x86-64
```

The directory `binaries` contains the binary samples.
Each sample is provided for 32-bit (`x86`) and 64-bit (`x86-64`).
We provide all samples in a random-padding variant as well, in which the padding between functions is replaced by random bytes.

The directory `ground-truth` contains the ground truth files for the samples.

## [BAP/ByteWeight](byteweight)
The BAP/ByteWeight dataset that was used in the paper [BYTEWEIGHT: Learning to Recognize Functions in Binary Code](https://www.usenix.org/conference/usenixsecurity14/technical-sessions/presentation/bao).


## FuncPEval

The FuncPEval dataset has been introduced in our paper `Padding Matters – Exploring Function Detection in PE Files`. It consists of the Chromium and Conti samples described below.

### [Chromium v109](chromium)
A release of Chromium v109 published [here (x86)](http://commondatastorage.googleapis.com/chromium-browser-snapshots/index.html?prefix=Win/1069956/) (`chrome.dll` has a SHA256 hash value of `1ce8b9551709581688a8199a0e0fcb48cfcac7fadf3671622ea8e66fbe39151f`) and [here (x86-64)](http://commondatastorage.googleapis.com/chromium-browser-snapshots/index.html?prefix=Win_x64/1069922/) (`chrome.dll` has a SHA256 hash value of `55f05fe24ebdf8eb263f75e88c8a71a42fb6240b59340a9abf9671ffe79a4f4a`).
Due to size limitation, the samples have to be downloaded separately.
To download the dataset, run the script [`download_chromium_dataset.sh`](download_chromium_dataset.sh).

### [Conti v3](conti)
A leaked version of the Conti ransomware. The resources can be found [here](https://github.com/vxunderground/MalwareSourceCode/blob/main/Win32/Ransomware/Win32.Conti.c.7z).
Microsoft Visual Studio 2022 has been used to build the binaries.
As the dataset contains malware, we only provide a password-protected ZIP file containing the binaries (password `infected`).
To unpack the binaries, run the script [`unzip_conti_dataset.sh`](unzip_conti_dataset.sh).
Keep in mind that this might trigger antivirus software, especially on Windows.
