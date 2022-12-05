# PBJE-ECPE

This repo provides the source code of our paper: Pair-Based Joint Encoding with Relational Graph Convolutional Networks for Emotion-Cause Pair Extraction (EMNLP 2022)

## How to use it
### Step 0: Prepare your environments
Reference environment settings:
```
python             3.8.5
pytorch            1.9.0+cu11.1
transformers       4.9.2
dgl                0.8.2+cuda11.1
tqdm
scikit-learn       0.24.2
scipy              1.5.2
```

### Step 1: Download the Bert-base-chinese
Please download the Bert-base-chinese from [here](https://huggingface.co/bert-base-chinese). And put the files to
```
./src/bert-base-chinese/
```

### Step 2: Train the model
Please change the directory to
```
./src/
```
and run the following command
```
python main.py
```
If you want to use the dataset of 20-fold, please run the following command
```
python main.py --split split20
```
