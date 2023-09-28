## Requirements

Software:
```
Python3
Pytorch >= 1.0
argparse == 1.1
```


## Prepare

* Download the ``google_model.bin`` from [here](https://share.weiyun.com/5GuzfVX), and save it to the ``models/`` directory.
* Download the ``CnDbpedia.spo`` from [here](https://share.weiyun.com/5BvtHyO), and save it to the ``brain/kgs/`` directory.
* Optional - Download the datasets for evaluation from [here](https://share.weiyun.com/5Id9PVZ), unzip and place them in the ``datasets/`` directory.

The directory tree of K-BERT:
```
KDT
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   │   ├── CnDbpedia.spo
│   │   ├── HowNet.spo
│   │   └── Medical.spo
│   └── knowgraph.py
├── datasets
│   ├── book_review
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│   ├── chnsenticorp
│   │   ├── dev.tsv
│   │   ├── test.tsv
│   │   └── train.tsv
│    ...
│
├── models
│   ├── google_config.json
│   ├── google_model.bin
│   └── google_vocab.txt
├── outputs
├── uer
├── README.md
├── requirements.txt
├── run_kbert_cls.py
└── run_kbert_ner.py
```


## KDT for text classification

### Classification example

Run example on Book review with CnDbpedia:
```sh
CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_cls.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/medicalQA/train.tsv \
    --dev_path ./datasets/medicalQA/dev.tsv \
    --test_path ./datasets/medicalQA/test.tsv \
    --epochs_num 5 --batch_size 32 --kg_name Symptom \
    --output_model_path ./outputs/kbert_medicalQA_cls_Medical.bin
#    > ./outputs/kbert_medical_ner_Medical.log 2>&1 &
```

Options of ``run_kbert_cls.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--kg_name] - The name of knowledge graph, "HowNet", "CnDbpedia" or "Medical".
        [--output_model_path] - Path to the output model.
```

## Dataset Splitting

| Dataset Name                  | Total Records | Train Set (70%) | Test Set (20%) | Validation Set (10%) |
|-------------------------------|---------------|------------------|----------------|-----------------------|
| Patient Syndrome Description  | 2.6 million   | 1,820,000        | 520,000        | 260,000               |
| Diagnosis Q&A                 | 1.73 million  | 1,211,000        | 346,000        | 173,000               |

### Classification benchmarks

Results of KDT and the baseline transformers on MDQA and MedQuAD (%)

| Models        | MedQuAD      | MDQA          |
| :-----        | :----:       | :----:        |
| BERT-base     | 94.3         | 94.9          |
| BERT-large    | 98.7         | 98.9          |
| ALBERT-base   | 85.3         | 87.4          |
| ALBERT-large  | 88.7         | 90.1          |
| DistilBERT    | 91.3         | 92.4          |
| RoBERTa-base  | 96.4         | 96.9          |
| RoBERTa-large | 97.3         | 97.7          |
| GPT-2         | 95.3         | 95.8          |
| GPT-3         | 99.1         | 99.5          |
| K-BERT        | 99.2         | 99.8          |
| KDT           | 99.4         | 99.9          |




## Acknowledgement

This work is a joint study with the support of Peking University and Tencent Inc.
