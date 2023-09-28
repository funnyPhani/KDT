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

Language      | ISO 639-1 Code | BBC subdomain(s) | Train | Dev | Test | Total | Link
--------------|----------------|------------------|-------|-----|------|-------|-----
Bengali | bn | https://www.bbc.com/bengali | 8102 | 1012 | 1012 | 10126 | [Download](https://docs.google.com/uc?export=download&id=1h3GY8Pk1xV3DWo3Ewc9ZJQ4bU7tCS_1R)
English | en | https://www.bbc.com/english, https://www.bbc.com/sinhala `*` | 306522 | 11535 | 11535 | 329592 | [Download](https://docs.google.com/uc?export=download&id=1KlTW4WTHzDdmigZnqBLdRTCkamISortQ)
Gujarati | gu | https://www.bbc.com/gujarati | 9119 | 1139 | 1139 | 11397 | [Download](https://docs.google.com/uc?export=download&id=1IJdTIR_Im2Saa_F2tW5UNnU2g_dWp1wG)
Hindi | hi | https://www.bbc.com/hindi | 70778 | 8847 | 8847 | 88472 | [Download](https://docs.google.com/uc?export=download&id=1H3PxMwEFyzNxGXpM0KMPOkt4UcdHbiky)
Marathi | mr | https://www.bbc.com/marathi | 10903 | 1362 | 1362 | 13627 | [Download](https://docs.google.com/uc?export=download&id=1WJNQ5PqqM4FPq7VSWezx1-OFOlZ9dUiU)
Tamil | ta | https://www.bbc.com/tamil | 16222 | 2027 | 2027 | 20276 | [Download](https://docs.google.com/uc?export=download&id=1ukjkPZktUBvckWliCSotUZYXBalZ3t7h)
Telugu | te | https://www.bbc.com/telugu | 10421 | 1302 | 1302 | 13025 | [Download](https://docs.google.com/uc?export=download&id=1cTbqTwYPu5U09U1mBVIN3b71W4B17gOl)
Urdu | ur | https://www.bbc.com/urdu | 67665 | 8458 | 8458 | 84581 | [Download](https://docs.google.com/uc?export=download&id=1Vie5jfHyHBkkW6jLbFNU5qjStcHstOKn)

## Acknowledgement

This work is a joint study with the support of Peking University and Tencent Inc.
