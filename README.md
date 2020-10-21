# Language-Adversarial Training for Cross-Lingual Text Classification

This repo contains the source code for our TACL journal paper:

[**Adversarial Deep Averaging Networks for Cross-Lingual Sentiment Classification**](https://arxiv.org/abs/1606.01614)
<br>
[Xilun Chen](http://www.cs.cornell.edu/~xlchen/),
Yu Sun,
[Ben Athiwaratkun](http://www.benathiwaratkun.com/),
[Claire Cardie](http://www.cs.cornell.edu/home/cardie/),
[Kilian Weinberger](http://kilian.cs.cornell.edu/)
<br>
Transactions of the Association for Computational Linguistics (TACL)
<br>
[paper (arXiv)](https://arxiv.org/abs/1606.01614),
[bibtex (arXiv)](http://www.cs.cornell.edu/~xlchen/resources/bibtex/adan.bib),
[paper (TACL)](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00039),
[bibtex](http://www.cs.cornell.edu/~xlchen/resources/bibtex/adan_tacl.bib),
[talk@EMNLP2018](https://vimeo.com/306129914)

## Introduction
<img src="http://www.cs.cornell.edu/~xlchen/assets/images/adan.png" width="320">

<p>ADAN transfers the knowledge learned from labeled data on a resource-rich source language to low-resource languages where only unlabeled data exists.
It achieves cross-lingual model transfer via learning language-invariant features extracted by Language-Adversarial Training.</p>

## Requirements
- Python 3.6
- PyTorch 0.4
- PyTorchNet (for confusion matrix)
- scipy
- tqdm (for progress bar)

## File Structure

```
.
├── README.md
└── code
    ├── data_prep                       (data processing scripts)
    │   ├── chn_hotel_dataset.py        (processing the Chinese Hotel Review dataset)
    │   └── yelp_dataset.py             (processing the English Yelp Review dataset)
    ├── layers.py                       (lower-level helper modules)
    ├── models.py                       (higher-level modules)
    ├── options.py                      (hyper-parameters aka. all the knobs you may want to turn)
    ├── train.py                        (main file to train the model)
    ├── utils.py                        (helper functions)
    └── vocab.py                        (vocabulary)
```

## Dataset

The datasets can be downloaded separately [here](https://drive.google.com/drive/folders/1_JSr_VBVQ33hS0PuFjg68d3ePBr_eISF?usp=sharing).

To support new datasets, simply write a new script under ```data_prep``` similar to the current ones and update ```train.py``` to correctly load it.

## Run Experiments

```bash
python train.py --model_save_file {path_to_save_the_model}
```

By default, the code uses CNN as the feature extractor. 
To use the LSTM (with dot attention) feature extractor:

```bash
python train.py --model lstm --F_layers 2 --model_save_file {path_to_save_the_model}
```
