# Simple Nueral Text Classification(NTC)

This repository contains implementation of text classification using recurrent neural network (LSTM) and convolutional neural network (from [[Kim et al.2014](http://arxiv.org/abs/1408.5882)]). You need to specify architecture to train, and you can select both. If you choose both arthictecture to classify sentences, inference will be done by ensemble (just simple average).

In addition, this repo is for [lecture](https://www.fastcampus.co.kr/data_camp_nlpbasic/) and [book](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/), what I conduct. Please, refer those site for further information.

## Pre-requisite

- Python 3.6 or higher
- PyTorch 0.4 or higher
- TorchText 0.3 or higher (You cannot install this version via pip.)
- Tokenized corpus (e.g. [Moses](https://www.nltk.org/_modules/nltk/tokenize/moses.html), Mecab, [Jieba](https://github.com/fxsjy/jieba))

## Usage

### Tokenization
```bash

```

### Shuffle and Split for Train-set and Valid-set

### Train

```bash
$ python train.py -h
usage: train.py [-h] --model MODEL --train TRAIN --valid VALID
                [--gpu_id GPU_ID] [--verbose VERBOSE]
                [--min_vocab_freq MIN_VOCAB_FREQ]
                [--max_vocab_size MAX_VOCAB_SIZE] [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--early_stop EARLY_STOP]
                [--dropout DROPOUT] [--word_vec_dim WORD_VEC_DIM]
                [--hidden_size HIDDEN_SIZE] [--rnn] [--n_layers N_LAYERS]
                [--cnn] [--window_sizes WINDOW_SIZES] [--n_filters N_FILTERS]
```
or you can check default hyper-parameter from train.py.

### Inference

```bash
```

## Evaluation

## Author

|Name|Kim, Ki Hyun|
|-|-|
|email|pointzz.ki@gmail.com|
|github|https://github.com/kh-kim/|
|linkedin|https://www.linkedin.com/in/ki-hyun-kim/|
