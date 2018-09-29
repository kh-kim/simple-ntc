# Simple Neural Text Classification(NTC)

This repository contains implementation of naive and simple text classification using recurrent neural network (LSTM) and convolutional neural network (from [[Kim 2014](http://arxiv.org/abs/1408.5882)]). You need to specify architecture to train, and you can select both. If you choose both arthictecture to classify sentences, inference will be done by ensemble (just simple average).

In addition, this repo is for [lecture](https://www.fastcampus.co.kr/data_camp_nlpbasic/) and [book](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/), what I conduct. Please, refer those site for further information.

## Pre-requisite

- Python 3.6 or higher
- PyTorch 0.4 or higher
- TorchText 0.3 or higher (You may need to install via [github](https://github.com/pytorch/text).)
- Tokenized corpus (e.g. [Moses](https://www.nltk.org/_modules/nltk/tokenize/moses.html), Mecab, [Jieba](https://github.com/fxsjy/jieba))

## Usage

### Preparation

#### Format

The input file would have a format with two columns, class and sentence. Those columns are delimited by tab. Class does not need to be a number, but a word (without white-space). Below is example corpus to explain.

```bash
$ cat ./data/raw_corpus.txt | shuf | head -n 5
cm_mac	맥에 아이폰 연결 후 아이폰 내부자료 파인더에서 폴더로 볼 수 없나요? : 클리앙
cm_ku	넵튠 번들 기념 느와르.jpg : 클리앙
cm_mac	패러랠즈로 디아를 하는건 좀 아니려나요? : 클리앙
cm_ku	덕당분들 폰에서 덕내나요 ㄸㄸㄸㄷ : 클리앙
cm_oversea	혹시 영어 영작에 도움좀 부탁드립니다. : 클리앙
```

#### Tokenization (Optional)

You may need to tokenize sentences in the corpus. You need to select your own tokenizer based on the language. (e.g. Mecab for Korean)

```bash
$ cat ./data/raw_corpus.txt | awk -F'\t' '{ print $2 }' | mecab -O wakati > ./data/tmp.txt
$ cat ./data/raw_corpus.txt | awk -F'\t' '{ print $1 }' > ./data/tmp_class.txt
$ paste ./data/tmp_class.txt ./data/tmp.txt > ./data/corpus.txt
$ rm ./data/tmp.txt ./data/tmp_class.txt
```

#### Shuffle and Split for Train-set and Valid-set

After correct formatting and tokenization, you need to split the corpus to train-set and valid-set.

```bash
$ wc -l ./data/corpus.txt
261664 ./data/corpus.txt
```

As you can see, we have more than 260k samples in corpus.

```bash
$ cat ./data/corpus.txt | shuf > ./data/corpus.shuf.txt
$ head -n 15000 ./data/corpus.shuf.txt > ./data/corpus.valid.txt
$ tail -n 246664 ./data/corpus.shuf.txt > ./data/corpus.train.txt
```

Now, you have 246,664 samples for train-set, and 15,000 samples for valid-set. Note that you can use 'rl' command, instead of 'shuf', if you are using MacOS.

### Train

Below is the example command for training. You can select your own hyper-parameter values via argument inputs.

```bash
python train.py --model ./models/model.pth --train ./data/corpus.train.txt --valid ./data/corpus.valid.txt --rnn --cnn --gpu_id 0
```

Note that you need to specify an architecture for training. You can select both rnn and cnn for ensemble method. Also, you can select the device to use for training. In order to use CPU only, you can put -1 for '--gpu_id' argument, which is default value.

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

You can feed standard input as input for inference, like as below. Prediction result consists of two columns(top-k classes and input sentence) with tab delimiter. The result will be shown as standard output.

```bash
$ head -n 10 ./data/corpus.valid.txt | awk -F'\t' '{ print $2 }' | python classify.py --model ./models/clien.pth --gpu_id -1 --top_k 3
cm_andro cm_iphonien cm_mac	갤 노트 잠금 화면 해제 어 플 사용 하 시 나요 ? : 클리앙
cm_baby cm_car cm_lego	[ 예비 아빠 ] 입당 신고 합니다 : 클리앙
cm_gym cm_oversea cm_vcoin	11 / 07 운동 일지 : 클리앙
cm_ku cm_baby cm_car	커플 이 알콩달콩 하 는 거 보 면 뭐 가 좋 습니까 . utb : 클리앙
cm_iphonien cm_mac cm_car	아이 포니 앙 분 들 께서 는 어떤 사이즈 의 아이 패드 를 더 선호 하 시 나요 ? : 클리앙
cm_coffee cm_lego cm_bike	잉여 잉여 ~ : 클리앙
cm_coffee cm_gym cm_lego	드 뎌 오늘 제대로 된 에스프레소 한잔 마셨 습니다 ! ! ^^ : 클리앙
cm_coffee cm_oversea cm_ku	동네 에 있 는 커피 집 에서 먹 는 커피 빙수 . .. : 클리앙
cm_car cm_oversea cm_bike	땡볕 에 두 시간 세차 하 기 : 클리앙
cm_gym cm_oversea cm_bike	268 . 1 / 22 생 서니 의 말랑말랑 클 핏 일지 ₩ 15 : 클리앙
```

Also, you can see the arguments, and see the default values on classify.py.

```bash
$ python classify.py -h
usage: classify.py [-h] --model MODEL [--gpu_id GPU_ID]
                   [--batch_size BATCH_SIZE] [--top_k TOP_K]
```

## Evaluation

I took an evaluation with my own corpus, which is crawled from [clien](https://www.clien.net/). The task is classify the correct category of the sentence. There are 15 categories, like as below.

|No|Class Name|#Samples|Topic|
|-|-|-|-|
|1|cm_andro|20000|Android development|
|2|cm_baby|15597|Raising baby|
|3|cm_bike|20000|Bike hobby|
|4|cm_car|20000|Car hobby|
|5|cm_coffee|19390|Coffee hobby|
|6|cm_gym|20000|Working out|
|7|cm_havehome|13062|About having(or rent) home|
|8|cm_iphonien|20000|About iPhone|
|9|cm_ku|20000|About anime|
|10|cm_lego|20000|Lego hobby|
|11|cm_mac|20000|About Macintosh|
|12|cm_nas|11206|About NAS(Network Attached Storage)|
|13|cm_oversea|10381|About living in oversea|
|14|cm_stock|12028|About stock trading|
|15|cm_vcoin|20000|About crypto-currency trading|
||Total|261664||

I split the corpus to make train-set and valid-set. 245,000 lines are sampled for train-set and 16,664 samples for valid-set. Architecture snapshots are like as below. You may increase the performance with hyper-parameter optimization.

```bash
RNNClassifier(
  (emb): Embedding(35532, 128)
  (rnn): LSTM(128, 256, num_layers=4, batch_first=True, dropout=0.3, bidirectional=True)
  (generator): Linear(in_features=512, out_features=15, bias=True)
  (activation): LogSoftmax()
)
```

```bash
CNNClassifier(
  (emb): Embedding(35532, 128)
  (cnn-3-100): Conv2d(1, 100, kernel_size=(3, 128), stride=(1, 1))
  (cnn-4-100): Conv2d(1, 100, kernel_size=(4, 128), stride=(1, 1))
  (cnn-5-100): Conv2d(1, 100, kernel_size=(5, 128), stride=(1, 1))
  (relu): ReLU()
  (dropout): Dropout(p=0.3)
  (generator): Linear(in_features=300, out_features=15, bias=True)
  (activation): LogSoftmax()
)
```

Following table shows that the evaluation result of each architecture. The size of validation set is 16,664. You can see that the ensemble is slightly better than others.

|Architecture|Valid Loss|Valid Accuracy|
|-|-|-|
|Bi-LSTM|7.9818e-01|0.7666|
|CNN|8.4225e-01|0.7497|
|Bi-LSTM + CNN||0.7679|

## Author

|Name|Kim, Ki Hyun|
|-|-|
|email|pointzz.ki@gmail.com|
|github|https://github.com/kh-kim/|
|linkedin|https://www.linkedin.com/in/ki-hyun-kim/|

## Reference

- [[Kim 2014](http://arxiv.org/abs/1408.5882)] Yoon Kim. 2014. Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
