# Simple Neural Text Classification(NTC)

This repository contains implementation of naive and simple text classification using recurrent neural network (LSTM) and convolutional neural network (from [[Kim 2014](http://arxiv.org/abs/1408.5882)]). You need to specify architecture to train, and you can select both. If you choose both arthictecture to classify sentences, inference will be done by ensemble (just simple average).

In addition, this repo is for [lecture](https://www.fastcampus.co.kr/data_camp_nlpbasic/) and [book](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/), what I conduct. Please, refer those site for further information.

## Pre-requisite

- Python 3.6 or higher
- PyTorch 1.6 or higher
- PyTorch Ignite
- TorchText 0.5 or higher
- [torch-optimizer 0.0.1a15](https://pypi.org/project/torch-optimizer/)
- Tokenized corpus (e.g. [Moses](https://www.nltk.org/_modules/nltk/tokenize/moses.html), Mecab, [Jieba](https://github.com/fxsjy/jieba))

if you want to use BERT finetuning, you may also need,

- Huggingface

## Usage

### Preparation

#### Format

The input file would have a format with two columns, class and sentence. Those columns are delimited by tab. Class does not need to be a number, but a word (without white-space). Below is example corpus to explain.

```bash
$ cat ./data/raw_corpus.txt | shuf | head
positive	나름 괜찬항요 막 엄청 좋은건 아님 그냥 그럭저럭임... 아직 까지 인생 디퓨져는 못찾은느낌
negative	재질은플라스틱부분이많고요...금방깨질거같아요..당장 물은나오게해야하기에..그냥설치했어요..지금도 조금은후회중.....
positive	평소 신던 신발보다 크긴하지만 운동화라 끈 조절해서 신으려구요 신발 이쁘고 편하네요
positive	두개사서 직장에 구비해두고 먹고있어요 양 많아서 오래쓸듯
positive	생일선물로 샀는데 받으시는 분도 만족하시구 배송도 빨라서 좋았네요
positive	아이가 너무 좋아합니다 크롱도 좋아라하지만 루피를 더..
negative	배송은 기다릴수 있었는데 8개나 주문했는데 샘플을 너무 적게보내주시네요ㅡㅡ;;
positive	너무귀여워요~~ㅎ아직사용은 못해? f지만 이젠 모기땜에 잠설치는일은 ? j겟죠
positive	13개월 아가 제일좋은 간식이네요
positive	지인추천으로 샀어요~ 싸고 가성비 좋다해서 낮기저귀로 써보려구요~
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
302680 ./data/corpus.txt
```

As you can see, we have more than 260k samples in corpus.

```bash
$ cat ./data/corpus.txt | shuf > ./data/corpus.shuf.txt
$ head -n 62680 ./data/corpus.shuf.txt > ./data/corpus.test.txt
$ tail -n 240000 ./data/corpus.shuf.txt > ./data/corpus.train.txt
```

Now, you have 240,000 samples for train-set, and 62,680 samples for valid-set. Note that you can use 'rl' command, instead of 'shuf', if you are using MacOS.

### Train

Below is the example command for training. You can select your own hyper-parameter values via argument inputs.

```bash
python train.py --model_fn ./models/model.pth --train ./data/corpus.train.txt --valid ./data/corpus.valid.txt --rnn --cnn --gpu_id 0
```

Note that you need to specify an architecture for training. You can select both rnn and cnn for ensemble method. Also, you can select the device to use for training. In order to use CPU only, you can put -1 for '--gpu_id' argument, which is default value.

```bash
$ python ./train.py --help
usage: train.py [-h] --model_fn MODEL_FN --train_fn TRAIN_FN [--gpu_id GPU_ID]
                [--verbose VERBOSE] [--min_vocab_freq MIN_VOCAB_FREQ]
                [--max_vocab_size MAX_VOCAB_SIZE] [--batch_size BATCH_SIZE]
                [--n_epochs N_EPOCHS] [--word_vec_size WORD_VEC_SIZE]
                [--dropout DROPOUT] [--max_length MAX_LENGTH] [--rnn]
                [--hidden_size HIDDEN_SIZE] [--n_layers N_LAYERS] [--cnn]
                [--use_batch_norm]
                [--window_sizes [WINDOW_SIZES [WINDOW_SIZES ...]]]
                [--n_filters [N_FILTERS [N_FILTERS ...]]]
```

or you can check default hyper-parameter from train.py.

### Inference

You can feed standard input as input for inference, like as below. Prediction result consists of two columns(top-k classes and input sentence) with tab delimiter. The result will be shown as standard output.

```bash
$ head ./data/review.sorted.uniq.refined.tok.shuf.test.tsv | awk -F'\t' '{ print $2 }' | python classify.py --model ./models/model.pth --gpu_id -1 --top_k 1
positive	생각 보다 밝 아요 ㅎㅎ
negative	쓸 대 가 없 네요
positive	깔 금 해요 . 가벼워 요 . 설치 가 쉬워요 . 타 사이트 에 비해 가격 도 저렴 하 답니다 .
positive	크기 나 두께 가 딱 제 가 원 하 던 사이즈 네요 . 책상 의자 가 너무 딱딱 해서 쿠션 감 좋 은 방석 이 필요 하 던 차 에 좋 은 제품 만났 네요 . 냄새 얘기 하 시 는 분 도 더러 있 던데 별로 냄새 안 나 요 .
positive	빠르 고 괜찬 습니다 .
positive	유통 기한 도 넉넉 하 고 좋 아요
positive	좋 은 가격 에 좋 은 상품 잘 쓰 겠 습니다 .
negative	사이트 에서 늘 생리대 사 서 쓰 는데 오늘 처럼 이렇게 비닐 에 포장 되 어 받 아 본 건 처음 입니다 . 위생 용품 이 고 자체 도 비닐 포장 이 건만 소형 박스 에 라도 넣 어 보내 주 시 지 . ..
negative	연결 부분 이 많이 티 가 납니다 . 재질 구김 도 좀 있 습니다 .
positive	애기 태열 때문 에 구매 해서 잘 쓰 고 있 습니다 .
```

Also, you can see the arguments, and see the default values on classify.py.

```bash
$ python classify.py -h
usage: classify.py [-h] --model_fn MODEL [--gpu_id GPU_ID]
                   [--batch_size BATCH_SIZE] [--top_k TOP_K]
```

## Evaluation

I split the corpus to make train-set and valid-set. 240,000 lines are sampled for train-set and 62,680 samples for valid-set. Architecture snapshots are like as below. You may increase the performance with hyper-parameter optimization.

```bash
RNNClassifier(
  (emb): Embedding(35532, 128)
  (rnn): LSTM(128, 256, num_layers=4, batch_first=True, dropout=0.3, bidirectional=True)
  (generator): Linear(in_features=512, out_features=2, bias=True)
  (activation): LogSoftmax()
)
```

```bash
CNNClassifier(
  (emb): Embedding(35532, 256)
  (feature_extractors): ModuleList(
    (0): Sequential(
      (0): Conv2d(1, 100, kernel_size=(3, 256), stride=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.3, inplace=False)
    )
    (1): Sequential(
      (0): Conv2d(1, 100, kernel_size=(4, 256), stride=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.3, inplace=False)
    )
    (2): Sequential(
      (0): Conv2d(1, 100, kernel_size=(5, 256), stride=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.3, inplace=False)
    )
  )
  (generator): Linear(in_features=300, out_features=2, bias=True)
  (activation): LogSoftmax()
)
```

|Architecture|Test Accuracy|
|-|-|
|Bi-LSTM|0.9035|
|CNN|0.9090|
|Bi-LSTM + CNN|0.9142|
|KcBERT|0.9598|

## Author

|Name|Kim, Ki Hyun|
|-|-|
|email|pointzz.ki@gmail.com|
|github|https://github.com/kh-kim/|
|linkedin|https://www.linkedin.com/in/ki-hyun-kim/|

## Reference

- Kim, Convolutional neural networks for sentence classification, EMNLP, 2014
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- [Lee, KcBERT: Korean comments BERT, GitHub, 2020](https://github.com/Beomi/KcBERT)
