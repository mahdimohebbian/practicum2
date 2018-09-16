# WMT Estonian to English news translation on GPU

In this tutorial we will train a German to English Sockeye model on a dataset from the
[Conference on Machine Translation (WMT) 2017](http://www.statmt.org/wmt17/).

## Setup

### Setup environment on rocket

1) Login to the rocket:

```bash
ssh username@rocket.hpc.ut.ee
```

1) Import the module and create local conda env:

```bash
module load python/3.6.3/CUDA-8.0
```

```bash
conda create -n mtenv-cuda8 python=3.6
```

Activate env:

```bash
source activate mtenv-cuda8
```

2) Install sockeye on GPU:

```bash
mkdir mt2018
cd mt2018
wget https://raw.githubusercontent.com/awslabs/sockeye/master/requirements/requirements.gpu-cu80.txt
pip install sockeye --no-deps -r requirements.gpu-cu80.txt
rm requirements.gpu-cu80.txt
```



## Data preprocessing
Sockeye expects preprocessed data as the input.

Convenient way of doing preprocessing is using moses-scripts:
```bash
git clone https://github.com/marian-nmt/moses-scripts
```

Preprocessing usually consists of following steps:
1. Tokenization
2. Truecasing
3. Cleaning
4. Subword segmentation

hw:



Let's first fetch the data:

Training set:
```bash
mkdir data
cd data
wget http://www.statmt.org/europarl/v7/et-en.tgz
    tar -xvzf et-en.tgz
```

Dev set:
```bash
download it from the github assigment's repo
```

Lets now take a small subset of training set for the purpose of this lab:
```bash
wc -l wc -l europarl-v7.et-en.e*
head -n 10000 europarl-v7.et-en.en > train.en
head -n 10000 europarl-v7.et-en.et > train.et
```

1. Tokenization

```bash
../moses-scripts/scripts/tokenizer/tokenizer.perl -h

../moses-scripts/scripts/tokenizer/tokenizer.perl < train.en > tok.train.en
../moses-scripts/scripts/tokenizer/tokenizer.perl < train.et > tok.train.et
```

Lets compare 2 files now:

```bash
head -2 train.en tok.train.en
```

Describe what tokenization does based on what you see.

2. Truecasing

First, train the truecasing model:

```bash
../moses-scripts/scripts/recaser/train-truecaser.perl --model en-truecase.mdl --corpus tok.train.en
../moses-scripts/scripts/recaser/train-truecaser.perl --model et-truecase.mdl --corpus tok.train.et
```

Next, do the truecasing itself:

```bash
../moses-scripts/scripts/recaser/truecase.perl --model en-truecase.mdl < tok.train.en > tc.tok.train.en
../moses-scripts/scripts/recaser/truecase.perl --model et-truecase.mdl < tok.train.et > tc.tok.train.et
```

Compare:

```bash
head tok.train.en tc.tok.train.en
```

Describe what truecasing does based on what you see.

3. Cleaning

It removes strange sentences, and sentences that are not in range 1-100 tokens

```bash
../moses-scripts/scripts/training/clean-corpus-n.perl tc.tok.train en et cleaned.tc.tok.train 1 100
```

4. Subwords segmentation

In addition to tokenization we will split words into subwords using Byte Pair Encoding (BPE).
In order to do so we use a tool called [subword-nmt](https://github.com/rsennrich/subword-nmt).
Run the following commands to set up the tool:

```bash
git clone https://github.com/rsennrich/subword-nmt.git
export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
```

First we need to build our BPE vocabulary:
```bash
python -m learn_joint_bpe_and_vocab --input cleaned.tc.tok.train.et cleaned.tc.tok.train.en \
                                    -s 3000 \
                                    -o bpe.codes \
                                    --write-vocabulary bpe.vocab.et bpe.vocab.en

```

Note, that we used 3000 as desired number of result subwords, however, in real setting the recommended
number is about 10 as big (concretely, you should use 32000 in your homework)

This will create a joint source and target BPE vocabulary.
Next, we use apply the Byte Pair Encoding to our training and development data:

```bash
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.en --vocabulary-threshold 50 < cleaned.tc.tok.train.en > bpe.cleaned.tc.tok.train.en

python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.et --vocabulary-threshold 50 < cleaned.tc.tok.train.et > bpe.cleaned.tc.tok.train.et
```

``` bash
head cleaned.tc.tok.train.en bpe.cleaned.tc.tok.train.en
```

Describe what it does based on what you see.


## Development sets

Now, preprocess the dev sets by yourself.

NB: you must not train truecasing and BPE models for dev sets.
You should use the one you trained on train set.
Just apply the truecasing and BPE with already trained models.

## Training

Having preprocessed our data we can start training.
Note that Sockeye will load all training data into memory in order to be able to easily reshuffle after every epoch.

Before we start training we will prepare the training data by splitting it into shards and serializing it in matrix format:
```bash
python -m sockeye.prepare_data \
                        -s bpe.cleaned.tc.tok.train.en \
                        -t bpe.cleaned.tc.tok.train.et \
                        -o train_data
```
While this is an optional step it has the advantage of considerably lowering the time needed before training starts and also limiting the memory usage as only one shard is loaded into memory at a time.


We can now kick off the training process.
```bash
cd ..
mkdir experiments
```

Create following file sbatch file:

```bash
#!/bin/bash

#The name of the job is test_job
#SBATCH -J mt_tut

#The job requires 1 compute node
#SBATCH -N 1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The maximum walltime of the job is a 8 days
#SBATCH -t 00:05:00

#SBATCH --mem=5G

#Leave this here if you need a GPU for your job
#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

# OUR COMMANDS GO HERE

module load python/3.6.3/CUDA-8.0

source activate mtenv-cuda8


python -m sockeye.train --disable-device-locking \
                        --device-ids 0 \
                        -s data/bpe.cleaned.tc.tok.train.et \
                        -t data/bpe.cleaned.tc.tok.train.en \
                        -vs data/bpe.cleaned.tc.tok.dev.et \
                        -vt data/bpe.cleaned.tc.tok.dev.en \
                        -o experiments/wmt_model_tutorial_et2en \
                        --encoder rnn \
                        --decoder rnn \
                        --num-embed 256 \
                        --rnn-num-hidden 512 \
                        --rnn-attention-type dot \
                        --max-seq-len 60 \
                        --decode-and-evaluate 0 \
                        --batch-type sentence \
                        --batch-size 200 \
                        --max-num-epochs 10 \
                        --checkpoint-frequency 25
```

And send it for training
:
```bash
sbatch train_tut.sbatch
```

To examine the training process, run:

```bash
tail -f -n 50 slurm-xxxxxx.out
```

This will train a 1-layer bi-LSTM encoder, 1-layer LSTM decoder with dot attention.
Sockeye offers a whole variety of different options regarding the model architecture,
such as stacked RNNs with residual connections (`--num-layers`, `--rnn-residual-connections`),
[Transformer](https://arxiv.org/abs/1706.03762) encoder and decoder (`--encoder transformer`, `--decoder transformer`),
[ConvS2S](https://arxiv.org/pdf/1705.03122) (`--encoder cnn`, `--decoder cnn`),
various RNN (`--rnn-cell-type`) and attention (`--attention-type`) types and more.

There are also several parameters controlling training itself.
Unless you specify a different optimizer (`--optimizer`) [Adam](https://arxiv.org/abs/1412.6980) will be used.
Additionally, you can control the batch size (`--batch-size`), the learning rate schedule (`--learning-rate-schedule`)
and other parameters relevant for training.

Training will run until the validation perplexity stops improving.
Sockeye starts a decoder in a separate process at every checkpoint running on the same device as training in order to evaluate metrics such as BLEU.
Note that these scores are calculated on the tokens provided to Sockeye, e.g. in this tutorial BLEU will be calculated on the sub-words we created above.
As an alternative to validation perplexity based early stopping you can stop early based on BLEU scores (`--optimized-metric bleu`).

To make sure the decoder finishes before the next checkpoint one can subsample the validation set for
BLEU score calculation.
For example `--decode-and-evaluate 500` will decode and evaluate BLEU on a random subset of 500 sentences.
We sample the random subset once and keep it the same during training and also across trainings by
fixing the random seed.
Therefore, validation BLEU scores across training runs are comparable.
Perplexity will not be affected by this and still be calculated on the full validation set.

Training a model on this data set is going to take a while.
In the next section we discuss how you can monitor the training progress.

## Inline task
```
Try to translate following sentence:

>> tere hommikust !

Write down what you got as a translation:

```

## Summary

Congratulations! You have successfully preprocessed the data and trained your first "real" Sockeye translation model.


## Credits:

https://github.com/awslabs/sockeye/tree/master/tutorials/wmt