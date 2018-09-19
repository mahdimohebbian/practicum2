# Homework

## Train your real-world baseline system
```
NB: this task should be done in a team. You can divide responsibilities or do the task jointly.
```

Your task is to train Sockeye on all the training data (Europarl corpus) we worked with in class.
Use the same dev set as in class.

Definitely you need to preprocess the training set again.

```
Short inline question:

Do we need to repeat preprocessing for the dev set that was used in class? Explain your answer.

your answer:

----
```

Your model and training `must` follow specs below:
Model params:

```
1. bidirectional LSTM encoder, 1 layer
2. unidirectional LSTM decoder, 1 layer
3. dot product attention type
4. adam optimizer
5. 0.001 learning rate
6. source and target embeddings should have 300 features each
7. encoder and decoder LSTM's should have 600 hidden units each
8. no additional model reinforcements (do not add any additional stuff to the model, left it vanilla (default))
```

Trainig params:

```
1. batches consist of sentences up to 100 words
2.  there should be 64 sentences per batch
3.  early stopping based on perplexity
4.  minimum number of epochs is 3
5.  3 checkpoints per epoch
6.  early stopping if model did not improve for 4 checkpoints
7.  evaluate 3 times per epoch
```

Note: some specs above can be set directly, some need calculations. It is your task to figure out details.

Submit the result config, slurm out file, and pass to your experiment folder on rocket.

```
pass to your experiment
```

## Inline task
```
Try to translate following sentence:

>> tere hommikust !

Write down what you got as a translation and compare to what you got in class with 10k data.

```


Do not forget to use SLURM manager because you will need your HPC account to finish the course.
