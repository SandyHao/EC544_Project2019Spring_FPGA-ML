Group work for ec544 project

The basic CNN is used to do dichotomy

We also try to explore model from https://pytorch.org/docs/stable/torchvision/models.html

These are models that have and complete layers and pretrained weights.

we can pruning layer by layer or use a general judgement to cut weights near zero

the second method is more useful when dealing with DNN or other complicated networks

char-rnn: used as basic text generation model

https://github.com/karpathy/char-rnn

Steps of text generation
1. load data
2. store characters/words in map
3. process data
4. build model and train
5.generate new sentences
