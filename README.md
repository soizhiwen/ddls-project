# Federated Learning on TSDiff

This repository contains the implementation of federated learning on [TSDiff](https://github.com/amazon-science/unconditional-time-series-diffusion).

### train:

python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml

### test:
python bin/guidance_experiment.py -c configs/guidance/guidance_solar.yaml --ckpt ./models/best_checkpoint_14.ckpt

### About GluonTS
https://zh.mxnet.io/blog/gluon-ts-release

### Next Step

我了解到你的工作：SiloFuse，利用Encoder将表格数据转化到latent space，很像LDM的做法，非常有趣！
我们正在做的课程作业是关于如何对Diffusion Model based Time Series Prediction做Federated learning。我们打算先实现HFL，这个不是很难。但实现VFL似乎就变得复杂了一点。我想到了你的方法，打算也使用一个encoder对时序信息进行embedding。
但这样做有一个问题：你的任务是做generate，但我们的任务还需要做predicate和forecast（使用原来的数据生成一段序列）。也就是说，Server上的LDM是没有办法直接使用latent feature来做forecast的，它需要跟真实数据差不多的数据。
所以我想到了一个新的结构：每个Client有一个encoder，负责embedding；Server有一个decoder和一个LDM，decoder负责生成和原始数据近似的数据，而LDM就是原本的DDIM。在最开始的时候，我先去掉LDM，只训练多个encoder和一个decoder，得到pretrained weights；之后我冻结encoder和decoder，只训练LDM。
通过这样做，Server上的DDIM就有了做forecast的能力，而且也像你论文中提到那样，具有使用encoder的种种优势。
我很想知道你对于我们这个想法的看法。我还没有看完你的那篇论文，我这周会抽时间看完的，非常希望能和你一起讨论！

 
I learned about your work : SiloFuse, using Encoder to transform tabular data into latent space, much like LDM does, very interesting!
The coursework we're doing is about how to do Federated learning on Diffusion Model based Time Series Prediction. we're planning to implement HFL first, which isn't that hard. But implementing VFL seems to get a bit more complicated. I thought of your approach and plan to embedding the time series information using an encoder as well.
But there is a problem with this: your task is to do generate, but our task also needs to do predicate and forecast (generate a sequence using the original data). That is to say, there is no way for LDM on Server to directly use latent feature to do a forecast, it needs data that is almost the same as the real data.
So I came up with a new structure: each Client has an encoder which is responsible for embedding; Server has a decoder and an LDM, the decoder is responsible for generating data that is similar to the original data, and the LDM is the original DDIM.
At the very beginning, I first remove the LDM and train only multiple encoders and one decoder to get pretrained weights; after that, I freeze the encoders and decoder and train only the LDM.
By doing this, DDIM on Server has the ability to do forecasts and also has all the advantages of using encoders as you mentioned in your paper.
I'd be interested to know what you think about this idea. I haven't finished reading that paper of yours yet, I'll take some time this week to read it and would very much like to discuss it with you!
 
 

### Env

- conda activate tsdiff

- pip install --editable "."

CUDA must be 11.8, and install pytorch again.

- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia