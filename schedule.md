
Hello Aditya,

I learned about your work : SiloFuse, using Encoder to transform tabular data into latent space, much like LDM does, very interesting!

The coursework we're doing is about how to do Federated learning on Diffusion Model based Time Series Prediction. we're planning to implement HFL first, which isn't that hard. But implementing VFL seems to get a bit more complicated. I thought of your approach and plan to embedding the time series information using an encoder as well.

But there is a problem with this: your task is to do generate, but our task also needs to do predicate and forecast (generate a sequence using the original data). That is to say, there is no way for LDM on Server to directly use latent feature to do a forecast, it needs data that is almost the same as the real data.

So I came up with a new structure: each Client has an encoder which is responsible for embedding; Server has a decoder and an LDM, the decoder is responsible for generating data that is similar to the original data, and the LDM is the original DDIM.

At the very beginning, I first remove the LDM and train only multiple encoders and one decoder to get pretrained weights; after that, I freeze the encoders and decoder and train only the LDM.

By doing this, DDIM on Server has the ability to do forecasts and also has all the advantages of using encoders as you mentioned in your paper.

I'd be interested to know what you think about this idea. I haven't finished reading that paper of yours yet, I'll take some time this week to read it and would very much like to discuss it with you!

Best regard,
Chenrui Fan

Hi Chenrui,

Glad to see your interest in our paper : ) I have a couple of questions regarding your problem statement and design: 

1) What is the exact scenario? That is, are there multiple clients holding features? If so, is the objective to forecast one time series feature or all of them simultaneously? 

There are multiple clients holding features, and the objective is to forecast all time series feature simultaneously.

2) Who is the party that “owns" the feature to be forecasted? This will heavily influence whether the server is the appropriate party to hold the decoder. Data privacy reasons.

Clients own those vertical split dataset, and yes, server does not have rights to get access to real data.


3) SiloFuse deals with tabular data, i.e., there is no sequential correlation between samples. With time series data, the auto encoders need to preserve this sequential correlation as well because time is an important factor. What are your ideas regarding this challenge?

These are the first things that come to mind and I’d like to hear your thoughts on this. We can schedule a meeting online to discuss further, let me know.

Best regards,
Aditya

我有两个问题：
1. 如果decoder生成的数据跟原数据只是相似，能不能保证privacy？
2. 如果不能使用类似于原数据的sequence，我不知道直接使用encoder得到的latent feature能不能做forecast任务（目前正在代码实现中）
3. Synthetic data只是DDPM生成的结果吧？它的column跟原数据相比应该更少，因为encoder做了压缩。论文里说“ In this regard, the database community is looking towards using synthetic data as an alternative to real data for protecting privacy”，指的就是这个合成的数据？我说这个是因为silofuse的fig. 1中Synthetic data也有三列：G R S，那么是不是我将autoencoder从VAE换成DDPM就行？