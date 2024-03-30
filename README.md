# Federated Learning on TSDiff

This repository contains the implementation of federated learning on [TSDiff](https://github.com/amazon-science/unconditional-time-series-diffusion).

### train:

python bin/train_model.py -c configs/train_tsdiff/train_uber_tlc.yaml

### test:
python bin/guidance_experiment.py -c configs/guidance/guidance_solar.yaml --ckpt ./models/best_checkpoint_14.ckpt

### About GluonTS
https://zh.mxnet.io/blog/gluon-ts-release

### 下一步

在train_model中实现server和clients

### Tutorial: 
Variational Inference | Evidence Lower Bound (ELBO): https://www.youtube.com/watch?v=HxQ94L8n0vU

### Env

- conda activate tsdiff

- pip install --editable "."

CUDA must be 11.8, and install pytorch again.

- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia