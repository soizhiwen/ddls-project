# Federated Learning on TSDiff

This repository contains the implementation of federated learning on [TSDiff](https://github.com/amazon-science/unconditional-time-series-diffusion).

### dataset 

https://www.kaggle.com/datasets/marquis03/heartbeat-classification/data?select=train.csv

### test:
python bin/guidance_experiment.py -c configs/guidance/guidance_solar.yaml --ckpt ./models/best_checkpoint_14.ckpt