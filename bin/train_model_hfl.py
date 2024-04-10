from train_model import evaluate_guidance, create_model
from gluonts.dataset.common import ListDataset, Map, TrainDatasets

import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.model.callback import EvaluateCallback
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

# TrainDatasets(
# metadata=MetaData(freq='H', target=None, feat_static_cat=[CategoricalFeatureInfo(name='feat_static_cat_0', cardinality='262')], feat_static_real=[], feat_dynamic_real=[], feat_dynamic_cat=[], prediction_length=24), 
# train=Map(fn=<gluonts.dataset.common.ProcessDataEntry object at 0x7f6f08675850>, iterable=JsonLinesFile(path=PosixPath('/home/fcr/.mxnet/gluon-ts/datasets/uber_tlc_hourly/train/data.json.gz'), start=0, n=None)), 
# test=Map(fn=<gluonts.dataset.common.ProcessDataEntry object at 0x7f6f08675c40>, iterable=JsonLinesFile(path=PosixPath('/home/fcr/.mxnet/gluon-ts/datasets/uber_tlc_hourly/test/data.json.gz'), start=0, n=None))
# )

# dataset.train(
# {
# 'start': Period('2015-02-22 13:00', 'H'), 
# 'target': array([1., 0., 0., ..., 0., 0., 0.], dtype=float32), 
# 'feat_static_cat': array([1], dtype=int32), 
# 'item_id': 1
# }
# )

# VFL module

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_data_part(iterable, part, total_parts):
    for i, item in enumerate(iterable):
        if i % total_parts == part:
            yield item


def split_dataset(dataset, nr_clients = 3):
    dataset_splits = []

    for part in range(nr_clients):
        train_part = Map(fn=lambda x: x, iterable=generate_data_part(dataset.train, part, nr_clients))
        test_part = Map(fn=lambda x: x, iterable=generate_data_part(dataset.test, part, nr_clients))
        split_dataset = TrainDatasets(metadata=dataset.metadata, train=train_part, test=test_part)
        dataset_splits.append(split_dataset)
    
    return dataset_splits


def fed_avg(models):
    """
    计算模型权重的平均值。

    :param models: 客户端模型的列表
    :return: 平均后的模型权重
    """
    global_weights = None
    nr_clients = len(models)

    # 遍历所有模型，累加权重
    for model in models:
        model_weights = model.get_weights()  # 获取模型权重
        if global_weights is None:
            global_weights = model_weights
        else:
            for i in range(len(global_weights)):
                global_weights[i] += model_weights[i]
    
    # 将累加的权重除以客户端数量，得到平均权重
    for i in range(len(global_weights)):
        global_weights[i] = global_weights[i] / nr_clients

    return global_weights
        

def main(config, log_dir, nr_clients = 3):
    # Load parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    total_length = context_length + prediction_length

    # Model Initialize
    clients_model = []
    for i in range(nr_clients):
        clients_model.append(create_model(config))
    server_model = create_model(config)

    # Data split
    dataset = get_gts_dataset(dataset_name)
    dataset_splits = split_dataset(dataset, nr_clients)
    # for dataset_split in dataset_splits:
    #     # print(dataset_split)
    #     for i, entry in enumerate(dataset_split.train):
    #         if i%30==0:
    #             print(i, entry['target'], len(entry['target']), entry['feat_static_cat'], entry['start'], entry['item_id'])

    for i in range(nr_clients):
        train(log_dir, clients_model[i], dataset_splits[i], dataset_name, freq, prediction_length, total_length)
        print(clients_model[i])
        return

    # 在所有客户端模型上应用 FedAvg
    server_model_weights = fed_avg(clients_model)

    # 更新服务器模型的权重
    server_model.set_weights(server_model_weights)
    print(server_model)
    return
    

def train(log_dir, model, dataset, dataset_name, freq, prediction_length, total_length):
    assert dataset.metadata.freq == freq
    assert dataset.metadata.prediction_length == prediction_length

    if config["setup"] == "forecasting":
        training_data = dataset.train
    elif config["setup"] == "missing_values":
        missing_values_splitter = OffsetSplitter(offset=-total_length)
        training_data, _ = missing_values_splitter.split(dataset.train)

    num_rolling_evals = int(len(list(dataset.test)) / len(list(dataset.train)))

    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=config["prediction_length"],
    )

    training_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="train",
    )

    for i, entry in enumerate(dataset.train):
        print(entry['target'], len(entry['target']), entry['feat_static_cat'], entry['start'], entry['item_id'])
        if i >= 10:
            break

    callbacks = []
    if config["use_validation_set"]:
        transformed_data = transformation.apply(training_data, is_train=True)
        train_val_splitter = OffsetSplitter(
            offset=-config["prediction_length"] * num_rolling_evals
        )
        _, val_gen = train_val_splitter.split(training_data)
        val_data = val_gen.generate_instances(
            config["prediction_length"], num_rolling_evals
        )

        callbacks = [
            EvaluateCallback(
                context_length=config["context_length"],
                prediction_length=config["prediction_length"],
                sampler=config["sampler"],
                sampler_kwargs=config["sampler_params"],
                num_samples=config["num_samples"],
                model=model,
                transformation=transformation,
                test_dataset=dataset.test,
                val_dataset=val_data,
                eval_every=config["eval_every"],
            )
        ]
    else:
        transformed_data = transformation.apply(training_data, is_train=True)

    log_monitor = "train_loss"
    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)
    callbacks.append(RichProgressBar())

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[int(config["device"].split(":")[-1])],
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
    )
    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(model, train_dataloaders=data_loader)
    logger.info("Training completed.")

    best_ckpt_path = Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"

    # if not best_ckpt_path.exists():
    #     torch.save(
    #         torch.load(checkpoint_callback.best_model_path)["state_dict"],
    #         best_ckpt_path,
    #     )
    logger.info(f"Loading {best_ckpt_path}.")
    # best_state_dict = torch.load(best_ckpt_path)
    # model.load_state_dict(best_state_dict, strict=True)

    metrics = (
        evaluate_guidance(config, model, dataset.test, transformation, logger = logger)
        if config.get("do_final_eval", True)
        else "Final eval not performed"
    )
    with open(Path(trainer.logger.log_dir) / "results.yaml", "w") as fp:
        yaml.dump(
            {
                "config": config,
                "version": trainer.logger.version,
                "metrics": metrics,
            },
            fp,
        )


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Path to results dir"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config, log_dir=args.out_dir, nr_clients = 3)