import os
import yaml
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
from models import *
from experiment import VAEXperiment
from dataset import VAEDataset
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


def main(config):
    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                name=config['model_params']['name'],)

    # For reproducibility
    torch.manual_seed(config['exp_params']['manual_seed'])  

    model = models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                            config['exp_params'])

    data = VAEDataset(**config["data_params"])

    data.setup()
    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2, 
                                        dirpath=os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor="val_loss",
                                        save_last=True),
                    ],
                    plugins=[SLURMEnvironment(auto_requeue=config['environment_params']['SLURM'])],
                    accelerator=config['environment_params']['accelerator'],
                    **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic runner for models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
