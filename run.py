import os
import yaml
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path
from models import *
from experiment import FelixExperiment
from dataset import FelixDataset

def main(config):
    torch.manual_seed(config['exp_params']['manual_seed'])

    print(f"=======> Dataset loading...")
    data = FelixDataset(**config["data_params"])
    data.setup()
    print(f"=======> Dataset loaded")

    print(f"=======> Model loading...")
    model = models[config['model_params']['name']](**config['model_params'])
    print(f"=======> Model loaded")





    print(f"=======> Experiment loading...")
    experiment = FelixExperiment(model, config['exp_params'])
    print(f"=======> Experiment loaded")

    print(f"=======> Trainer loading...")
    trainer = pl.Trainer(
        max_epochs=config['trainer_params']['max_epochs'],
        accelerator="cpu",
        # accelerator=config['environment_params']['accelerator']
    )
    print(f"=======> Trainer loaded")

    print(f"=======> Running fit...")
    trainer.fit(experiment, data)
    print(f"=======> Running fit finished")


    # tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
    #                             name=config['model_params']['name'],)

    # For reproducibility 


    # runner = Trainer(
    #     logger=tb_logger,
    #     # callbacks=[TQDMProgressBar(refresh_rate=10)],
    #     plugins=[SLURMEnvironment(auto_requeue=config['environment_params']['SLURM'])],
    #     accelerator=config['environment_params']['accelerator'],
    #     **config['trainer_params']
    # )

    # Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    # Path(f"{tb_logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)

    # print(f"======= Training {config['model_params']['name']} =======")
    # runner.fit(experiment, datamodule=data)

if __name__ == "__main__":
    print(f"=======> Config loading...")
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
    print(f"=======> Config loaded")
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    main(config)
