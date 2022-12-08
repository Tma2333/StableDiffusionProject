import os
import uuid
from pathlib import Path

import fire
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DataParallelStrategy
from pytorch_lightning.callbacks import (ModelCheckpoint, 
                                         EarlyStopping, 
                                         LearningRateMonitor)

from utils import read_yaml
from lightning_task import Classification

def train(cfg):
    cfg = read_yaml(cfg)
    seed = cfg['seed']
    seed_everything(seed=seed, workers=True)


    trainer_cfg = cfg['training']['trainer']

    # get task
    task = Classification(cfg)
    
    # GPU
    gpus = trainer_cfg.get('gpus', 1)
    strategy = trainer_cfg.get('strategy', None)
    unused_params = trainer_cfg.get('unused_params', False)

    num_gpu = gpus if isinstance(gpus, int) else len(gpus)
    if gpus == -1 or num_gpu > 1:
        if strategy == 'dp':
            strategy = DataParallelStrategy()
        if strategy == 'ddp':
            strategy = DDPStrategy(find_unused_parameters=unused_params)
    else:
        strategy = None
    
    # Logging
    save_dir = trainer_cfg.get('save_dir', '/deep2/u/yma42/files/results')
    exp_name = cfg.get('exp_name', f'exp_{str(uuid.uuid1())[:8]}') 
    version = cfg.get('version', 0)
    wandb_entity = cfg.get('w&b_entity', 'cs229-stable-diffusion')
    wandb_project = cfg.get('w&b_project', 'cs230')

    logger = WandbLogger(name=f'{exp_name}_v{version}', save_dir=save_dir,
                         project=wandb_project, entity=wandb_entity)

    # Callbacks - Checkpointing and early stop
    save_top_k = trainer_cfg.get('save_top_k', 5)
    monitor_metric = trainer_cfg.get('monitor_metric', 'Eval_Loss')
    monitor_mode = trainer_cfg.get('monitor_mode', 'min')
    patience = trainer_cfg.get('patience', 10)

    ckpt_dir = Path(save_dir)/ exp_name/ f'version_{version}'/ 'ckpt'
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=save_top_k, 
                              verbose=True, monitor=monitor_metric, 
                              mode=monitor_mode, every_n_epochs=1)
    earlystop_cb = EarlyStopping(monitor=monitor_metric, 
                                 patience=patience, 
                                 verbose=True, mode=monitor_mode)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Trainer config
    gradient_clip_val = trainer_cfg.get('gradient_clip_val', 0)
    limit_train_batches = trainer_cfg.get('limit_train_batches', 1.0)
    enable_model_summary = trainer_cfg.get('enable_model_summary', False)
    max_epochs = trainer_cfg.get('max_epochs', 100)

    trainer = Trainer(accelerator="gpu",
                      devices=gpus,
                      strategy=strategy,
                      logger=logger,
                      callbacks=[ckpt_cb, earlystop_cb, lr_monitor],
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      enable_model_summary = enable_model_summary,
                      max_epochs=max_epochs,
                      log_every_n_steps=5)
    trainer.fit(task)


if __name__ == "__main__":
    fire.Fire()
