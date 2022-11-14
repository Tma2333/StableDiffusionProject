import os
import logging
import uuid
from pathlib import Path

import fire
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DataParallelStrategy
from pytorch_lightning.callbacks import (ModelCheckpoint, 
                                         EarlyStopping, 
                                         LearningRateMonitor)

from core.serialization import read_yaml
from T5_fine_tune_task import T5FineTuner

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


def train(cfg):
    cfg = read_yaml(cfg)
    trainer_cfg = cfg['training']['trainer']

    # get task
    task = T5FineTuner(cfg)
    
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
    save_dir = trainer_cfg.get('save_dir', '/deep2/group/aicc-bootcamp/self-sup/results/default')
    exp_name = cfg.get('exp_name', f'exp_{str(uuid.uuid1())[:8]}') 
    version = cfg.get('version', 0)
    wandb_entity = cfg.get('w&b_entity', 'cs229-stable-diffusion')
    wandb_project = cfg.get('w&b_project', 'wikohow-t5')

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
                      callbacks=[ckpt_cb, earlystop_cb, lr_monitor, LoggingCallback()],
                      gradient_clip_val=gradient_clip_val,
                      limit_train_batches=limit_train_batches,
                      enable_model_summary = enable_model_summary,
                      max_epochs=max_epochs,
                      val_check_interval=cfg.get('val_check_interval'),
                      log_every_n_steps=5)
    trainer.fit(task)


if __name__ == "__main__":
    fire.Fire()
