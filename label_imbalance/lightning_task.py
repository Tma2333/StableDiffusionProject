import numpy as np
import math

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics
import pytorch_lightning as pl
import timm 

from dataset import (CIFAR10FullDataset, 
                     CIFAR10ImbalanceDataset,
                     CIFAR10OverSampleDataset,
                     CIFAR10OverSampleRandomAugDataset,
                     CIFAR10StableDiffusionDataset)



def create_model (cfg): 
    model_cfg = cfg.get("models", {'type': 'resnet18', 'pretrained': False})
    model_type = model_cfg['type']
    pretrained = model_cfg['pretrained']
    model = timm.create_model(model_type, num_classes=10, pretrained=pretrained)
    # modify for CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class Classification(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.model = create_model(self.hparams)

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.eval_acc = torchmetrics.Accuracy()
        self.eval_acc_per_cls = torchmetrics.Accuracy(average='none', num_classes=10)

        self.training_cfg = self.hparams["training"]
        self.batch_size = self.training_cfg["batch_size"]
        gpus = params['training']['trainer']['gpus']
        self.num_gpu = gpus if isinstance(gpus, int) else len(gpus)
        self.epoch_progress = 1


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_nb):
        x, y = batch

        logit = self.forward(x)
        loss = self.criterion(logit, y)
        pred = logit.argmax(1)
        acc = self.train_acc(pred, y)

        self.log('Train/loss', loss, prog_bar=False)
        self.log('Train/acc', acc, prog_bar=True)
        return {'loss': loss,
                'accuracy': acc}

    
    def training_epoch_end(self, outputs):
        self.train_acc.reset()
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
            tag = 'Train/epoch_avg_' + metric_key
            self.log(tag, avg_val, logger=True, sync_dist=True)


    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            x, y = batch
            logit = self.forward(x)
            loss = self.criterion(logit, y)
            pred = logit.argmax(1)
            acc = self.eval_acc(pred, y)
            acc_per_cls = self.eval_acc_per_cls(pred, y)


        self.log('Val/loss', loss, prog_bar=False)
        self.log('Val/acc', acc, prog_bar=True)
        return {'loss': loss,
                'accuracy': acc, 
                'accuracy_per_cls': acc_per_cls}
    

    def validation_epoch_end(self, outputs):
        self.eval_acc.reset()
        metric_keys = list(outputs[0].keys())
        for metric_key in metric_keys:
            if metric_key == 'accuracy_per_cls':
                for cls_idx, cls_name in self.idx_to_cls.items():
                    avg_val = sum(batch[metric_key][cls_idx] for batch in outputs) / len(outputs)
                    tag = f'Val_per_cls/{cls_name}_acc'
                    self.log(tag, avg_val, logger=True, sync_dist=True)
            else:
                avg_val = sum(batch[metric_key] for batch in outputs) / len(outputs)
                tag = 'Val/epoch_avg_' + metric_key
                self.log(tag, avg_val, logger=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer_name = self.training_cfg.get("optimizer", "Adam")
        optimizer_cfg = self.training_cfg.get("optimizer_cfg", {'lr': 0.001})

        if 'lr' not in optimizer_cfg:
            raise KeyError("You must provide learning rate in optimizer cfg")
        
        if optimizer_name == 'Adam':
            optimizer_class = optim.Adam
        elif optimizer_name == "SGD":
            optimizer_class = optim.SGD
        elif optimizer_name == "AdamW":
            optimizer_class = optim.AdamW
        else:
            raise ValueError(f"{optimizer_name} is not supported, add it to configure_optimizers in base lightning class.")

        optimizer = optimizer_class(self.parameters(), **optimizer_cfg)

        self.scheduler_name = self.training_cfg.get("scheduler", None)
        scheduler_cfg = self.training_cfg.get("scheduler_cfg", {})

        # Only ReduceLROnPlateau operates on epoch interval
        if self.scheduler_name == "Plateau":
            monitor_metric = scheduler_cfg.pop("monitor_metric", "Val/epoch_avg_loss")
            scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_cfg)
            return [optimizer], [{"scheduler": scheduler_class, 
                                  "monitor": monitor_metric}]
        # Manually schedule for Cosine and Polynomial scheduler
        elif self.scheduler_name in [None, "Cosine", "Poly"]:
            return optimizer
        else:
            raise ValueError(f"{self.scheduler_name} is not supported, add it to configure_optimizers in BaseDownstreamTask.")


    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, 
                             on_tpu=False, using_native_amp=False, using_lbfgs=False):
        """Overwrite with warmup epoch and manual lr decay"""

        self.epoch_progress = self.current_epoch + min((batch_idx+1)/self.num_steps_per_train_epoch , 1)
        initial_lr = self.training_cfg["optimizer_cfg"]['lr']
        warm_up_epoch = self.training_cfg.get("warm_up_epoch", 0)
        max_epochs = self.training_cfg['trainer']['max_epochs']

        if self.scheduler_name in ["Cosine", "Poly"] or self.epoch_progress <= warm_up_epoch:
            if self.epoch_progress <= warm_up_epoch:
                lr = initial_lr * self.epoch_progress / warm_up_epoch
            elif self.scheduler_name == "Cosine":
                lr = initial_lr * 0.5 * (1. + math.cos(math.pi * (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)))
            else:
                power = self.training_cfg.get("scheduler_cfg", {}).get("power", 0.5)
                lr = initial_lr * (1. - (self.epoch_progress - warm_up_epoch) / (max_epochs - warm_up_epoch)) ** power
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr    

        optimizer.step(closure=optimizer_closure)


    def train_dataloader(self):
        datasetcls_name = self.hparams['dataset']['train_dataset']
        datasetcls_cfg = self.hparams['dataset']['train_args']
        dataset = globals()[datasetcls_name](**datasetcls_cfg)
        data_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.training_cfg['train_loader_worker'])

        self.num_steps_per_train_epoch = math.ceil(len(data_loader) / self.num_gpu)
        return data_loader



    def val_dataloader(self):
        datasetcls_name = self.hparams['dataset']['eval_dataset']
        datasetcls_cfg = self.hparams['dataset']['eval_args']
        dataset = globals()[datasetcls_name](**datasetcls_cfg)
        self.idx_to_cls = dataset.index_mapping
        data_loader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.training_cfg['eval_loader_worker'])
        
        return data_loader