"""
Base experiment runner
"""
from abc import ABC, abstractmethod
from ast import Dict
import os
import copy
import pathlib
from typing import List, Optional
from tqdm import tqdm
import inspect
import numpy as np
import pandas as pd
import warnings 

import torch
from torch import nn
from torch.utils.data import DataLoader

# Suppress dynamo errors for older GPUs
if torch.__version__ >= "2.0.0":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

import pylot
from pylot.experiment.util import absolute_import, eval_config
from pylot.experiment.base import BaseExperiment as PylotBaseExperiment
from pylot.util import Timer
from pylot.util.torchutils import to_device
from pylot.nn.util import num_params, split_param_groups_by_weight_decay
from pylot.util.ioutil import autohash, autoload, autosave
from pylot.util.hash import json_digest
from pylot.util.meter import MeterDict


class BaseExperiment(PylotBaseExperiment):

    def __init__(self, path, compile: Optional[bool] = None):
        """
        Args:
            path (str): Path to experiment directory
        """
        torch.backends.cudnn.benchmark = True
        super().__init__(path)
        print(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = self.config['experiment'].get("verbose", True)
        self.build_data()
        self.build_prompt_generator()
        self.build_model(compile=compile)
        self.build_optim()
        self.build_loss()
        self.build_metrics()
        self.build_augmentations()
        self._epoch = 0
        print("Finished init....")
        # Save class module (name is already saved)
        self.properties["experiment.module"] = self.__class__.__module__
        # Run shape/value checks
        self.safe = self.config.get("experiment.safe", False)

    def build_data(self):
        data_cfg = self.config["data"].to_dict()
        dataset_cls = absolute_import(data_cfg.pop("_class"))

        def load_task(task_cfg, split):
            cfg = task_cfg.copy()
            task_cls = absolute_import(cfg.pop("_class"))
            task_cfg = {
                **cfg,
                "split": split,
            }
            return task_cls(**task_cfg)

        train_task_cfgs = data_cfg.pop("train_tasks")
        val_task_cfgs = data_cfg.pop("val_tasks")

        train_tasks = [load_task(cfg, "train") for cfg in train_task_cfgs]

        self.train_dataset = dataset_cls(
            tasks=train_tasks, **data_cfg
        )

        # Disable superpixel augmentations at eval
        if "superpixel_prob" in data_cfg:
            data_cfg["superpixel_prob"] = 0

        val_id_tasks = [load_task(cfg, "val") for cfg in train_task_cfgs]
        val_od_tasks = [load_task(cfg, "val") for cfg in val_task_cfgs]

        self.val_id_dataset = dataset_cls(
            tasks=val_id_tasks, **data_cfg
        )
        self.val_od_dataset = dataset_cls(
            tasks=val_od_tasks, **data_cfg
        )

    def build_prompt_generator(self):
        """
        Build prompt generator for generating ineractions during training
        """
        generator_cfg = self.config["prompt_generator"]
        self.prompt_generator = eval_config(generator_cfg)

    def build_dataloader(self):
        """
        Build pytorch dataloaders
        """
        dl_cfg = self.config["dataloader"]

        train_tasks = self.train_dataset.task_df.copy()
        val_id_tasks = self.val_id_dataset.task_df.copy()
        val_od_tasks = self.val_od_dataset.task_df.copy()
        train_tasks["phase"] = "train"
        val_id_tasks["phase"] = "val_id"
        val_od_tasks["phase"] = "val_od"

        all_tasks = pd.concat(
            [train_tasks, val_id_tasks, val_od_tasks], ignore_index=True,
        )

        if not (p := self.path / "data.parquet").exists() or ("data_digest" not in self.properties):
            autosave(all_tasks, p)
            self.properties["data_digest"] = autohash(all_tasks)
        else:
            if autohash(all_tasks) != self.properties["data_digest"]:
                warnings.warn(
                    f"Underlying data has changed since experiment creation: {self.properties['data_digest']} {autohash(all_tasks)}"
                )

        self.train_dl = DataLoader(
            self.train_dataset, shuffle=True, **dl_cfg
        )
        self.val_id_dl = DataLoader(
            self.val_id_dataset, shuffle=False, drop_last=False, **dl_cfg
        )
        self.val_od_dl = DataLoader(
            self.val_od_dataset, shuffle=False, drop_last=False, **dl_cfg
        )
        
    @property
    def state(self):
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "_epoch": self.properties.get("epoch", self._epoch),
        }

    def build_model(self, compile: Optional[bool] = None):
        """
        Build basic model
        """
        model_config = self.config["model"].to_dict()
    
        if compile is None:
            compile_model = model_config.pop("compile", True)
        else:
            compile_model = compile
            model_config.pop('compile', None)

        pretrained_weights = model_config.pop("pretrained_weights", None)
        
        self.model = eval_config(model_config)

        if torch.__version__ >= "2.0.0" and compile_model:
            self.model = torch.compile(self.model)
            self.compiled = True
        else:
            self.compiled = False
        
        self.properties["num_params"] = num_params(self.model)

        if self.config.get("train.data_parallel", False):
            self.model = nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        if pretrained_weights is not None:
            with open(pretrained_weights, "rb") as f:
                # Only load the weights, not the optimizer or epoch
                state = torch.load(f, map_location=self.device)
                self.model.load_state_dict(state)
                if self.verbose:
                    print(
                        f"Loaded pretrained weights from: {pretrained_weights}"
                    )

    def build_loss(self):
        """
        Build basic loss
        """
        loss_kws = self.config["loss_func"].to_dict()
        self.loss_func = eval_config(loss_kws)
        if self.config.get("train.fp16", False) or self.config.get("train.bf16", False):
            assert torch.cuda.is_available()
            self.grad_scaler = torch.cuda.amp.GradScaler()

    def build_optim(self):
        """
        Build the optimizer
        """
        optim_cfg = self.config["optim"].to_dict()

        if "weight_decay" in optim_cfg:
            optim_cfg["params"] = split_param_groups_by_weight_decay(
                self.model, optim_cfg["weight_decay"]
            )
        else:
            optim_cfg["params"] = self.model.parameters()

        self.optim = eval_config(optim_cfg)

    def build_metrics(self):
        self.metric_fns = {}
        if "log.metrics" in self.config:
            self.metric_fns = eval_config(copy.deepcopy(self.config["log.metrics"]))

    def build_initialization(self):
        if "initialization" in self.config:
            init_cfg = self.config["initialization"].to_dict()
            path = pathlib.Path(init_cfg["path"])
            with path.open("rb") as f:
                state = torch.load(f, map_location=self.device)
            if not init_cfg.get("optim", True):
                state.pop("optim", None)
            strict = init_cfg.get("strict", True)
            self.set_state(state, strict=strict)
            if self.verbose:
                print(f"Loaded initialization state from: {path}")

    @property
    def checkpoints(self, as_paths=False) -> List[str]:
        checkpoints = list((self.path / "checkpoints").iterdir())
        checkpoints = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
        if as_paths:
            return checkpoints
        return [c.stem for c in checkpoints]

    def checkpoint(self, tag : str = None) -> None:
        """
        Save the current state of the experiment to a checkpoint file.
        """
        self.properties["epoch"] = self._epoch

        checkpoint_dir = self.path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        tag = tag if tag is not None else "last"
        if self.verbose:
            print(f"Checkpointing with tag:{tag} at epoch:{self._epoch}")

        with (checkpoint_dir / f"{tag}.pt").open("wb") as f:
            torch.save(self.state, f)

    def set_state(self, state, strict=True):
        """
        Load model weights or optimizer state
        """
        for attr, state_dict in state.items():
            if not attr.startswith("_"):
                x = getattr(self, attr)
                if isinstance(x, nn.Module):
                    if self.compiled: 
                        for key in list(state_dict.keys()):
                            if "orig_mod" not in key:
                                state_dict['_orig_mod'+key] = state_dict.pop(key)
                    else:
                        for key in list(state_dict.keys()):
                            if '_orig_mod' in key:
                                state_dict[key.replace('_orig_mod.','')] = state_dict.pop(key)
                    x.load_state_dict(state_dict, strict=strict)
                elif isinstance(x, torch.optim.Optimizer):
                    x.load_state_dict(state_dict)
                else:
                    raise TypeError(f"Unsupported type {type(x)}")
            else:
                if attr == "_epoch":
                    self._checkpoint_epoch = state_dict # for compatibility with universeg eval code
                    self._epoch = state_dict

    def load(self, tag=None):
        """
        Load the state of the model from a checkpoint file.
        """
        checkpoint_dir = self.path / "checkpoints"
        tag = tag if tag is not None else "last"
        with (checkpoint_dir / f"{tag}.pt").open("rb") as f:
            state = torch.load(f, map_location=self.device)
            self.set_state(state)
            if self.verbose:
                print(
                    f"Loaded checkpoint with tag:{tag} from epoch {self._epoch}. Last epoch:{self.properties['epoch']}"
                )
        return self

    def stepwise_steps(self):
        from scribbleprompt.analysis.pertask import load_epoch_task_stats
        return load_epoch_task_stats(self.path / "store")
    
    def to_device(self, gpu_idx=None):
        """
        Move the model to cpu or gpu
        """
        if gpu_idx:
            self.model = to_device(
                self.model, gpu_idx, self.config.get("train.channels_last", False)
            )
        else:
            self.model = to_device(
                self.model, self.device, self.config.get("train.channels_last", False)
            )

    def run_callbacks(self, callback_group, **kwargs):
        for callback in self.callbacks.get(callback_group, []):
            callback(**kwargs)

    def compute_metrics(self, metric_fns: dict, outputs: dict) -> dict:
        """
        Apply functions given in metric_fns to the outputs dict and return results

        To calculate std of a metric accross all the examples in an epoch
        - need to update metric functions to return vector
        - modify MeterDict to update using a vector of values
        """
        metrics = {}
        # Calculate specified metrics
        for name, fn in metric_fns.items():
            # Automatically pass the right argments to the metric function
            fn_args = inspect.getfullargspec(fn).args
            fn_kwargs = {k: v for k, v in outputs.items() if k in fn_args}
            value = fn(**fn_kwargs)
            # Store results
            if isinstance(value, torch.Tensor):
                if len(value.size()) == 0:
                    # 0-d tensor
                    metrics[name] = value.item()
                elif len(value) > 1:
                    # For metrics that return a vector of values (i.e. sequence metrics), store each value separately
                    for i in range(len(value)):
                        metrics[f"{name}.{i}"] = value[i].item()
                else:
                    metrics[name] = value.item()
            elif isinstance(value, dict):
                for (k, v) in value.items():
                    if isinstance(value, torch.Tensor):
                        metrics[f"{name}.{k}"] = v.item()
                    else:
                        metrics[f"{name}.{k}"] = v
            else:
                metrics[name] = value
        return metrics

    def build_augmentations(self):
        
        if "augmentations" in self.config:
            from scribbleprompt.augmentation import augmentations_from_config
        
            aug_cfg = self.config.to_dict()["augmentations"]
            self.aug_pipeline = augmentations_from_config(aug_cfg)
            self.properties["aug_digest"] = json_digest(self.config["augmentations"])[:8]
        else:
            self.aug_pipeline = None
