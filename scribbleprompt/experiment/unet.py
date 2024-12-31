"""
ScribblePrompt-UNet Trainer
"""
import os
import sys
from token import OP
import warnings
from typing import List, Union, Optional, Dict, Any, Literal, Tuple
import yaml
import argparse
import time
import numpy as np
import pandas as pd
import socket
import copy
from tqdm.autonotebook import tqdm
import torch
import pathlib

from pylot.experiment.util import eval_config, absolute_import, generate_tuid
from pylot.util.torchutils import to_device
from pylot.util.meter import MeterDict
from pylot.util.ioutil import autohash, autosave
from pylot.util.config import config_digest

from scribbleprompt.experiment.base import BaseExperiment
from scribbleprompt.experiment.utils import fmt_time


class ScribblePromptExperiment(BaseExperiment):
    """
    Interactive segmentation trainer
    """
    def __init__(self, path, compile: Optional[bool] = False):
        """
        Args:
            path (str): Path to experiment directory
            compile (bool): Compile the model when loading 
                This option is useful if you trained with torch.compile but 
                then want to reload the model on a GPU that doesn't support it
        """
        super().__init__(path, compile)

    def run(self, resume_tag=None):

        start_t = time.time()
        
        self.to_device()
        self.build_dataloader()
        self.build_callbacks()
        
        if self.verbose:
            print(f"Running {str(self)}")
        epochs: int = self.config["train.epochs"]

        last_epoch: int = self.properties.get("epoch", -1)
        if last_epoch >= 0:
            self.load(tag="last" if resume_tag is None else resume_tag)
            last_epoch = self._epoch
            df = self.metrics.df
            autosave(df[df.epoch < last_epoch], self.path / "metrics.jsonl")
        else:
            self.build_initialization()
            self._epoch = 0

        self.to_device()
        self.optim.zero_grad()

        print(self.model)
        print(self.device)

        checkpoint_freq: int = self.config.get("log.checkpoint_freq", 1)
        eval_freq: int = self.config.get("train.eval_freq", 1)
        save_freq: int = self.config.get("log.save_freq", 100)

        self.run_callbacks("setup")

        try:
            if self.verbose:
                epoch_iter = range(last_epoch + 1, epochs)
            else:
                epoch_iter = tqdm(range(last_epoch + 1, epochs))

            for epoch in epoch_iter:

                if self.verbose:
                    print(f"Start epoch {epoch}")

                self._epoch = epoch
                self.run_phase("train", epoch)

                if epoch % eval_freq == 0 or epoch == epochs - 1:
                    self.run_phase("val_id", epoch)
                    self.run_phase("val_od", epoch)

                self.run_callbacks("epoch", epoch=epoch)

                if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                    self.checkpoint(tag="last")
                    print(self.path)

                if save_freq > 0 and epoch % save_freq == 0:
                    self.checkpoint(tag=f"epoch-{epoch}")
                    print("Saving copy of weights")

            self.checkpoint(tag="last")

            end_t = time.time()

            self.run_callbacks("wrapup")

        except KeyboardInterrupt:
            print(f"Interrupted at epoch {epoch}. Tearing Down... {self.path}")
            self.checkpoint(tag="interrupt")
            sys.exit(1)

        end_t = time.time()
        print(f"Training Time: {fmt_time(start_t, end_t)}")
        print(f"{fmt_time(start_t, end_t, epochs=self.config['train.epochs'])} per epoch")

    def run_phase(self, phase, epoch):

        self._stats = []
        
        grad_enabled = phase == "train"
        augmentation = (phase == "train") and ("augmentations" in self.config)
        prompt_iter = self.config["train"].get("prompt_iter", 1) 

        self.model.train(grad_enabled)  # For dropout, batchnorm, &c

        metrics = {}
        meters = MeterDict()
        dl = getattr(self, f"{phase}_dl")

        batch_iter = tqdm(enumerate(dl), total=len(dl)) if self.verbose \
            else enumerate(dl)
        
        with torch.set_grad_enabled(grad_enabled):
            for batch_idx, batch in batch_iter:
                outputs = self.run_step(
                    batch_idx,
                    batch,
                    backward=grad_enabled,
                    augmentation=augmentation,
                    prompt_iter=prompt_iter,
                )
                metrics.update(self.compute_metrics(outputs))
                meters.update(metrics)
                self.run_callbacks("batch", epoch=epoch, batch_idx=batch_idx)

        metrics = {
            "phase": phase, 
            "epoch": epoch, 
            **meters.collect("mean")
        }

        self.metrics.log(metrics)

        df = pd.DataFrame.from_records(self._stats)
        self.store[f"stats.{phase}.{epoch:05d}"] = df

        return metrics

    def run_step(self, batch_idx, batch, prompt_iter=1, backward=True, augmentation=False):
        
        task, img, seg = batch
        img, seg = to_device((img, seg), self.device)
        # img and seg shape: B x 1 x H x W

        if augmentation:
            with torch.no_grad():
                # Only 1 channel so doesn't matter if we use regular 
                # augmentations or channelwise augmentations
                img, seg = self.aug_pipeline.forward(img, seg)

        # Run iterative training for a batch
        loss, yhat, y = self.run_iter(img, seg, prompt_iter=prompt_iter)

        if backward:
            if self.config.get("train.fp16", False) or self.config.get("train.bf16", False):
                if not self.config.get("train.grad_accumulation", False):
                    final_loss = loss.mean()
                    # Scales the loss, and calls backward()
                    # to create scaled gradients
                    self.grad_scaler.scale(final_loss).backward()
                # Unscales gradients and calls
                # or skips optimizer.step()
                self.grad_scaler.step(self.optim)
                # Updates the scale for next iteration
                self.grad_scaler.update()
            else:
                if not self.config.get("train.grad_accumulation", False):
                    loss.mean().backward()
                self.optim.step()

            self.optim.zero_grad()

        results = {
            "loss": loss,
            "seg_pred": yhat,
            "seg_true": y,
            "batch_idx": batch_idx,
            "task": task,
        }
        return results
    

    def run_iter(self, img, seg, prompt_iter=1):

        prompts = {}
        seg_pred = []
        loss_lst = []

        for i in range(prompt_iter):

            with torch.no_grad():
                if i == 0:
                    # Generate prompt (model input) after augmentation
                    prompts = self.prompt_generator(img, seg)
                else:
                    # Get the mask with the lowest loss
                    # bs = img.shape[0]
                    # # yhat shape: (B, 1, H, W)
                    # input_mask = yhat[torch.arange(bs), argmin_loss, ...]
                    if self.config.get('experiment.detach', False):
                        input_mask = yhat.detach()

                    # Sample new click/scribble from error region
                    prompts = self.prompt_generator.subsequent_prompt(
                        mask_pred=input_mask,
                        prev_input=prompts,
                        new_prompt=True
                    )

            x = prompts.get("x") # shape: B x 1 x H x W 
            y = prompts.get("y") # Shape: B x 1 x H x W

            # Out Shape: B x C x H x W
            if self.config.get("train.fp16", False):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    yhat = self.model(x)
                    loss = self.loss_func(y_pred=yhat, y_true=y)
            elif self.config.get("train.bf16", False):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    yhat = self.model(x)
                    loss = self.loss_func(y_pred=yhat, y_true=y)
            else:
                yhat = self.model(x)
                loss = self.loss_func(y_pred=yhat, y_true=y)
            
            seg_pred.append(yhat)
            loss_lst.append(loss)

        loss = torch.cat(loss_lst, dim=0)
        yhat = torch.cat(seg_pred, dim=0)
        y = prompts.get("y").repeat(prompt_iter, 1, 1, 1)

        return loss, yhat, y

    # ----------------------------------------------------------------------

    def compute_metrics(self, outputs):
        """
        Modified to handle training on multiple iterations of prompts
        """
        metrics = {
            "loss": outputs["loss"].mean().item()
        }
        b = outputs["batch_idx"]
        prompt_iter = self.config['train'].get('prompt_iter', 1)
        unreduced_metrics = []
        for i in range(prompt_iter):
            for task in outputs["task"]:
                unreduced_metrics.append({"batch": b, "task": task, 'iter': i})
        for i, val in enumerate(outputs["loss"]):
            unreduced_metrics[i]["loss"] = val.item()
        for name, fn in self.metric_fns.items():
            value = fn(
                outputs["seg_pred"], outputs["seg_true"],
                batch_reduction=None,
                from_logits=self.config.get("loss_func.from_logits", False)
                )
            for i, val in enumerate(value):
                unreduced_metrics[i][name] = val.item()
            metrics[name] = value.mean().item()
        self._stats.extend(unreduced_metrics)

        return metrics

    # ----------------------------------------------------------------------

    def run_inference(self, 
                    generator_config: Dict[str,str] = None,
                    prompt_iter: int = 5,
                    n_predictions: int = 10, 
                    checkpoint: str = "max-val_od-dice_score",
                    tasks: Optional[List[callable]] = None,
                    preloaded: bool = False, # whether to use self.val_od_dataset (instead of tasks)
                    resize_input: Optional[Tuple[int,int]] = None,
                    calc_hd95: bool = False,
                    seed: int = 42,
                    save: bool = True,
        ):
        pass
        

# Example use:
# >>> from scribbleprompt.experiment.unet import ScribblePromptExperiment
# >>> exp =  ScribblePromptExperiment.from_config(config)
# >>> exp.run()

if __name__ == "__main__":

    from multiversegdev.utils import paths
    config_dir = pathlib.Path(os.environ.get("SP_CONFIGPATH", None))
    inf_config_dir = config_dir / "inference"

    config_dir = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent / "configs"

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-config",
        default=str(config_dir / "train_unet.yaml"),
        help="Path to experiment config file")

    parser.add_argument("--time", "-time",
        action='store_true',
        default=False,
        help="Run callbacks to time data loading and model forward pass")

    parser.add_argument("--scratch", "-scratch",
        action='store_true',
        default=False,
        help="Set path to scratch folder and disable wandb logging (for debugging)")

    parser.add_argument("--resume", "-resume",
        action='store_true',
        default=False,
        help="Whether to resume training from the given config file"
    )

    parser.add_argument("--tag", "-tag",
        type=str,
        default=None,
        help="Checkpoint to load before resuming training or performing inference"
    )

    parser.add_argument("--inference", "-inference",
        action='store_true',
        default=False,
        help="Run inference only"
    )

    parser.add_argument("--inf_config", "-inf_config",
        type=str,
        default=None,
        help="Name of inference config file"
    )

    args = parser.parse_args()
    
    if args.resume or args.inference:
        this_cfg = args.config
        print(f"Using config from {args.config}")
    else:
        config_fpath = config_dir / args.config
        print(f"Loading config from {args.config}")
        this_cfg = yaml.safe_load(open(config_fpath, 'r'))

    if args.time or args.scratch:
        this_cfg['log']['root'] = os.environ.get("SCRATCHPATH", "../scratch")
        # Remove wandb logging
        this_cfg['callbacks']['epoch'] = [x for x in this_cfg['callbacks']['epoch'] if 'WandbLogger' not in str(x)]

    if args.time:
        import pylot.callbacks
        # Benchmark training loop
        exp = ScribblePromptExperiment.from_config(this_cfg)
        exp.build_dataloader()
        exp.to_device()
        pylot.callbacks.Throughput(exp, n_iter=100)

    else:
        start = time.time()

        if args.inference:
            # Load inference config
            if args.inf_config is not None:
                print(f"Loading config from {args.inf_config}")

                from scribbleprompt.experiment.utils import copy_load_yaml
                generator_config = copy_load_yaml(inf_config_dir / args.inf_config)
                generator_config["sam"] = False

                generator_config_name = args.inf_config.split('.')[0]
            else:
                generator_config = None
                generator_config_name = ""

            exp = ScribblePromptExperiment(this_cfg)

            # Run inference
            start_t = time.time()
            exp.run_inference(
                generator_config_name = generator_config_name,
                generator_config=generator_config,
                checkpoint=(args.tag or "max-val_od-dice_score"),
                preloaded=True,
            )

        else:
            # Training
            if args.resume:
                exp = ScribblePromptExperiment(this_cfg)
                start_t = time.time()
                start_epoch = exp._epoch
                exp.run(resume_tag=args.tag)
            else:
                exp = ScribblePromptExperiment.from_config(this_cfg)
                start_t = time.time()
                start_epoch = 0
                exp.run()
            
        end = time.time()

        print(f"Initialization Time: {fmt_time(start, start_t)}")
        print(f"Total Time: {fmt_time(start_t, end)}")
        if not args.inference:
            delta_epochs = exp._epoch - start_epoch
            print(f"Average {fmt_time(start_t, end, epochs=delta_epochs)} per epoch")
        print(exp.path)
