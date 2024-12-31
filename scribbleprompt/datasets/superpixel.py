
from dataclasses import dataclass, field
from typing import Literal, Dict, List, Any, NamedTuple
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from skimage.segmentation import felzenszwalb

import pylot
import pylot.pandas
from pylot.experiment.util import absolute_import, eval_config

from ..interactions.utils import _as_single_val

class Sample(NamedTuple):
    task: str
    images: torch.Tensor
    segs: torch.Tensor

@dataclass
class MultiDataset(Dataset):
    """
    Data loader for sampling from multiple datasets
    """
    tasks: List[Dataset]
    sampling: Literal["task", "hierarchical"] = "hierarchical"
    samples_per_epoch: int = 1_000

    def __post_init__(self):
        self.task_df = self.get_task_df()

    def get_task_df(self):
        df = pd.DataFrame.from_records([task.attr for task in self.tasks])

        # create full_task column
        def full_task(dataset, subdataset, modality, axis, label):
            return f"{dataset}/{subdataset}/{modality}/{axis}/{label}"
        
        df.augment(full_task)

        return df

    def _sample_task(self):

        # fmt: off
        sampling_order = {
            'task': ('full_task',),
            'hierarchical': ("dataset", "subdataset", "modality", "axis", "label"),
        }[self.sampling]
        # fmt: on

        df = self.task_df

        for attr in sampling_order:
            val = random.choice(df[attr].unique())
            df = df[df[attr] == val]
        assert len(df) == 1, f"Sampling failed: {len(df)} tasks found in {df.T}"
        row = df

        i = row.index.item()
        row = row.iloc[0].to_dict()
        
        return row['full_task'], self.tasks[i]

    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, _) -> Sample:

        task, target_dataset = self._sample_task()
        idx = np.random.randint(len(target_dataset))
        img, seg = target_dataset[idx]

        return (task, img, seg)


@dataclass
class SuperpixelMultiDataset(MultiDataset):
    """
    Data loader for sampling from multiple datasets with superpixel labels
    """
    tasks: List[Dataset]
    sampling: Literal["task", "hierarchical"] = "hierarchical"
    samples_per_epoch: int = 1_000
    # Superpixel arguments
    superpixel_prob: float = 0.5
    superpixel_method: Literal["felzenszwalb"] = "felzenszwalb"
    superpixel_kwargs: Dict[str,Any] = field(default_factory=lambda: {'scale':[1,500], 'sigma': 0.5, 'min_size': 32})

    def __post_init__(self):
        super().__post_init__()
        
    def get_superpixels(self, img: np.ndarray) -> np.ndarray:
        if self.superpixel_method == "felzenszwalb":
            # Sample parameters
            fn_kwargs = {k: _as_single_val(v) for k,v in self.superpixel_kwargs.items()}
            assert img.dtype == np.float32, f"Image should be np.float32, got {img.dtype}"
            return felzenszwalb(img.squeeze(), **fn_kwargs)
        else:
            raise NotImplementedError(f"Superpixel method {self.superpixel_method} not implemented")
    
    def __getitem__(self, idx) -> Sample:
        """
        With some probability replace the real label with a superpixel label
        """
        if random.uniform(0,1) < self.superpixel_prob:
            
            task, img, _ = super().__getitem__(idx)
            superpixel_seg = self.get_superpixels(img.numpy())

            # Choose a random superpixel
            label = random.randint(0,superpixel_seg.max())
            seg = (superpixel_seg == label).astype(np.float32)[None,...]
            seg = torch.from_numpy(seg)

            return (task, img, seg)
        else:
            # Get image and label as normal
            return super().__getitem__(idx)
