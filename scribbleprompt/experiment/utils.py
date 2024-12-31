import time
import datetime
import os
import yaml
import pathlib
import shutil
from random import randint
from typing import Union


def fmt_time(init_time, finish_time, epochs: int = 1):
    """
        Format time difference as Hours:Minutes:Seconds
    """
    return str(datetime.timedelta(seconds=max(finish_time - init_time,1e-7)/max(epochs,1)))


def copy_load_yaml(path: Union[pathlib.Path,str], save_path: bool = True) -> dict:
    """
    To prevent issues with slurm jobs reading from the same file at the same time
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
        
    # Make a temporary copy
    new_path = "/tmp/" + str(randint(1,9999)) + path.name
    shutil.copy(str(path), new_path)

    # Load the copy
    config = yaml.safe_load(open(new_path, 'r'))
    if save_path: 
        config['path'] = path

    # Delete temp file
    # os.remove(new_path)

    return config