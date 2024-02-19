import pandas as pd
import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import struct, jax_utils
from flax.training.common_utils import shard
import optax
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import functools
from typing import Any, List, Type, Union, Optional, Dict
#import albumentations as albu

from numpy import clip
import plotly.express as px

import random
import shutil
# gpu and tpu saving didn't work in same way
try:
    from flax.training import orbax_utils
    from orbax.checkpoint import PyTreeCheckpointer
    USE_ORBAX_WITH_FLAX = True
except:
    from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, JsonCheckpointHandler, PyTreeCheckpointer
    import nest_asyncio
    nest_asyncio.apply()
    USE_ORBAX_WITH_FLAX = False

import utils

class CFG:
    inference = False    
    frac = .01
    
    # these were pretrained on Landsat-8
    pretrained = (
        '/kaggle/input/deeplabv3-resnet-50/ckpt',
        '/kaggle/input/deeplabv3-resnet-101/ckpt',
        '/kaggle/input/pan-resnet-50/ckpt',
        '/kaggle/input/pan-resnet-101/ckpt'
    )
    
    # you can change these parameters, but you don't have to
    seed = 42
    # specify correct optimizer name for optax (getattr(optax, optimizer_name: str))
    # https://optax.readthedocs.io/en/latest/api.html - list of optimizers
    optimizer = 'adam'
    # specify correct parameters dict, you can find them here - https://optax.readthedocs.io/en/latest/api.html
    optimizer_params = {
        'b1': 0.95,
        'b2': 0.98,
        'eps': 1e-8
    }
    # scheduler_params with such keys will be set to ttl_iters after calculating of total steps (ttl_iters)
    ttl_iters_keys = ['transition_steps', 'decay_steps']
    # specify correct scheduler name for optax (getattr(optax, scheduler_name: str))
    # https://optax.readthedocs.io/en/latest/api.html#schedules - list of schedulers
    scheduler = 'cosine_onecycle_schedule'
    # specify correct parameters dict, you can find them here - https://optax.readthedocs.io/en/latest/api.html#schedules
    # if your scheduler has one of the ttl_iters_keys, you can set it to anything, such as None
    scheduler_params = {
        'transition_steps': None,
        'peak_value': 1e-2,
        'pct_start': 0.25,
        'div_factor': 25,
        'final_div_factor': 100
    }
    # hyperparameters
    epochs = 150
    test_size = 0.1
    batch_size = 64
    
    # previous demo
    shape = (1, 256, 256, 10) # list of Landsat-8 channels - https://landsat.gsfc.nasa.gov/satellites/landsat-8/landsat-8-bands/
                              # input image shape, currently using 10 of 11 Landsat-8 channels, excluding channel number 8
    
    # current demo
    shape = (1, 352, 352, 7)
    
    # if you want to use specific channels to train the model, specify them in Tuple[int] format and change the shape tuple to the correct format
    channels = None
    
    # number of workers for torch DataLoader, don't set too high, I prefer to use 4 workers
    num_workers = 4

    # metadata keys
    metadata = ['config', 'model', 'loss']
    
    # which architecture to use, only when inference = False
    model = 'PAN_ResNet152'
    model = 'DeepLabV3_ResNet101'

    # path to save checkpoint
    ckpt_path = f'/kaggle/working/{model}_ckpt'
