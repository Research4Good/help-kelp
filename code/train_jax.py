import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from spectral import imshow 

import os, sys, cv2

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
import albumentations as albu
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
 
import plotly.express as px
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image 
import rasterio 
from numpy import clip

def rescale( pixels ):
    mean, std = pixels.mean(), pixels.std()
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    print( pixels.min(), pixels.max() )
    return pixels

print( albu.__version__ )
