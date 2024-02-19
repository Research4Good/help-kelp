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


def checker( a, b, n=50 ):        
    assert a.shape == b.shape
    w,h = a.shape        
    for d in range( n ):    
        r = np.zeros( n )        
        if d==0:            
            r[:n//2]=1
            r1=r
        elif d<n//2:
            r[:n//2]=1
            r1=np.vstack( (r1,r) )
        else:
            r[n//2:]=1    
            r1=np.vstack( (r1,r) )
            
    ar = np.array( r1 )    
    
    W,H=ar.shape     
    W=w//W
    H=h//H
    x0 = np.tile( ar, [W,H])    
    x = np.pad( x0, ((0,0),(w - x0.shape[0], h - x0.shape[1])) )    
    
    res = a*(1 - x) + b*x
    plt.imshow(res)    
    return res, x0, x, r1

res, x0, x, r1 = checker( rescale(img[:,:,1]), label )
