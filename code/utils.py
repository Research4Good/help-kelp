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
from tqdm import tqdm, trange
import functools
from typing import Any, List, Type, Union, Optional, Dict
#import albumentations as albu

from numpy import clip
# import plotly.express as px

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


def rescale( pixels, verbose=False ):
    mean, std = pixels.mean(), pixels.std()+1e-10
     
    pixels = (pixels - mean) / std
    
    # clip pixel values to [-1,1]
    # pixels = clip(pixels, -1.0, 1.0)
    
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + np.abs( pixels.min()) ) 

    pixels /= (pixels.max()+1e-10 )
    
    if verbose:
        print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))        
        print( pixels.min(), pixels.max() )
    return pixels


class Satellite_Dataset(Dataset):
    """
    Args:
        df - pandas.DataFrame with columns (image, mask, region)
        transfrom - torchvision.transforms for image augmentations
        inference - use the test mode
        channels - list of channels that to be used
    """
    def __init__(self, df: Any, img_dir, lab_dir: Any = None, transform: Any = None, inference: bool = False, channels: List[int] = None):      
        self.df = df
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.transform = transform
        self.inference = inference
        self.channels = channels
            
    def __len__(self):
        return len(self.df)
    
    def _read_img(self, path: str, channels: List[int]):
        path = self.img_dir + path + '_satellite.tif'
        if channels:
            img = rasterio.open(path).read(channels).transpose((1, 2, 0))
        else:
            img = rasterio.open(path).read().transpose((1, 2, 0))
        
        img[ img == -32768 ] = 99
        img = np.float16(img)/ 65536        
        if 0:
            for d in range(img.shape[2] ):
                img[:,:,d] = img[:,:,d] #np.float32(img) / 65535
        return img
    
    def _read_mask(self, path: str):
        path = self.lab_dir + path + '_kelp.tif'
        mask = rasterio.open(path).read().transpose((1, 2, 0))
        mask = np.uint8(mask>0)
        return mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        #path = row['image']  # fire landsat
        path = row['tile_id'] 
        
        try:
            image = self._read_img( path, self.channels)
        except:
            image = np.zeros( (350,350,7) ) 
            print('Cannot read image')
                     
        if self.inference:
            return image        
        try:
            mask = self._read_mask( path )
        except:
            mask = np.zeros( (350,350,1) ) 
            print('Cannot read mask')
        
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image= np.pad( image, ((1,1),(1,1),(0,0)) )
        mask = np.pad( mask, ((1,1),(1,1),(0,0)) )

        print( 'tileid:',path, 'AreaPercent:',mask.sum()/350/350 )
        
        return image, mask
    
    

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

