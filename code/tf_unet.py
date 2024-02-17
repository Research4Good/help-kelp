from IPython.display import clear_output

import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import struct, jax_utils
from flax.training.common_utils import shard
import optax

from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np # linear algebra
from numpy import clip

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from spectral import imshow 
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image 
import rasterio 

import os, sys, cv2
from glob import glob
from tqdm import tqdm
import functools
from typing import Any, List, Type, Union, Optional, Dict
import random
import shutil


from sklearn.model_selection import train_test_split

#import albumentations as albu

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

def seed_everything(seed: int):  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

IMSZ=[352,352,7] # input size
EPOCHS = 10
OUTPUT_CLASSES = 2  # segmentation mask represents 2 classes
SEED=111
seed_everything(SEED)



def rescale( pixels ):
    mean, std = pixels.mean(), pixels.std()
    # global standardization of pixels
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0

    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    #print( pixels.min(), pixels.max() )
    return pixels

class DataGenerator(Sequence):
    """
    Args:
        df - pandas.DataFrame with columns (image, mask, region)
        transfrom - torchvision.transforms for image augmentations
        inference - use the test mode
        channels - list of channels that to be used
    """
    def __init__(self, df: Any, transform: Any = None, inference: bool = False, channels: List[int] = None, 
                 imgdir = '', labeldir=None, channel_last= True ):      
        self.df = df
        self.transform = transform
        self.inference = inference
        self.channels = channels
        self.imgdir = imgdir 
        self.labeldir = labeldir
        self.channel_last = channel_last
    def __len__(self):
        return len(self.df)
    
    def _read_img(self, path: str, channels: List[int]):
        if channels:
            img = rasterio.open( self.imgdir + path).read(channels)
        else:
            img = rasterio.open( self.imgdir + path).read()        
        
        for d in channels:
            dd=d-1
            img[:,:,dd] = rescale( img[:,:,dd])

        return img
    
    def _read_mask(self,  path: str):
        mask = rasterio.open( self.labeldir + path).read()
        mask = np.int32(mask)
        return mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._read_img( row['tile_id'] + '_satellite.tif', self.channels)
         
        print(idx, row['tile_id'] )
        if self.inference:
            return image
        
        mask = self._read_mask(row['tile_id'] + '_kelp.tif')
        
        if self.transform:
            sample = self.transform(image=image)
            mask = self.transform( mask=mask )
            image, mask = sample['image'], sample['mask']
            
        if self.channel_last:
            image=image.transpose((1, 2, 0))
            mask=mask.transpose((1, 2, 0))

        image= np.pad( image, ((1,1),(1,1),(0,0)) )
        mask = np.pad( mask, ((1,1),(1,1),(0,0)) )
        image= np.expand_dims( image, 0 )
        mask = np.expand_dims( mask, 0 )
        return image, mask




meta_pd=pd.read_csv('/kaggle/input/landsat-30m-350x350-7bands/metadata_fTq0l2T.csv')
meta_pd.query('type == "kelp"')
meta_pd=meta_pd.sort_values('filename').query( 'in_train == True')
meta_pd.head()

train_df, test_df = train_test_split( meta_pd, test_size=CFG.test_size, random_state=CFG.seed, )
           
train_dataset = DataGenerator(train_df, channels=[1,2,3,4,5,6,7], transform=None, #tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
                        imgdir='/kaggle/input/landsat-30m-350x350-7bands/train_features.tar_MLIC14m/train_satellite/',                        
                        labeldir='/kaggle/input/landsat-30m-350x350-7bands/train_labels.tar_l8u2RP0/train_kelp/')

test_dataset = DataGenerator(test_df, channels=[1,2,3,4,5,6,7], transform=None,
                        imgdir='/kaggle/input/landsat-30m-350x350-7bands/train_features.tar_MLIC14m/train_satellite/',                        
                        labeldir='/kaggle/input/landsat-30m-350x350-7bands/train_labels.tar_l8u2RP0/train_kelp/')

x,y = train_dataset.__getitem__(10)
x,y = test_dataset.__getitem__(10)
# a | b | rgb -> 1 | cloud | elevation 
print( 'Shapes of Image, Mask:', x.shape, y.shape )


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels




base_model = tf.keras.applications.MobileNetV2(input_shape=IMSZ, weights = None, include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
#down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512*2, 2),  # 4x4 -> 8x8
    pix2pix.upsample(512, 2),  # 4x4 -> 8x8     
    pix2pix.upsample(256, 2),  # 8x8 -> 16x16
    pix2pix.upsample(128, 2),  # 16x16 -> 32x32    
    pix2pix.upsample( 64, 2),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=IMSZ)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        print( x.shape, '<-x')
        print( skip.shape, '<-skips')
        x = concat([x, skip])
        x = tf.keras.layers.Dropout(.15)(x)

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
    filters=output_channels, kernel_size=2, strides=2,  padding='same')  #64x64 -> 128x128
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(output_channels=OUTPUT_CLASSES)

early_stop=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=8,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)

checkpoint_path = "training_unet/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
save_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=50)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

def dice_coef(y_true, y_pred, smooth=100):           
    y_pred = tf.math.argmax(y_pred,3)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=100):
    return 1 - dice_coef(y_true, y_pred, smooth)


model.compile(optimizer='adam',run_eagerly=True, 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[dice_coef_loss])

tf.keras.utils.plot_model(model, show_shapes=True)

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch= train_df.shape[0],
                          validation_steps= test_df.shape[0],
                          validation_data=test_dataset,
                          verbose=0,
                          callbacks=[early_stop, save_callback ])

loss = model_history.history['loss']
hist = model_history.history

for h in hist.keys():
    plt.plot( hist[h], label=h )

plt.legend()
