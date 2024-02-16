import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from spectral import imshow 

import os, sys

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
print(flax.__version__)
print(jax.devices()) 

class CFG:
    # if you want to train model you must set inference = False
    # if you want to test model you must set inference = True and set pretrained to actual folders tuple
    # other CFG parameters won't be used if you set inference = True (except seed, test_size and channels)
    inference = False
    
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
    epochs = 50
    test_size = 0.1
    batch_size = 32
    # input image shape, currently using 10 of 11 Landsat-8 channels, excluding channel number 8
    # list of Landsat-8 channels - https://landsat.gsfc.nasa.gov/satellites/landsat-8/landsat-8-bands/
    shape = (1, 256, 256, 10)
    
    shape = (1, 350, 350, 7)
    # if you want to use specific channels to train the model, specify them in Tuple[int] format and change the shape tuple to the correct format
    channels = None
    # number of workers for torch DataLoader, don't set too high, I prefer to use 4 workers
    num_workers = 4
    # path to save checkpoint
    ckpt_path = '/kaggle/working/ckpt'
    # metadata keys
    metadata = ['config', 'model', 'loss']
    # which architecture to use, only when inference = False
    model = 'PAN_ResNet101'
    
    
N_DC=256 
    
class DEEPLABV3_RESNET18:
    # resnet-18
    block = 'BasicBlock'
    layers = [2, 2, 2, 2]
    replace_stride_with_dilation = [False, False, False]
    strides = [2, 2, 2]
    # deelabv3
    decoder_channels = N_DC 
    atrous_rates = (6, 12, 24)
    # segmentation head
    upsampling = 32
    classes = 1
    activation = ''

    
class DEEPLABV3_RESNET34:
    # resnet-34
    block = 'BasicBlock'
    layers = [3, 4, 6, 3]
    replace_stride_with_dilation = [False, False, False]
    strides = [2, 2, 2]
    # deelabv3
    decoder_channels = N_DC
    atrous_rates = (6, 12, 24)
    # segmentation head
    upsampling = 32
    classes = 1
    activation = ''
    

class DEEPLABV3_RESNET50:
    # resnet-50
    block = 'Bottleneck'
    layers = [3, 4, 6, 3]
    replace_stride_with_dilation = [False, True, True]
    strides = [2, 2, 4]
    # deelabv3
    decoder_channels = N_DC*2
    atrous_rates = (6, 12, 24)
    # segmentation head
    upsampling = 8
    classes = 1
    activation = ''
    

class DEEPLABV3_RESNET101:
    # resnet-101
    block = 'Bottleneck'
    layers = [3, 4, 23, 3]
    replace_stride_with_dilation = [False, True, True]
    strides = [2, 2, 4]
    # deelabv3
    decoder_channels = N_DC*2
    atrous_rates = (6, 12, 24)
    # segmentation head
    upsampling = 8
    classes = 1
    activation = ''
    

class DEEPLABV3_RESNET152:
    # resnet-152
    block = 'Bottleneck'
    layers = [3, 8, 36, 3]
    replace_stride_with_dilation = [False, True, True]
    strides = [2, 2, 4]
    # deelabv3
    decoder_channels = N_DC*3
    atrous_rates = (6, 12, 24)
    # segmentation head
    upsampling = 8
    classes = 1
    activation = ''
    

rng = jax.random.PRNGKey(CFG.seed)

def seed_everything(seed: int):  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(CFG.seed)






class PAN_RESNET18:
    # resnet-18
    block = 'BasicBlock'
    layers = [2, 2, 2, 2]
    replace_stride_with_dilation = [False, False, False]
    strides = [2, 2, 2]
    # pan
    decoder_channels = N_DC/4
    # segmentation head
    upsampling = 4
    classes = 1
    activation = ''
    

class PAN_RESNET34:
    # resnet-34
    block = 'BasicBlock'
    layers = [3, 4, 6, 3]
    replace_stride_with_dilation = [False, False, False]
    strides = [2, 2, 2]
    # pan
    decoder_channels = 32
    # segmentation head
    upsampling = 4
    classes = 1
    activation = ''
    

class PAN_RESNET50:
    # resnet-50
    block = 'Bottleneck'
    layers = [3, 4, 6, 3]
    replace_stride_with_dilation = [False, False, True]
    strides = [2, 2, 2]
    # pan
    decoder_channels = 32
    # segmentation head
    upsampling = 4
    classes = 1
    activation = ''
    

class PAN_RESNET101:
    # resnet-101
    block = 'Bottleneck'
    layers = [3, 4, 23, 3]
    replace_stride_with_dilation = [False, False, True]
    strides = [2, 2, 2]
    # pan
    decoder_channels = N_DC/16
    # segmentation head
    upsampling = 4
    classes = 1
    activation = ''
    

class PAN_RESNET152:
    # resnet-152
    block = 'Bottleneck'
    layers = [3, 8, 36, 3]
    replace_stride_with_dilation = [False, False, True]
    strides = [2, 2, 2]
    # pan
    decoder_channels = N_DC/16
    # segmentation head
    upsampling = 4
    classes = 1
    activation = ''
class LOSS:
    # see more detailed explanation in 'Train / eval step, loss function, metrics' section
    # tversky loss
    alpha = 0.3
    beta = 0.7
    # focal-tversky loss
    gamma = 1.0
    # weights for mIOU loss, delta - IOU class 0, theta - IOU class 1
    delta = 0.2
    theta = 0.8
    # combination parameter for focal-tversky loss and mIOU
    mu = 0.5
    # smooth parameter to prevent zero-division
    smooth = 1e-8


def conv3x3(out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv(features=out_planes,
                   kernel_size=(3, 3),
                   strides=stride,
                   padding=dilation,
                   feature_group_count=groups,
                   use_bias=False,
                   kernel_dilation=dilation)


def conv1x1(out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv(features=out_planes,
                   kernel_size=(1, 1),
                   strides=stride,
                   padding=0,
                   use_bias=False)
    
class BasicBlock(nn.Module):
    planes: int
    stride: int = 1
    downsample: Any = None
    groups: int = 1
    base_width: int = 64
    dilation: int = 1
    expansion: int = 1
    train: bool = True
    
    def setup(self):
        if self.groups != 1 or self.base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if self.dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        
        self.conv1 = conv3x3(self.planes, self.stride)
        self.conv2 = conv3x3(self.planes)
        
        self.bn1 = nn.BatchNorm(use_running_average=not self.train)
        self.bn2 = nn.BatchNorm(use_running_average=not self.train)
        self.bn3 = nn.BatchNorm(use_running_average=not self.train)
    
    @nn.compact
    def __call__(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.activation.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn3(identity)
            
        out += identity
        out = nn.activation.relu(out)
        
        return out
    
class Bottleneck(nn.Module):
    planes: int
    stride: int = 1
    downsample: Any = None
    groups: int = 1
    base_width: int = 64
    dilation: int = 1
    expansion: int = 4
    train: bool = True
        
    def setup(self):
        width = int(self.planes * (self.base_width / 64.0)) * self.groups
        
        self.conv1 = conv1x1(width)
        self.conv2 = conv3x3(width, self.stride, self.groups, self.dilation)
        self.conv3 = conv1x1(self.planes * self.expansion)
        
        self.bn1 = nn.BatchNorm(use_running_average=not self.train)
        self.bn2 = nn.BatchNorm(use_running_average=not self.train)
        self.bn3 = nn.BatchNorm(use_running_average=not self.train)
        self.bn4 = nn.BatchNorm(use_running_average=not self.train)
        
    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.activation.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.activation.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn4(identity)

        out += identity
        out = nn.activation.relu(out)

        return out
        
class ResNetModule(nn.Module):
    block: Type[Union[BasicBlock, Bottleneck]]
    layers: List[int]
    groups: int = 1
    width_per_group: int = 64
    strides: Optional[List[int]] = (2, 2, 2)
    replace_stride_with_dilation: Optional[List[bool]] = None
    train: bool = True
    
    def setup(self):
        self.repl = self.replace_stride_with_dilation
        
        if self.repl is None:
            self.repl = [False, False, False]
        
        if len(self.repl) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {self.repl}"
            )
        
        self.inplanes = 64
        self.dilation = 1
        self.base_width = self.width_per_group
        
        self.conv1 = nn.Conv(self.inplanes, kernel_size=(7, 7), strides=2, padding=3, use_bias=False)
        self.norm = nn.BatchNorm(use_running_average=not self.train)
        
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=self.strides[0], dilate=self.repl[0])
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=self.strides[1], dilate=self.repl[1])
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=self.strides[2], dilate=self.repl[2])

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(planes * block.expansion, stride)

        layers = []
        layers.append(block(planes, 
                            stride, 
                            downsample, 
                            self.groups, 
                            self.base_width, 
                            previous_dilation,
                            train=self.train))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    train=self.train
                )
            )

        return layers
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = nn.activation.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1)))

        for blocks in self.layer1:
            x = blocks(x)
        f1 = x
        
        for blocks in self.layer2:
            x = blocks(x)
        f2 = x
        
        for blocks in self.layer3:
            x = blocks(x)
        f3 = x
        
        for blocks in self.layer4:
            x = blocks(x)
        f4 = x

        return [f1, f2, f3, f4]
    
class ResNet(nn.Module):
    block: Type[Union[BasicBlock, Bottleneck]]
    layers: List[int]
    groups: int = 1
    width_per_group: int = 64
    strides: Optional[List[int]] = (2, 2, 2)
    replace_stride_with_dilation: Optional[List[bool]] = None
        
    @nn.compact
    def __call__(self, x, train: bool):
        x = ResNetModule(block=self.block,
                         layers=self.layers,
                         groups=self.groups,
                         width_per_group=self.width_per_group,
                         strides=self.strides,
                         replace_stride_with_dilation=self.replace_stride_with_dilation,
                         train=train)(x)
        return x       
    
class ASPPConv(nn.Module):
    out_channels: int
    dilation: int
    train: bool = True
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, 
                    kernel_size=(3, 3), 
                    strides=(1, 1), 
                    padding=self.dilation,
                    kernel_dilation=self.dilation,
                    use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = nn.activation.relu(x)
        return x
    
    
class ASPPPooling(nn.Module):
    out_channels: int
    train: bool = True
        
    @nn.compact
    def __call__(self, x):
        shape = x.shape
        size = x.shape[1], x.shape[2]
        
        x = nn.avg_pool(x, window_shape=size, strides=size, padding=((0, 0), (0, 0)))
        x = nn.Conv(self.out_channels, 
                    kernel_size=(1, 1), 
                    strides=(1, 1), 
                    padding=0,
                    use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not self.train)(x)
        x = nn.activation.relu(x)
        x = jax.image.resize(x, shape=(shape[0], shape[1], shape[2], x.shape[3]), method='bilinear')
        return x
    
class ASPP(nn.Module):
    out_channels: int = 256
    atrous_rates: List[int] = (12, 24, 36)
    separable: bool = False
    train: bool = True
        
    def setup(self):
        self.mod1 = nn.Sequential([
            conv1x1(self.out_channels),
            nn.BatchNorm(use_running_average=not self.train),
        ])
        
        rate1, rate2, rate3 = self.atrous_rates
        
        self.aspp1 = ASPPConv(out_channels=self.out_channels, dilation=rate1, train=self.train)
        self.aspp2 = ASPPConv(out_channels=self.out_channels, dilation=rate2, train=self.train)
        self.aspp3 = ASPPConv(out_channels=self.out_channels, dilation=rate3, train=self.train)
        self.aspp_pool = ASPPPooling(out_channels=self.out_channels, train=self.train)
        
        self.project = nn.Sequential([
            conv1x1(self.out_channels),
            nn.BatchNorm(use_running_average=not self.train),
        ])
        
        self.modules = [self.aspp1, self.aspp2, self.aspp3, self.aspp_pool]
          
    def __call__(self, x):       
        res = [nn.activation.relu(self.mod1(x))]
        
        for mod in self.modules:
            res.append(mod(x))

        out = jnp.concatenate(res, axis=3)
        prj = nn.activation.relu(self.project(out))
        return prj
    
class DeepLabV3Decoder(nn.Module):
    out_channels: int = 256
    atrous_rates: List[int] = (12, 24, 36)
    
    @nn.compact
    def __call__(self, features, train: bool):
        x = features[-1]
        x = ASPP(out_channels=self.out_channels, 
                 atrous_rates=self.atrous_rates,
                 train=train)(x)
        x = nn.Conv(self.out_channels, 
                    kernel_size=(3, 3), 
                    strides=(1, 1), 
                    padding=(1, 1), 
                    use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.activation.relu(x)
        return x
    
class DeepLabV3(nn.Module):
    block: str
    layers: List[int]
    decoder_channels: int = 256
    atrous_rates: List[int] = (12, 24, 36)
    classes: int = 1
    upsampling: int = 8
    activation: str = ''
    strides: Optional[List[int]] = (2, 2, 2)
    replace_stride_with_dilation: Optional[List[bool]] = None
        
    def setup(self):
        block = BasicBlock if self.block == 'BasicBlock' else Bottleneck
        self.encoder = ResNet(block=block, 
                              layers=self.layers,
                              strides=self.strides,
                              replace_stride_with_dilation=self.replace_stride_with_dilation)
        
        self.decoder = DeepLabV3Decoder(out_channels=self.decoder_channels, 
                                        atrous_rates=self.atrous_rates)
        
        self.segmentation_head = SegmentationHead(out_channels=self.classes,
                                                  activation=self.activation,
                                                  upsampling=self.upsampling)
        
    def __call__(self, x, train: bool):
        features = self.encoder(x, train)
        decoder_output = self.decoder(features, train)
        masks = self.segmentation_head(decoder_output)
        return masks
    
    
class ConvBnRelu(nn.Module):
    out_channels: int
    kernel_size: List[int]
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    add_relu: bool = True
    interpolate: bool = False
    train: bool = True
        
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    padding=self.padding,
                    kernel_dilation=self.dilation,
                    feature_group_count=self.groups,
                    use_bias=self.bias)(x)
        x = nn.BatchNorm(use_running_average=not self.train)(x)
        if self.add_relu:
            x = nn.activation.relu(x)
        if self.interpolate:
            b, h, w, c = x.shape 
            x = jax.image.resize(x, shape=(b, h * 2, w * 2, c), method='bilinear')
        return x
    
class FPABlock(nn.Module):
    out_channels: int
    train: bool = True
        
    def setup(self):
        # global pooling branch
        self.branch1 = ConvBnRelu(out_channels=self.out_channels, 
                                  kernel_size=(1, 1), 
                                  stride=1, 
                                  padding=0, 
                                  train=self.train)
        
        # middle branch
        self.mid = ConvBnRelu(out_channels=self.out_channels, 
                              kernel_size=(1, 1), 
                              stride=1, 
                              padding=0, 
                              train=self.train)
        
        self.down1 = ConvBnRelu(out_channels=1, 
                                kernel_size=(7, 7), 
                                stride=1, 
                                padding=3,
                                train=self.train)
        
        self.down2 = ConvBnRelu(out_channels=1, 
                                kernel_size=(5, 5), 
                                stride=1, 
                                padding=2,
                                train=self.train)
        
        self.down3 = nn.Sequential([
            ConvBnRelu(out_channels=1, kernel_size=(3, 3), stride=1, padding=1, train=self.train),
            ConvBnRelu(out_channels=1, kernel_size=(3, 3), stride=1, padding=1, train=self.train),
        ])
        
        self.conv2 = ConvBnRelu(out_channels=1, 
                                kernel_size=(5, 5), 
                                stride=1, 
                                padding=2,
                                train=self.train)
        
        self.conv1 = ConvBnRelu(out_channels=1, 
                                kernel_size=(7, 7), 
                                stride=1, 
                                padding=3,
                                train=self.train)

    def __call__(self, x):
        size = x.shape[1], x.shape[2]
        b, h, w, c = x.shape
        
        b1 = nn.avg_pool(x, window_shape=size, strides=size, padding=((0, 0), (0, 0)))
        b1 = self.branch1(b1)
        b1 = jax.image.resize(b1, shape=(b, h, w, b1.shape[3]), method='bilinear')
    
        mid = self.mid(x)
        
        x1 = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x1 = self.down1(x1)
        
        x2 = nn.max_pool(x1, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x2 = self.down2(x2)
        
        x3 = nn.max_pool(x2, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x3 = self.down3(x3)
        x3 = jax.image.resize(x3, shape=(b, h // 4, w // 4, x3.shape[3]), method='bilinear')
        
        x2 = self.conv2(x2)
        x = x2 + x3
        x = jax.image.resize(x, shape=(b, h // 2, w // 2, x.shape[3]), method='bilinear')
        
        x1 = self.conv1(x1)
        x = x + x1
        x = jax.image.resize(x, shape=(b, h, w, x.shape[3]), method='bilinear')
        
        x = jax.lax.mul(x, mid)
        x = x + b1
        
        return x
    
class GAUBlock(nn.Module):
    out_channels: int
    train: bool = True
        
    @nn.compact
    def __call__(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        xsize = x.shape[1], x.shape[2]
        bx, hx, wx, cx = x.shape
        
        ysize = y.shape[1], y.shape[2]
        by, hy, wy, cy = y.shape
        
        y_up = jax.image.resize(y, shape=(bx, hx, wx, y.shape[3]), method='bilinear')
        x = ConvBnRelu(out_channels=self.out_channels, kernel_size=(3, 3), padding=1, train=self.train)(x)
        
        y = nn.avg_pool(y, window_shape=ysize, strides=ysize, padding=((0, 0), (0, 0)))
        y = ConvBnRelu(out_channels=self.out_channels, kernel_size=(1, 1), add_relu=False, train=self.train)(y)
        y = nn.activation.sigmoid(y)
        
        z = jax.lax.mul(x, y)
        
        return y_up + z
    
class PANDecoder(nn.Module):
    decoder_channels: int
    
    @nn.compact
    def __call__(self, features, train: bool):
        x5 = FPABlock(out_channels=self.decoder_channels, train=train)(features[-1])  # 1/32
        x4 = GAUBlock(out_channels=self.decoder_channels, train=train)(features[-2], x5)  # 1/16
        x3 = GAUBlock(out_channels=self.decoder_channels, train=train)(features[-3], x4)  # 1/8
        x2 = GAUBlock(out_channels=self.decoder_channels, train=train)(features[-4], x3)  # 1/4
        return x2
    
class PAN(nn.Module):
    block: str
    layers: List[int]
    decoder_channels: int = 32
    classes: int = 1
    upsampling: int = 4
    activation: str = ''
    strides: Optional[List[int]] = (2, 2, 2)
    replace_stride_with_dilation: Optional[List[bool]] = None
        
    def setup(self):
        block = BasicBlock if self.block == 'BasicBlock' else Bottleneck
        self.encoder = ResNet(block=block, 
                              layers=self.layers,
                              strides=self.strides,
                              replace_stride_with_dilation=self.replace_stride_with_dilation)
        
        self.decoder = PANDecoder(decoder_channels=self.decoder_channels)
        
        self.segmentation_head = SegmentationHead(out_channels=self.classes,
                                                  activation=self.activation,
                                                  upsampling=self.upsampling)
        
    def __call__(self, x, train: bool):
        features = self.encoder(x, train)
        decoder_output = self.decoder(features, train)
        masks = self.segmentation_head(decoder_output)
        return masks
    
    
class SegmentationHead(nn.Module):
    out_channels: int
    activation: str = ''
    upsampling: int = 8
        
    @nn.compact
    def __call__(self, x):
        ks = 3
        x = nn.Conv(self.out_channels, 
                    kernel_size=(ks, ks), 
                    strides=(1, 1), 
                    padding=ks // 2)(x)
        
        if self.upsampling > 1:
            b, h, w, c = x.shape
            x = jax.image.resize(x, shape=(b, h * self.upsampling, w * self.upsampling, c), method='bilinear')
        
        if len(self.activation) > 0:
            x = getattr(nn.activation, self.activation)(x)
        return x
    
    
def get_model(name: str, only_dct: bool = False, dct: Dict[str, Any] = None):
    res = [None, None]
    
    if name == 'DeepLabV3_ResNet18':
        res[0] = class_to_dct(DEEPLABV3_RESNET18) if not dct else dct
        if not only_dct:
            res[1] = DeepLabV3(**res[0])
        return res
    elif name == 'DeepLabV3_ResNet34':
        res[0] = class_to_dct(DEEPLABV3_RESNET34) if not dct else dct
        if not only_dct:
            res[1] = DeepLabV3(**res[0])
        return res
    elif name == 'DeepLabV3_ResNet50':
        res[0] = class_to_dct(DEEPLABV3_RESNET50) if not dct else dct
        if not only_dct:
            res[1] = DeepLabV3(**res[0])
        return res
    elif name == 'DeepLabV3_ResNet101':
        res[0] = class_to_dct(DEEPLABV3_RESNET101) if not dct else dct
        if not only_dct:
            res[1] = DeepLabV3(**res[0])
        return res
    elif name == 'DeepLabV3_ResNet152':
        res[0] = class_to_dct(DEEPLABV3_RESNET152) if not dct else dct
        if not only_dct:
            res[1] = DeepLabV3(**res[0])
        return res
    elif name == 'PAN_ResNet18':
        res[0] = class_to_dct(PAN_RESNET18) if not dct else dct
        if not only_dct:
            res[1] = PAN(**res[0])
        return res
    elif name == 'PAN_ResNet34':
        res[0] = class_to_dct(PAN_RESNET34) if not dct else dct
        if not only_dct:
            res[1] = PAN(**res[0])
        return res
    elif name == 'PAN_ResNet50':
        res[0] = class_to_dct(PAN_RESNET50) if not dct else dct
        if not only_dct:
            res[1] = PAN(**res[0])
        return res
    elif name == 'PAN_ResNet101':
        res[0] = class_to_dct(PAN_RESNET101) if not dct else dct
        if not only_dct:
            res[1] = PAN(**res[0])
        return res
    elif name == 'PAN_ResNet152':
        res[0] = class_to_dct(PAN_RESNET152) if not dct else dct
        if not only_dct:
            res[1] = PAN(**res[0])
        return res

    return None    




def read_img(path: str, channels: List[int] = None):
    if channels is None:
        img = rasterio.open(path).read().transpose((1, 2, 0))
    else:
        img = rasterio.open(path).read(channels).transpose((1, 2, 0))    
    #img = np.float32(img) / 65535
    for d in channels:
        img[:,:,d] = rescale( img[:,:,d])
    return img

# read mask into np.ndarray
def read_mask(path: str):
    mask = rasterio.open(path).read().transpose((1, 2, 0)) 
    mask = np.int32(mask)
    return mask

class TrainState(train_state.TrainState):
    batch_stats: Any


@functools.partial(jax.pmap, static_broadcasted_argnums=(1, 2))
def create_train_state(rng: Any, lr_function: Any, shape: List[int]):
    _, model = get_model(CFG.model)
    
    # turn on train mode, jnp.ones(shape) - dummy input with correct shape
    variables = model.init(rng, jnp.ones(shape), train=True)
    params = variables['params']
    batch_stats = variables['batch_stats']
    tx = getattr(optax, CFG.optimizer)(lr_function, **CFG.optimizer_params)
    
    return TrainState.create(apply_fn=model.apply, 
                             params=params,
                             batch_stats=batch_stats,
                             tx=tx)


def create_learning_rate_fn(ttl_iters: int):
    scheduler = getattr(optax, CFG.scheduler)
    
    for key in CFG.ttl_iters_keys:
        if key in CFG.scheduler_params.keys():
            CFG.scheduler_params[key] = ttl_iters
    
    return scheduler(**CFG.scheduler_params)



def custom_loss(logits: Any, labels: Any):
    """
    Args:
        logits - raw output from model
        labels - real labels
    """
    alpha = LOSS.alpha
    beta = LOSS.beta
    gamma = LOSS.gamma
    delta = LOSS.delta
    theta = LOSS.theta
    mu = LOSS.mu
    smooth = LOSS.smooth

    # scale logits into [0, 1] interval
    preds = nn.activation.sigmoid(logits)

    # reshape into 1-D
    flat_logits = jnp.ravel(preds)
    flat_labels = jnp.ravel(labels)

    # calculate true-positives, false-positives, false-negatives
    tp = jnp.sum(flat_logits * flat_labels)
    fp = jnp.sum(flat_logits * (1 - flat_labels))
    fn = jnp.sum((1 - flat_logits) * flat_labels)

    # iou for class 0
    union0 = jnp.clip((1 - preds) + (1 - labels), a_min=0, a_max=1)
    intersection0 = (1 - preds) * (1 - labels)
    iou0 = jnp.sum(intersection0) / (jnp.sum(union0) + smooth)

    # iou for class 1
    union1 = jnp.clip(preds + labels, a_min=0, a_max=1)
    intersection1 = preds * labels
    iou1 = jnp.sum(intersection1) / (jnp.sum(union1) + smooth)

    # focal-tversky loss
    tversky_loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    tversky_focal_loss = tversky_loss**gamma

    # miou loss
    miou_loss = (1 - iou0) * delta + (1 - iou1) * theta
    
    # focal-tversky and moiu combination
    loss = mu * tversky_focal_loss + (1 - mu) * miou_loss
    
    return loss

def custom_loss(logits: Any, labels: Any):
    """
    Args:
        logits - raw output from model
        labels - real labels
    """
    alpha = LOSS.alpha
    beta = LOSS.beta
    gamma = LOSS.gamma
    delta = LOSS.delta
    theta = LOSS.theta
    mu = LOSS.mu
    smooth = LOSS.smooth

    # scale logits into [0, 1] interval
    preds = nn.activation.sigmoid(logits)

    # reshape into 1-D
    flat_logits = jnp.ravel(preds)
    flat_labels = jnp.ravel(labels)

    # calculate true-positives, false-positives, false-negatives
    tp = jnp.sum(flat_logits * flat_labels)
    fp = jnp.sum(flat_logits * (1 - flat_labels))
    fn = jnp.sum((1 - flat_logits) * flat_labels)

    # iou for class 0
    union0 = jnp.clip((1 - preds) + (1 - labels), a_min=0, a_max=1)
    intersection0 = (1 - preds) * (1 - labels)
    iou0 = jnp.sum(intersection0) / (jnp.sum(union0) + smooth)

    # iou for class 1
    union1 = jnp.clip(preds + labels, a_min=0, a_max=1)
    intersection1 = preds * labels
    iou1 = jnp.sum(intersection1) / (jnp.sum(union1) + smooth)

    # focal-tversky loss
    tversky_loss = 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    tversky_focal_loss = tversky_loss**gamma

    # miou loss
    miou_loss = (1 - iou0) * delta + (1 - iou1) * theta
    
    # focal-tversky and moiu combination
    loss = mu * tversky_focal_loss + (1 - mu) * miou_loss
    
    return loss

@functools.partial(jax.pmap, axis_name='batch')
def compute_metrics(state: Any, image: Any, mask: Any):
    logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats},
                            image,
                            train=False)
    preds = nn.activation.sigmoid(logits) > 0.5
    labels = mask
    
    smooth = LOSS.smooth
    
    tp = jnp.sum((preds == 1) * (labels == 1))
    fp = jnp.sum((preds == 1) * (labels == 0))
    tn = jnp.sum((preds == 0) * (labels == 0))
    fn = jnp.sum((preds == 0) * (labels == 1))
    
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    
    union0 = jnp.clip((1 - preds) + (1 - labels), a_min=0, a_max=1)
    intersection0 = (1 - preds) * (1 - labels)
    iou0 = jnp.sum(intersection0) / (jnp.sum(union0) + smooth)
    
    union1 = jnp.clip(preds + labels, a_min=0, a_max=1)
    intersection1 = preds * labels
    iou1 = jnp.sum(intersection1) / (jnp.sum(union1) + smooth)
    
    miou = (iou0 + iou1) / 2
    return precision, recall, iou0, iou1, miou



def train_epoch(state: Any, train_loader: Any, epoch: int, lr_fn: Any):
    pbar = tqdm(train_loader)
    pbar.set_description(f'train epoch: {epoch + 1}')

    epoch_loss = 0.0
    
    for step, batch in enumerate(pbar):
        image, mask = batch
        # https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#common-utilities
        image = shard(jnp.array(image, dtype=jnp.float32))
        mask = shard(jnp.array(mask, dtype=jnp.int32))
        
        state, loss = train_step(state, image, mask)
        
        if USE_ORBAX_WITH_FLAX:
            # we use 8 cores at the same time, so we have 8 copies of state, use unreplicate to get a single copy
            lr = lr_fn(jax_utils.unreplicate(state).step)
        else:
            lr = lr_fn(state.step)[0]

        # we use 8 cores at the same time, so we have 8 copies of loss, use unreplicate to get a single copy
        epoch_loss += jax_utils.unreplicate(loss)
        pbar.set_description(f'train epoch: {epoch + 1} loss: {(epoch_loss / (step + 1)):.3f} lr: {lr:.6f}')
    
    return state
def test_epoch(state: Any, test_loader: Any, epoch: int):
    pbar = tqdm(test_loader)
    pbar.set_description(f'test epoch: {epoch + 1}')
    
    num = len(test_loader)
    
    epoch_loss = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0
    epoch_iou0 = 0.0
    epoch_iou1 = 0.0
    epoch_miou = 0.0
    
    for step, batch in enumerate(pbar):
        image, mask = batch
        # https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#common-utilities
        image = shard(jnp.array(image, dtype=jnp.float32))
        mask = shard(jnp.array(mask, dtype=jnp.int32))
        
        loss = eval_step(state, image, mask)
        precision, recall, iou0, iou1, miou = compute_metrics(state, image, mask)
        
        # we use 8 cores at the same time, so we have 8 copies of loss, precision, etc, use unreplicate to get a single copy
        epoch_loss += jax_utils.unreplicate(loss)
        epoch_precision += jax_utils.unreplicate(precision)
        epoch_recall += jax_utils.unreplicate(recall)
        epoch_iou0 += jax_utils.unreplicate(iou0)
        epoch_iou1 += jax_utils.unreplicate(iou1)
        epoch_miou += jax_utils.unreplicate(miou)
        
        pbar_str = f'test epoch: {epoch + 1} '
        pbar_str += f'loss: {(epoch_loss / (step + 1)):.3f} '
        pbar_str += f'precision: {(epoch_precision / (step + 1)):.3f} '
        pbar_str += f'recall: {(epoch_recall / (step + 1)):.3f} '
        pbar_str += f'iou0: {(epoch_iou0 / (step + 1)):.3f} '
        pbar_str += f'iou1: {(epoch_iou1 / (step + 1)):.3f} '
        pbar_str += f'miou: {(epoch_miou / (step + 1)):.3f}'
        
        pbar.set_description(pbar_str)
        
    epoch_loss /= num
    epoch_precision /= num
    epoch_recall /= num
    epoch_iou0 /= num
    epoch_iou1 /= num
    epoch_miou /= num
    
    metrics = {
        'loss': epoch_loss,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'iou0': epoch_iou0,
        'iou1': epoch_iou1,
        'miou': epoch_miou
    }
        
    return metrics

def class_to_dct(cls: Any):
    # converts any Python class to dictionary, without underscored methods
    dct = {}
    for attr in dir(cls):
        if attr[:2] != '__' and attr[-2:] != '__':
            dct[attr] = getattr(cls, attr)
    return dct


def best_fn(metrics: Dict[str, float]):
    # these metrics look the most relevant to me, this function will be used to determine the best checkpoint
    return metrics['precision'] + metrics['recall'] + metrics['iou1'] + metrics['miou']


class Dataset(Dataset):
    """
    Args:
        df - pandas.DataFrame with columns (image, mask, region)
        transfrom - torchvision.transforms for image augmentations
        inference - use the test mode
        channels - list of channels that to be used
    """
    def __init__(self, df: Any, transform: Any = None, inference: bool = False, channels: List[int] = None, 
                 imgdir = '', labeldir=None):      
        self.df = df
        self.transform = transform
        self.inference = inference
        self.channels = channels
        self.imgdir = imgdir 
        self.labeldir = labeldir
            
    def __len__(self):
        return len(self.df)
    
    def _read_img(self, path: str, channels: List[int]):
        if channels:
            img = rasterio.open( self.imgdir + path).read(channels).transpose((1, 2, 0))
        else:
            img = rasterio.open( self.imgdir + path).read().transpose((1, 2, 0))
        # normalize values to [0, 1] interval
        
        # img = np.float32(img) / 65535
        for d in channels:
            img[:,:,d] = rescale( img[:,:,d])

        return img
    
    def _read_mask(self,  path: str):
        mask = rasterio.open( self.imgdir + path).read().transpose((1, 2, 0))
        mask = np.int32(mask)
        return mask

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = self._read_img(row['tile_id'] + '_satellite.tif', self.channels)
         
        if self.inference:
            return image
        
        mask = self._read_mask(row['tile_id'] + '_kelp.tif')
        if self.transform:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask    


train_df, test_df = train_test_split( meta_pd, test_size=CFG.test_size, random_state=CFG.seed, )
                                    
train_dataset = Dataset(train_df, channels=[1,2,3,4,5,6,7], transform=None,
                        imgdir='/kaggle/input/landsat-30m-350x350-7bands/train_features.tar_MLIC14m/train_satellite/',                        
                        labeldir='/kaggle/input/landsat-30m-350x350-7bands/train_labels.tar_l8u2RP0/train_kelp/')

print( train_dataset.__getitem__(1) )



def main(rng: Any, train_df: Any, test_df: Any):
    # hyperparameters
    epochs = CFG.epochs
    test_size = CFG.test_size
    batch_size = CFG.batch_size
    shape = CFG.shape
    channels = CFG.channels
    num_workers = CFG.num_workers
    
    # define transformations
    transform = albu.Compose([
        albu.Rotate((-45, 45)),
        albu.HorizontalFlip (p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(0.1, 0.1)
    ])

    # create datasets
    train_dataset = Dataset(train_df, channels=channels, transform=transform)
    test_dataset = Dataset(test_df, channels=channels)

    # create dataloaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              drop_last=True, 
                              shuffle=True, 
                              pin_memory=False, 
                              num_workers=num_workers)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             drop_last=True, 
                             shuffle=False, 
                             pin_memory=False, 
                             num_workers=num_workers)
    
    # total steps
    ttl_iters = epochs * len(train_loader)
    
    # create lr_function
    lr_fn = create_learning_rate_fn(ttl_iters)
    
    # init PRNG and state
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(jax.random.split(init_rng, jax.device_count()),
                               lr_fn,
                               shape)
    
    # clear ckpt folder if it exists
    if os.path.exists(CFG.ckpt_path):
        shutil.rmtree(CFG.ckpt_path)

    # get model metadata
    model_dct, _ = get_model(CFG.model, only_dct=True)
    metadata_dct = [class_to_dct(CFG), model_dct, class_to_dct(LOSS)]
    
    # create checkpoint saver
    if USE_ORBAX_WITH_FLAX:
        orbax_checkpointer = PyTreeCheckpointer()
        # we use 8 cores at the same time, so we have 8 copies of state, use unreplicate to get a single copy
        ckpt = {'state': jax_utils.unreplicate(state)}
        for metadata_idx, metadata in enumerate(CFG.metadata):
            ckpt[metadata] = metadata_dct[metadata_idx]
        save_args = orbax_utils.save_args_from_target(ckpt)
        save_dct = {'state': None}
        for metadata_idx, metadata in enumerate(CFG.metadata):
            save_dct[metadata] = metadata_dct[metadata_idx]
        
    else:
        metadata_ckptr = Checkpointer(JsonCheckpointHandler())
        for metadata_idx, metadata in enumerate(CFG.metadata):
            metadata_ckptr.save(CFG.ckpt_path + '/' + metadata, 
                                metadata_dct[metadata_idx], 
                                force=True)
        ckptr = Checkpointer(PyTreeCheckpointHandler())
    
    # train cycle
    best_metrics = 0.0
    for epoch in range(epochs):
        state = train_epoch(state, train_loader, epoch, lr_fn)
        metrics = test_epoch(state, test_loader, epoch)
        # we use 8 cores at the same time, so we have 8 copies state, use unreplicate to get a single copy
        save_state = jax_utils.unreplicate(state)
        
        comb_metrics = best_fn(metrics)
        if USE_ORBAX_WITH_FLAX:
            if comb_metrics > best_metrics:
                if os.path.exists(CFG.ckpt_path):
                    shutil.rmtree(CFG.ckpt_path)
                best_metrics = comb_metrics
                ckpt['state'] = save_state
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(CFG.ckpt_path, ckpt, save_args=save_args)
        else:
            if comb_metrics > best_metrics:
                best_metrics = comb_metrics
                ckptr.save(CFG.ckpt_path + '/ckpt', save_state, force=True)
    
    return state



if not CFG.inference:
    state = main(rng, train_df, test_df)
    model, state, metadata = inference(CFG.ckpt_path, local=CFG.inference)
    models = [(model, state, metadata)]
else:
    if isinstance(CFG.pretrained, str):
        model, state, metadata = inference(CFG.pretrained, local=CFG.inference)
        models = [(model, state, metadata)]
    elif isinstance(CFG.pretrained, (tuple, list)):
        models = []
        for path in CFG.pretrained:
            model, state, metadata = inference(path, local=CFG.inference)
            models.append((model, state, metadata))
    else:
        raise TypeError(f'CFG.pretrained has incorrect type {type(CFG.pretrained)}')

