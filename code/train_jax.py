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



@functools.partial(jax.pmap, axis_name='batch')
def train_step(state: Any, image: Any, mask: Any):   
    def loss_fn(params: Any):
        logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, 
                                         image,
                                         train=True,
                                         mutable=['batch_stats'])
        labels = mask
        loss = custom_loss(logits, labels)
        return loss, (logits, updates)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    return state, loss


@functools.partial(jax.pmap, axis_name='batch')
def eval_step(state: Any, image: Any, mask: Any):   
    def loss_fn(params: Any):
        logits = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                image,
                                train=False)
        labels = mask
        loss = custom_loss(logits, labels)
        return loss
    
    loss = loss_fn(state.params)
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

 
def main(rng: Any, train_df: Any, test_df: Any):
    # hyperparameters
    epochs = CFG.epochs
    test_size = CFG.test_size
    batch_size = CFG.batch_size
    shape = CFG.shape
    channels = CFG.channels
    num_workers = CFG.num_workers
    
    if 0:
        # define transformations
        transform = albu.Compose([
            albu.Rotate((-45, 45)),
            albu.HorizontalFlip (p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(0.1, 0.1)
        ])

    # create datasets
    train_dataset = Satellite_Dataset(train_df, channels=channels, transform=None )
    test_dataset = Satellite_Dataset(test_df, channels=channels)

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

# retrieve model, state and metadata from stored local checkpoint for inference stage
def inference(path: str, local: bool = False):
    metadata_dct = []
    
    if USE_ORBAX_WITH_FLAX:
        print( path )
        orbax_checkpointer = PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(path)
        restored_state = raw_restored['state']
        
        for metadata in CFG.metadata:
            restored_dct = raw_restored[metadata]
            metadata_dct.append(restored_dct)
    else:
        if len(os.listdir(path)) > 1:
            ckptr = Checkpointer(PyTreeCheckpointHandler())
            restored_state = ckptr.restore(path + '/ckpt')

            metadata_ckptr = Checkpointer(JsonCheckpointHandler())
            metadata_path = CFG.pretrained if local else CFG.ckpt_path

            for metadata in CFG.metadata:
                restored_dct = metadata_ckptr.restore(metadata_path + '/' + metadata)
                metadata_dct.append(restored_dct)
        else:
            orbax_checkpointer = PyTreeCheckpointer()
            raw_restored = orbax_checkpointer.restore(path)
            restored_state = raw_restored['state']
        
            for metadata in CFG.metadata:
                restored_dct = raw_restored[metadata]
                metadata_dct.append(restored_dct)

    config_dct = metadata_dct[0]
    model_dct = metadata_dct[1]
    
    _, model = get_model(config_dct['model'], dct=model_dct)
    return model, restored_state, metadata_dct


# predict mask using loaded model, state and input image
def predict(model: Any, state: Any, img: Any):
    jnp_img = jnp.array(img, dtype=jnp.float32)[jnp.newaxis, :, :, :]
    logits = model.apply({'params': state['params'], 
                          'batch_stats': state['batch_stats']},
                         jnp_img,
                         train=False)
    preds = nn.activation.sigmoid(logits) > 0.5
    return preds

# vizualize results
def vizualize(model: Any, state: Any, img: Any, mask: Any, name: str):
    # img - array from read_img function
    preds = predict(model, state, img)
        
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(name, x=0.5, y=0.785)

    _ = axs[0].imshow(preds[0])
    _ = axs[1].imshow(mask)
    
    
