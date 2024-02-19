 
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


channels = range(1,8)    
train_df, test_df = train_test_split( subset, test_size=CFG.test_size, random_state=CFG.seed)
train_dataset = Dataset(train_df, channels=channels, transform=None)
test_dataset = Dataset(test_df, channels=channels)
# all: x,y=train_dataset.__getitem__(4310)
x,y=train_dataset.__getitem__(431)

if 0:
    for d in range(7):
        plt.figure()  
        i=x[:,:,d]
        fig=px.imshow( i, title = f'Band{d}: [{i.min()},{i.max()}]' )
        fig.show()    
    plt.figure()
    fig=px.imshow( y.squeeze(), title = 'Kelp mask' )
    fig.show()

subset = meta_df.query( f'kelp_areas > {CFG.frac}')
print(CFG.frac, '>>', subset.shape)
