import utils

class DEEPLABV3_RESNET18:
    # resnet-18
    block = 'BasicBlock'
    layers = [2, 2, 2, 2]
    replace_stride_with_dilation = [False, False, False]
    strides = [2, 2, 2]
    # deelabv3
    decoder_channels = 256
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
    decoder_channels = 256
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
    decoder_channels = 512
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
    decoder_channels = 512
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
    decoder_channels = 512
    atrous_rates = (6, 12, 24)
    # segmentation head
    upsampling = 8
    classes = 1
    activation = ''
    

class PAN_RESNET18:
    # resnet-18
    block = 'BasicBlock'
    layers = [2, 2, 2, 2]
    replace_stride_with_dilation = [False, False, False]
    strides = [2, 2, 2]
    # pan
    decoder_channels = 32
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
    decoder_channels = 32
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
    decoder_channels = 32
    # segmentation head
    upsampling = 4
    classes = 1
    activation = ''




# ===================================
# Define segmentation head
# ===================================

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

# ===================================
# Define basic block 
# ===================================

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


# ===================================
# Define bottleneck block 
# ===================================

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







# ===================================
# Define ResNet Module
# ===================================
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



# ===================================
# Define ASPP block
# ===================================

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



# ===================================
# Define ASPP Pooling block
# ===================================


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


# ===================================
# Define DeepLabV3Decoder 
# ===================================


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
    
    

# ===================================
# Define DeepLabV3 module
# ===================================

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



# ===================================
# Define ConvBnRelu block
# ===================================

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


# ===================================
# Define FRA block
# ===================================


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


# ===================================
# Define GAU block
# ===================================

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
