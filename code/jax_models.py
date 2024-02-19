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
