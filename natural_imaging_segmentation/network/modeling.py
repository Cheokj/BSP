from .utils import IntermediateLayerGetter, IntermediateLayerGetter_swin
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .backbone import mobilenetv2
from .backbone import swin_transformer

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, bn_freeze):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier, bn_freeze)
    return model

def _segm_swin(name, backbone_name, num_classes, output_stride, pretrained_backbone, bn_freeze):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = swin_transformer._swin_b(pretrained=pretrained_backbone)

    inplanes = 1024 # for swin transformer
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'norm3': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    else:
        raise NotImplementedError
    backbone = IntermediateLayerGetter_swin(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier, bn_freeze)
    return model

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone, bn_freeze):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier, bn_freeze)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, bn_freeze):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, 
                                pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, 
                             pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)
    elif backbone=='swin_transformer':
        model = _segm_swin(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)
    else:
        raise NotImplementedError
        
    return model


# Deeplab v3

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)


# Deeplab v3+

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)

def deeplabv3_swin_transformer(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3+ model with a swin transformer backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
"""
    return _load_model('deeplabv3', 'swin_transformer', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)
