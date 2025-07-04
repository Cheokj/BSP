import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, bn_freeze, return_intermediate=True, proposal_flag=True):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.bn_freeze = bn_freeze
        self.return_intermediate = return_intermediate
        self.proposal_flag = proposal_flag
        
    def forward(self, x, proposal=None):
        input_shape = x.shape[-2:]
        # print('x',x.shape) # 3 513 513

        features, feature_all = self.backbone(x)

        x_token, logit, feature_pre_logit = self.classifier(features, self.proposal_flag, proposal)

        if x_token is not None:
            x = torch.einsum('bnc,bnhw->bchw', x_token, proposal)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        logit = F.interpolate(logit, size=input_shape, mode='bilinear', align_corners=False)
        if self.return_intermediate:
            return x, feature_pre_logit, logit, feature_all, x_token
        else:
            return x
    
    def train(self, mode=True):
        super(_SimpleSegmentationModel, self).train(mode=mode)
        
        if self.bn_freeze:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                        

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out, (x,)

class IntermediateLayerGetter_swin(nn.ModuleDict):

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        super(IntermediateLayerGetter_swin, self).__init__()
        orig_return_layers = return_layers
        self.model = model
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        x = self.model(x)
        out['out'] = x[-1]
        return out, x