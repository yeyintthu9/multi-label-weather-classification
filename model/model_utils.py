# -*- coding: utf-8 -*-
"""
@Author: Ye Yint Thu
@Email: yeyintthu536@gmail.com
"""

import torch
import torchvision


def freeze_resnet50_layers(
    model: torch.nn.Module, freeze_layer_count: int = 7
) -> torch.nn.Module:
    """Freeze pretrained weights in some layers of resnet50 model

    Args:
        model (torch.nn.Module): Resnet50 nn module with pretrained weights
            from torchvision models
        freeze_layer_count (int, optional): Number of child layers to be frozen.
            Defaults to 7.

    Returns:
        torch.nn.Module: Resnet50 nn module with frozen weights in some layers
    """
    layer_count = 0
    for child in model.children():
        layer_count += 1
        if layer_count <= freeze_layer_count:
            for param in child.parameters():
                param.requires_grad = False
    return model


def freeze_efficientnetv2_stages(
    model: torch.nn.Module, freeze_stage_count: int = 3
) -> torch.nn.Module:
    """Frezee pretrained weights in some stages of efficientnet_v2 model

    Args:
        model (torch.nn.Module): Efficientnetv2_s nn module with pretrained weights
            from torchvision models
        freeze_stage_count (int, optional): Number of conv stages to be frozen.
            Defaults to 3

    Returns:
        torch.nn.Module: Efficientnetv2_s nn module with frozen weights in some stages
    """
    frozen_stages_prefixes = [str(stage) for stage in range(freeze_stage_count + 1)]
    for nn_module in model.children():
        for name, module in nn_module.named_modules():
            if name.split(".")[0] not in frozen_stages_prefixes:
                continue
            for param in module.parameters():
                param.requires_grad = False
    return model


def get_backbone_model(
    backbone_type: str = "resnet50", frozen_layer_count: int = 7
) -> torch.nn.Module:
    """Get backbone model from torchvision models with some frozen pretrained layers
    and remove fc layer

    Args:
        backbone_type (str, optional): Type of backbone model options
            "efficientnetv2_m or resent50". Defaults to "resnet50".
        frozen_layer_count (int, optional): Number of layers or stages to be frozen.
            Defaults to 7.

    Returns:
        torch.nn.Module: Efficientnetv2_m or resnet50 model with some frozen layers
            and no fc layer
    """
    if backbone_type == "resnet50":
        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if frozen_layer_count > 0 else None
        )
        backbone = freeze_resnet50_layers(backbone, frozen_layer_count)
    else:
        backbone = torchvision.models.efficientnet_v2_m(
            weights=torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if frozen_layer_count > 0 else None
        )
        backbone = freeze_efficientnetv2_stages(backbone, frozen_layer_count)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
    return backbone
