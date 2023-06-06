# -*- coding: utf-8 -*-
"""
@Author: Ye Yint Thu
@Email: yeyintthu536@gmail.com
"""

import torch
from typing import List
from .model_utils import get_backbone_model


class MultiLabelBinaryClassifier(torch.nn.Module):
    def __init__(
        self,
        classes: List[str],
        backbone_type: str,
        frozen_layers: int = 2,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        """Multi-label binary classifier class for classification problems that
            have image as input and multiple binary values as outputs
        
        Args:
            classes (List[str]): List of classes that define number of binary
                classifiers
            backbone_type (str): Type of backbone for the model
                (resnet50 or efficientnet_v2)
            frozen_layers (int): Number of layers that the pretrained weights
                will be frozen for them
            device (torch.device): Device by which the model's weights will
                be calculated
        """
        self.backbone = get_backbone_model(backbone_type, frozen_layers).to(device)
        self.binary_classifier_heads = {}
        in_features = 2048 if backbone_type == "resnet50" else 1280
        for i, class_ in enumerate(classes):
            self.binary_classifier_heads[class_] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.4),
                torch.nn.Linear(in_features=in_features, out_features=256),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.4),
                torch.nn.Linear(in_features=256, out_features=1),
                torch.nn.Sigmoid(),
            ).to(device)

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        out = {}
        for key, classifier_val in self.binary_classifier_heads.items():
            out[key] = classifier_val(features)
        return out
