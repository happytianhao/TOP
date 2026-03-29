# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from typing import Dict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList
from mmengine.model import BaseModel
from mmaction.models import BaseRecognizer, BaseHead

from .dsta_src import DSTA 


@MODELS.register_module()
class DSTAHead(BaseHead):
    """Adapter to wrap DSTA model into MMAction2's BaseHead.

    Expected inputs for training (loss mode):
        - inputs: torch.Tensor, feature tensor from backbone
        - data_samples: SampleList containing labels and metadata

    For inference (predict mode):
        - inputs: torch.Tensor, feature tensor from backbone
        - data_samples: SampleList containing metadata
    """

    def __init__(
        self,
        in_channels: int = 2048,
        x_dim: int = 4096,
        h_dim: int = 512,
        z_dim: int = 256,
        n_layers: int = 1,
        n_obj: int = 19,
        n_frames: int = 30,
        fps: float = 30.0,
        with_saa: bool = True,
    ) -> None:
        super().__init__(num_classes=2, in_channels=in_channels, loss_cls=dict(type="CrossEntropyLoss"))

        self.n_obj = n_obj
        
        # Global average pooling layer
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection layer to convert backbone features to DSTA input
        self.feature_proj = nn.Linear(in_channels, x_dim)
        
        # Initialize DSTA model with proper parameters
        self.model = DSTA(
            x_dim=x_dim,
            h_dim=h_dim,
            z_dim=z_dim,
            n_layers=n_layers,
            n_obj=n_obj,
            n_frames=n_frames,
            fps=fps,
            with_saa=with_saa
        )
        self.epoch = None

    def forward(self, x: torch.Tensor, num_segs: int, **kwargs):
        """Forward function that processes features and returns classification scores.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, T, C, H, W) or (B*T, C, H, W)
                where B is batch size, T is number of clips/frames
            num_segs (int): Number of segments/frames per batch (T)
            
        Returns:
            torch.Tensor: Classification scores with shape (B*T, num_classes)
        """
        # Extract features
        features, B, T = self._extract_features(x, num_segs=num_segs)
        
        # Create dummy labels and toa for forward pass (needed by DSTA)
        labels = torch.zeros((B, 2), device=features.device, dtype=torch.float32)
        labels[:, 0] = 1  # No accident
        toa = torch.full((B,), T, device=features.device, dtype=torch.float32)
        
        # Run forward pass through DSTA model
        losses_dict, all_outputs, all_hidden, all_alphas = self.model(
            features, labels, toa, hidden_in=None, nbatch=B, testing=True
        )
        
        # Stack outputs and reshape to (B*T, num_classes) for compatibility
        # all_outputs is list of T tensors, each (B, num_classes)
        cls_scores = torch.stack(all_outputs, dim=1)  # (B, T, num_classes)
        cls_scores = cls_scores.reshape(B * T, self.num_classes)  # (B*T, num_classes)
        
        return cls_scores
    
    def _extract_features(self, x: torch.Tensor, num_segs: int):
        """Extract and process features for DSTA model.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, T, C, H, W) or (B*T, C, H, W)
            num_segs (int): Number of segments/frames per batch (T)
            
        Returns:
            tuple: (processed_features, B, T) where features has shape (B, T, 1+n_obj, x_dim)
        """
        if x.dim() == 5:
            # Shape: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            # Reshape to (B*T, C, H, W) for processing
            x = x.reshape(B * T, C, H, W)
        else:
            # Shape: (B*T, C, H, W)
            # Use provided num_segs as T
            T = num_segs
            B = x.shape[0] // T
        
        # Apply global average pooling: (B*T, C, H, W) -> (B*T, C)
        features = self.avg_pool2d(x).flatten(1)  # (B*T, C)
        
        # Project features to DSTA input dimension
        features = self.feature_proj(features)  # (B*T, x_dim)
        
        # Reshape to (B, T, x_dim) and add dummy object dimension
        features = features.reshape(B, T, -1)  # (B, T, x_dim)
        # Add object dimension: (B, T, 1+n_obj, x_dim)
        # Repeat the feature 20 times (1 + n_obj)
        features = features.unsqueeze(2)  # (B, T, 1, x_dim)
        features = features.repeat(1, 1, self.n_obj + 1, 1)  # (B, T, 20, x_dim)
        
        return features, B, T
    
    def loss(self, feats: torch.Tensor, data_samples: SampleList, num_segs: int, **kwargs) -> dict:
        """Calculate losses.
        
        Args:
            feats (torch.Tensor): Feature tensor with shape (B, T, C, H, W) or (B*T, C, H, W)
            data_samples (SampleList): List of data samples containing labels
            num_segs (int): Number of segments/frames per batch (T)
            
        Returns:
            dict: Dictionary of losses
        """
        # Process features using the extraction helper
        features, B, T = self._extract_features(feats, num_segs=num_segs)
        
        # Extract labels from data_samples
        # Create labels in format B*2: [1-have_accident, have_accident]
        labels = []
        toa_list = []
        for sample in data_samples:
            have_accident = int(sample.have_accident)
            label = [1 - have_accident, have_accident]
            labels.append(label)
            # Fixed: toa is always set to T
            toa = T
            toa_list.append(toa)
        
        labels = torch.tensor(labels, device=feats.device, dtype=torch.float32)
        toa = torch.tensor(toa_list, device=feats.device, dtype=torch.float32)
        
        # Call the DSTA model's forward method
        # Returns: losses dict, all_outputs, all_hidden, all_alphas
        losses_dict, all_outputs, all_hidden, all_alphas = self.model(
            features, labels, toa, hidden_in=None, nbatch=B, testing=False
        )
        
        # IMPORTANT: MMEngine only sums dict items with 'loss' in the key name for backward
        # The DSTA model returns: 'cross_entropy', 'auxloss', 'total_loss'
        # Problem: 'cross_entropy' doesn't contain 'loss' so it won't be used!
        # Solution: Rename keys to ensure both losses are used correctly
        processed_losses = {}
        
        # Rename 'cross_entropy' to 'loss_ce' so MMEngine will use it
        if 'cross_entropy' in losses_dict:
            processed_losses['loss_ce'] = losses_dict['cross_entropy']
        
        # Rename 'auxloss' to 'loss_aux' for consistency
        if 'auxloss' in losses_dict:
            processed_losses['loss_aux'] = losses_dict['auxloss']
        
        return processed_losses
    
    def predict(self, feats: torch.Tensor, data_samples: SampleList, num_segs: int, **kwargs) -> SampleList:
        """Predict results.
        
        Args:
            feats (torch.Tensor): Feature tensor with shape (B, T, C, H, W) or (B*T, C, H, W)
            data_samples (SampleList): List of data samples
            num_segs (int): Number of segments/frames per batch (T)
            
        Returns:
            SampleList: List of data samples with predictions
        """
        # Process features using the extraction helper
        features, B, T = self._extract_features(feats, num_segs=num_segs)
        
        # For inference, we still need labels and toa to run the model
        labels = []
        toa_list = []
        for sample in data_samples:
            have_accident = int(sample.have_accident)
            label = [1 - have_accident, have_accident]
            labels.append(label)
            # Fixed: toa is always set to T
            toa = T
            toa_list.append(toa)
        
        labels = torch.tensor(labels, device=feats.device, dtype=torch.float32)
        toa = torch.tensor(toa_list, device=feats.device, dtype=torch.float32)
        
        # Run forward pass
        with torch.no_grad():
            losses_dict, all_outputs, all_hidden, all_alphas = self.model(
                features, labels, toa, hidden_in=None, nbatch=B, testing=True
            )
        
        # Process predictions and attach to data_samples
        # all_outputs is a list of length T, each tensor has shape (B, 2)
        # Stack all outputs to (B, T, 2) and apply softmax
        all_outputs = torch.stack(all_outputs, dim=1)  # (B, T, 2)
        
        # Update data_samples with predictions
        for i, data_sample in enumerate(data_samples):
            # Apply softmax and take the second value (accident probability)
            data_sample.set_pred_score(F.softmax(all_outputs[i], dim=-1)[:, 1].detach())  # Shape: (T,)
        
        return data_samples
