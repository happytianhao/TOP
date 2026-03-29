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

from .cap_src import accident 


@MODELS.register_module()
class CAPModel(BaseModel):
    """Adapter to wrap cap_src.model.accident into MMAction2's BaseModel.

    Expected inputs for training (loss mode):
        - inputs: dict with keys 'x' and 'y'
            x: torch.Tensor, model-specific input tensor expected by `accident`
            y: torch.Tensor, labels tensor expected by `accident`

    For inference (predict mode):
        - inputs: dict with key 'x'
    """

    def __init__(
        self,
        data_preprocessor: Dict = None,
    ) -> None:
        if data_preprocessor is None:
            data_preprocessor = dict(type="ActionDataPreprocessor")
        super().__init__(data_preprocessor=data_preprocessor)

        self.model = accident()
        self.epoch = None

    def forward(
        self, 
        inputs: torch.Tensor, 
        data_samples: SampleList = None, 
        mode: str = 'tensor',
        **kwargs
    ):
        """Forward function for CAPModel.
        
        Args:
            inputs (torch.Tensor): Input tensor with shape (B, T, C, H, W)
                where B is batch size, T is number of clips/frames
            data_samples (SampleList): List of data samples containing labels and metadata
            mode (str): Forward mode, one of 'loss', 'predict', 'tensor'
            
        Returns:
            dict or list: Depends on the mode
                - 'loss': return dict of losses
                - 'predict': return list of predictions
                - 'tensor': return raw tensor outputs
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, **kwargs)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'loss', 'predict', or 'tensor'")
    
    def loss(self, inputs: torch.Tensor, data_samples: SampleList, **kwargs) -> dict:
        """Calculate losses.
        
        Args:
            inputs (torch.Tensor): Input tensor with shape (B, T, C, H, W)
            data_samples (SampleList): List of data samples containing labels
            
        Returns:
            dict: Dictionary of losses
        """
        # Extract labels from data_samples
        # Create labels in format B*2: [1-have_accident, have_accident] converted to 0/1
        labels = []
        for sample in data_samples:
            have_accident = int(sample.have_accident)
            label = [1 - have_accident, have_accident]
            labels.append(label)
        labels = torch.tensor(labels, device=inputs.device, dtype=torch.float32)
        
        # Call the accident model's forward method
        # The accident model expects x (rgb input) and y (labels)
        losses, all_output = self.model(inputs, labels)
        
        return losses
    
    def predict(self, inputs: torch.Tensor, data_samples: SampleList, **kwargs) -> SampleList:
        """Predict results.
        
        Args:
            inputs (torch.Tensor): Input tensor with shape (B, T, C, H, W)
            data_samples (SampleList): List of data samples
            
        Returns:
            SampleList: List of data samples with predictions
        """
        # For inference, we still need labels to run the model
        # Create labels in format B*2: [1-have_accident, have_accident] converted to 0/1
        labels = []
        for sample in data_samples:
            have_accident = int(sample.have_accident)
            label = [1 - have_accident, have_accident]
            labels.append(label)
        labels = torch.tensor(labels, device=inputs.device, dtype=torch.float32)
        
        # Run forward pass
        with torch.no_grad():
            _, all_output = self.model(inputs, labels)
        
        # Process predictions and attach to data_samples
        # all_output is a list of length T, each tensor has shape (B, 2)
        # Stack all outputs to (B, T, 2) and apply softmax
        all_output = torch.stack(all_output, dim=1)  # (B, T, 2)
        
        # Update data_samples with predictions
        for i, data_sample in enumerate(data_samples):
            # Apply softmax and take the second value (accident probability)
            data_sample.set_pred_score(F.softmax(all_output[i], dim=-1)[:, 1].detach())  # Shape: (T,)
        
        return data_samples
    
    def _forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Raw forward pass without loss calculation.
        
        Args:
            inputs (torch.Tensor): Input tensor with shape (B, T, C, H, W)
            
        Returns:
            torch.Tensor: Raw output tensor
        """
        # Create dummy labels for forward pass
        batch_size = inputs.shape[0]
        labels = torch.zeros((batch_size, 2), device=inputs.device)
        labels[:, 0] = 1
        
        # Run forward pass
        _, all_output = self.model(inputs, labels)
        
        # Return outputs from all time steps
        return torch.stack(all_output, dim=1)  # (B, T, num_classes)
