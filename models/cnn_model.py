import torch
import torch.nn as nn
from .base_model import RCLModel

class CNNModel(RCLModel):
    """
    A Convolutional Neural Network (CNN) model designed for unstructured (image) data.

    This model extends the `RCLModel` and implements a basic CNN architecture suitable
    for image classification tasks. It dynamically calculates the size of the
    flattened features to allow flexibility in input image dimensions.
    """
    def __init__(self, input_shape, num_classes):
        """
        Initializes the CNNModel.

        Args:
            input_shape (tuple): The shape of the input image data, expected as
                                   (channels, height, width).
            num_classes (int): The number of output classes for the classification task.

        Raises:
            ValueError: If `input_shape` is not a tuple of length 3.
        """
        super(CNNModel, self).__init__(input_shape=input_shape, num_classes=num_classes)
        channels, height, width = input_shape
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, H/2, W/2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64, H/4, W/4)
            nn.Flatten()
        )

        # Calculate the size of the flattened features dynamically
        # We'll need a dummy input to calculate this
        dummy_input = torch.zeros(1, *input_shape)
        with torch.no_grad():
            flattened_features_size = self.feature_extractor(dummy_input).view(1, -1).size(1)

        self.classifier = nn.Linear(flattened_features_size, num_classes)

    def compute_task_loss(self, outputs, targets):
        """
        Computes the Cross-Entropy loss for the classification task.

        Args:
            outputs (torch.Tensor): The raw logits from the model's forward pass.
            targets (torch.Tensor): The true labels for the input data.

        Returns:
            torch.Tensor: The computed Cross-Entropy loss.
        """
        return nn.CrossEntropyLoss()(outputs, targets)
