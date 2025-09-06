import torch.nn as nn
from .base_model import RCLModel

class MLPModel(RCLModel):
    """
    A Multi-Layer Perceptron (MLP) model designed for structured (tabular) data.

    This model extends the `RCLModel` and implements a basic feed-forward neural
    network architecture suitable for classification tasks on tabular datasets.
    """
    def __init__(self, input_dim, num_classes):
        """
        Initializes the MLPModel.

        Args:
            input_dim (int): The dimensionality of the input features.
            num_classes (int): The number of output classes for the classification task.
        """
        super(MLPModel, self).__init__(input_shape=(input_dim,), num_classes=num_classes)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

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
