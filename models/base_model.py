import torch.nn as nn

class RCLModel(nn.Module):
    """
    Base class for all models within the Regularized Continual Learning (RCL) Framework.

    This abstract base class defines the common interface and functionalities expected
    from any model designed to work with the RCL Framework. It ensures a consistent
    structure for feature extraction, classification, and task-specific loss computation.
    """
    def __init__(self, input_shape, num_classes):
        """
        Initializes the base RCLModel.

        Args:
            input_shape (tuple): The shape of the input data. For structured data,
                                   this might be (input_dim,). For unstructured data
                                   (e.g., images), this might be (channels, height, width).
            num_classes (int): The number of output classes for the classification task.
        """
        super(RCLModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.feature_extractor = None  # To be defined by child classes
        self.classifier = None         # To be defined by child classes

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The raw logits output by the classifier.
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def compute_task_loss(self, outputs, targets):
        """
        Computes the task-specific loss for the given outputs and targets.
        This method must be implemented by concrete child classes.

        Args:
            outputs (torch.Tensor): The raw logits from the model's forward pass.
            targets (torch.Tensor): The true labels for the input data.

        Raises:
            NotImplementedError: If the method is not implemented by a child class.

        Returns:
            torch.Tensor: The computed task-specific loss.
        """
        raise NotImplementedError("compute_task_loss must be implemented by child classes.")
