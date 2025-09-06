import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import mlflow # Import MLflow

from models.mlp_model import MLPModel
from models.cnn_model import CNNModel
from core.rcl_trainer import RCLTrainer # Import RCLTrainer to inherit common functionalities
from utils.fairness_metrics import evaluate_fairness_metrics

class SimpleTrainer(RCLTrainer):
    """
    A simple trainer class for single-task learning (baseline fine-tuning).

    This class extends `RCLTrainer` but overrides the `train_task` method to perform
    standard model training without any continual learning regularization (EWC or LwF).
    It is intended as a baseline to compare against continual learning approaches.
    MLflow integration is included for experiment tracking.
    """
    def __init__(self, learning_rate=0.001, device='cpu'):
        """
        Initializes the SimpleTrainer.

        Args:
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
            device (str, optional): The device to use for training ('cpu' or 'cuda').
                                    Defaults to 'cpu'.
        """
        super().__init__(learning_rate, device)
        print(f"SimpleTrainer initialized. Using device: {self.device}")

    def train_task(self, data_type, input_shape, num_classes, task_id, train_loader, epochs=10, lambda_reg=0.1, alpha_lwf=0.0, use_ewc=False, use_lwf=False):
        """
        Trains the model for a single task using standard fine-tuning.

        This method explicitly disables EWC and LwF regularization and only computes
        the task-specific loss. Logs training hyperparameters and per-epoch losses to MLflow.

        Args:
            data_type (str): 'structured' or 'unstructured'.
            input_shape (tuple): Shape of input data.
            num_classes (int): Number of classes for the task.
            task_id (int): Identifier for the current task.
            train_loader (torch.utils.data.DataLoader): The DataLoader for the current task's training data.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            lambda_reg (float, optional): Not used in SimpleTrainer (kept for signature consistency).
            alpha_lwf (float, optional): Not used in SimpleTrainer (kept for signature consistency).
            use_ewc (bool, optional): Always False for SimpleTrainer. Kept for signature consistency.
            use_lwf (bool, optional): Always False for SimpleTrainer. Kept for signature consistency.
        """
        print(f"\n--- Starting SimpleTrainer training for Task {task_id} ({data_type} data) ---")
        if self.model is None or self.model.__class__.__name__ != (
            'MLPModel' if data_type == 'structured' else 'CNNModel'):
            print(f"Initializing new model for data type: {data_type}")
            self._initialize_model(data_type, input_shape, num_classes)
        else:
            print(f"Using existing model for data type: {data_type} for fine-tuning.")
            # For fine-tuning, we typically re-initialize or reset the optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Log hyperparameters to MLflow
        mlflow.log_params({
            f"task{task_id}_learning_rate": self.learning_rate,
            f"task{task_id}_epochs": epochs,
            f"task{task_id}_lambda_reg": 0.0, # Always 0 for baseline
            f"task{task_id}_alpha_lwf": 0.0,  # Always 0 for baseline
            f"task{task_id}_use_ewc": False,
            f"task{task_id}_use_lwf": False,
        })

        self.model.train()
        
        for epoch in range(epochs):
            total_loss_epoch = 0
            task_loss_epoch = 0
            
            for batch_idx, (inputs, targets, sensitive_attrs) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                task_loss = self.model.compute_task_loss(outputs, targets)
                
                total_loss = task_loss # No regularization loss
                
                total_loss.backward()
                self.optimizer.step()
                
                total_loss_epoch += total_loss.item()
                task_loss_epoch += task_loss.item()

            avg_total_loss = total_loss_epoch / len(train_loader)
            avg_task_loss = task_loss_epoch / len(train_loader)
            
            print(f"SimpleTrainer Task {task_id}, Epoch [{epoch+1}/{epochs}], "
                  f"Avg Total Loss: {avg_total_loss:.4f}, "
                  f"Avg Task Loss: {avg_task_loss:.4f}")

            # Log metrics to MLflow for each epoch
            mlflow.log_metric(f"task{task_id}_avg_total_loss", avg_total_loss, step=epoch)
            mlflow.log_metric(f"task{task_id}_avg_task_loss", avg_task_loss, step=epoch)

        # No EWC parameter update for simple fine-tuning
        print(f"--- SimpleTrainer Task {task_id} training finished ---")

    def evaluate_task(self, test_loader, num_classes, sensitive_feature_info=None, task_id=None):
        """
        Evaluates the model on a given task using the inherited `evaluate_task` method.

        Logs evaluation metrics to MLflow.

        Args:
            test_loader (torch.utils.data.DataLoader): The DataLoader for the task's test/validation set.
            num_classes (int): Number of classes for the task.
            sensitive_feature_info (dict, optional): Dictionary containing information about sensitive features.
                                                    Defaults to None.
            task_id (int, optional): The identifier of the task being evaluated. Used for MLflow logging.
                                    Defaults to None.
        Returns:
            dict: A dictionary containing accuracy and fairness metrics.
        """
        # Call the base class's evaluate_task, which already includes MLflow logging
        # Ensure task_id is passed for logging consistency.
        results = super().evaluate_task(test_loader, num_classes, sensitive_feature_info, task_id=task_id)
        return results
