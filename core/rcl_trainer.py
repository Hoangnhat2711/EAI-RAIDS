import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import mlflow # Import MLflow

from models.mlp_model import MLPModel
from models.cnn_model import CNNModel
from regularization.ewc import compute_fisher_matrix, update_ewc_params as ewc_update_params, compute_ewc_loss
from regularization.lwf import lwf_loss # Import LwF loss
from utils.fairness_metrics import evaluate_fairness_metrics

class RCLTrainer:
    """
    Core trainer class for the Regularized Continual Learning (RCL) Framework.

    This class orchestrates the training and evaluation process for models
    under continual learning settings, incorporating regularization techniques
    like Elastic Weight Consolidation (EWC) and Learning without Forgetting (LwF).
    It also integrates MLflow for experiment tracking.
    """
    def __init__(self, learning_rate=0.001, device='cpu'):
        """
        Initializes the RCLTrainer.

        Args:
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
            device (str, optional): The device to use for training ('cpu' or 'cuda').
                                    Defaults to 'cpu'.
        """
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.optimal_params = OrderedDict()  # Stores parameters after each task for EWC regularization
        self.fisher_matrix = OrderedDict()   # Stores Fisher Information Matrix for each task
        self.previous_model = None           # Stores the model from the previous task for LwF

        print(f"RCLTrainer initialized. Using device: {self.device}")

    def _initialize_model(self, data_type, input_shape, num_classes):
        """
        Initializes the appropriate model (MLP or CNN) based on the data type.

        Args:
            data_type (str): Specifies the type of data ('structured' for MLP, 'unstructured' for CNN).
            input_shape (tuple): The shape of the input data.
            num_classes (int): The number of output classes.

        Raises:
            ValueError: If `data_type` is unsupported or `input_shape` is incorrect for the chosen type.

        Returns:
            torch.nn.Module: The initialized model.
        """
        if data_type == 'structured':
            # input_shape for MLP should be a single dimension (e.g., (784,))
            if not isinstance(input_shape, tuple) or len(input_shape) != 1:
                raise ValueError("For 'structured' data, input_shape must be a tuple like (input_dim,).")
            self.model = MLPModel(input_dim=input_shape[0], num_classes=num_classes).to(self.device)
            print(f"Initialized MLPModel with input_dim={input_shape[0]}, num_classes={num_classes}")
        elif data_type == 'unstructured':
            # input_shape for CNN should be (channels, height, width)
            if not isinstance(input_shape, tuple) or len(input_shape) != 3:
                raise ValueError("For 'unstructured' data, input_shape must be a tuple like (channels, height, width).")
            self.model = CNNModel(input_shape=input_shape, num_classes=num_classes).to(self.device)
            print(f"Initialized CNNModel with input_shape={input_shape}, num_classes={num_classes}")
        else:
            raise ValueError(f"Unsupported data_type: {data_type}. Choose 'structured' or 'unstructured'.")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model

    def _save_task_specific_params(self, task_id, input_shape, num_classes, use_ewc, use_lwf):
        """
        Saves task-specific parameters required for regularization methods (EWC, LwF).

        Args:
            task_id (int): The identifier of the current task.
            input_shape (tuple): The input shape of the model, used for re-initializing LwF previous model.
            num_classes (int): The number of output classes, used for re-initializing LwF previous model.
            use_ewc (bool): True if EWC regularization is enabled.
            use_lwf (bool): True if LwF regularization is enabled.
        """
        if use_ewc:
            ewc_update_params(self.model, task_id, input_shape, num_classes, self.optimal_params, self.fisher_matrix, self.device)
        if use_lwf:
            # Create a detached copy of the current model for LwF
            if isinstance(self.model, MLPModel):
                self.previous_model = MLPModel(input_shape[0], num_classes).to(self.device)
            elif isinstance(self.model, CNNModel):
                self.previous_model = CNNModel(input_shape, num_classes).to(self.device)
            else:
                raise ValueError("Unsupported model type for LwF previous model copy.")
            
            self.previous_model.load_state_dict(self.model.state_dict())
            self.previous_model.eval() # Set to eval mode, no gradients
            print(f"Previous model for LwF saved for task {task_id}.")

    def train_task(self, data_type, input_shape, num_classes, task_id, train_loader, epochs=10, lambda_reg=0.1, alpha_lwf=0.0, use_ewc=True, use_lwf=False):
        """
        Trains the model for a single task, incorporating EWC and LwF regularization.

        Logs training hyperparameters and per-epoch losses to MLflow.

        Args:
            data_type (str): 'structured' or 'unstructured'.
            input_shape (tuple): Shape of input data (e.g., (784,) for MLP, (1, 28, 28) for CNN).
            num_classes (int): Number of classes for the task.
            task_id (int): Identifier for the current task.
            train_loader (torch.utils.data.DataLoader): The DataLoader for the current task's training data.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            lambda_reg (float, optional): Regularization strength for EWC. Defaults to 0.1.
            alpha_lwf (float, optional): Distillation loss strength for LwF. Defaults to 0.0.
            use_ewc (bool, optional): Whether to use EWC regularization. Defaults to True.
            use_lwf (bool, optional): Whether to use LwF regularization. Defaults to False.
        """
        print(f"\n--- Starting training for Task {task_id} ({data_type} data) ---")
        if self.model is None or self.model.__class__.__name__ != (
            'MLPModel' if data_type == 'structured' else 'CNNModel'):
            print(f"Initializing new model for data type: {data_type}")
            self._initialize_model(data_type, input_shape, num_classes)
        else:
            print(f"Using existing model for data type: {data_type}")

        # Log hyperparameters to MLflow
        mlflow.log_params({
            f"task{task_id}_learning_rate": self.learning_rate,
            f"task{task_id}_epochs": epochs,
            f"task{task_id}_lambda_reg": lambda_reg,
            f"task{task_id}_alpha_lwf": alpha_lwf,
            f"task{task_id}_use_ewc": use_ewc,
            f"task{task_id}_use_lwf": use_lwf,
        })

        self.model.train()
        
        for epoch in range(epochs):
            total_loss_epoch = 0
            task_loss_epoch = 0
            reg_loss_epoch = 0
            lwf_loss_epoch = 0 # New: LwF loss accumulation

            for batch_idx, (inputs, targets, sensitive_attrs) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute task-specific loss
                task_loss = self.model.compute_task_loss(outputs, targets)
                
                # Compute regularization loss (EWC-like)
                regularization_loss = torch.tensor(0.0).to(self.device)
                if use_ewc:
                    regularization_loss = compute_ewc_loss(self.model, task_id, lambda_reg, self.optimal_params, self.fisher_matrix, self.device)
                
                # Compute LwF distillation loss
                distillation_loss = torch.tensor(0.0).to(self.device)
                if use_lwf and self.previous_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.previous_model(inputs) # Get outputs from previous model
                    distillation_loss = lwf_loss(outputs, teacher_outputs) * alpha_lwf

                # Total loss
                total_loss = task_loss + regularization_loss + distillation_loss
                
                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                
                total_loss_epoch += total_loss.item()
                task_loss_epoch += task_loss.item()
                reg_loss_epoch += regularization_loss.item()
                lwf_loss_epoch += distillation_loss.item() # Accumulate LwF loss

            avg_total_loss = total_loss_epoch / len(train_loader)
            avg_task_loss = task_loss_epoch / len(train_loader)
            avg_reg_loss = reg_loss_epoch / len(train_loader)
            avg_lwf_loss = lwf_loss_epoch / len(train_loader) # New: Average LwF loss

            print(f"Task {task_id}, Epoch [{epoch+1}/{epochs}], "
                  f"Avg Total Loss: {avg_total_loss:.4f}, "
                  f"Avg Task Loss: {avg_task_loss:.4f}, "
                  f"Avg EWC Loss: {avg_reg_loss:.4f}, "
                  f"Avg LwF Loss: {avg_lwf_loss:.4f}")
            
            # Log metrics to MLflow for each epoch
            mlflow.log_metric(f"task{task_id}_avg_total_loss", avg_total_loss, step=epoch)
            mlflow.log_metric(f"task{task_id}_avg_task_loss", avg_task_loss, step=epoch)
            mlflow.log_metric(f"task{task_id}_avg_ewc_loss", avg_reg_loss, step=epoch)
            mlflow.log_metric(f"task{task_id}_avg_lwf_loss", avg_lwf_loss, step=epoch)

        # After training a task, save parameters for regularization methods
        self._save_task_specific_params(task_id, input_shape, num_classes, use_ewc, use_lwf)
        print(f"--- Task {task_id} training finished ---")

    def evaluate_task(self, test_loader, num_classes, sensitive_feature_info=None, task_id=None):
        """
        Evaluates the model on a given task, including accuracy and fairness metrics.

        Logs evaluation metrics to MLflow.

        Args:
            test_loader (torch.utils.data.DataLoader): The DataLoader for the task's test/validation set.
            num_classes (int): Number of classes for the task.
            sensitive_feature_info (dict, optional): Dictionary containing information about sensitive features.
                                                    Expected keys: 'names', 'privileged_groups', 'unprivileged_groups'.
                                                    Example: {'names': ['sex', 'race'],
                                                              'privileged_groups': {'sex': 0, 'race': 0},
                                                              'unprivileged_groups': {'sex': 1, 'race': [1,2,3,4]}}
                                                    Defaults to None.
            task_id (int, optional): The identifier of the task being evaluated. Used for MLflow logging.
                                    Defaults to None.
        Returns:
            dict: A dictionary containing accuracy and fairness metrics.
        """
        if self.model is None:
            print("No model initialized. Cannot evaluate.")
            return {'accuracy': 0.0}

        print("\n--- Starting evaluation ---")
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_sensitive_attrs = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets, sensitive_attrs in test_loader:
                inputs, targets, sensitive_attrs = inputs.to(self.device), targets.to(self.device), sensitive_attrs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_targets.append(targets.cpu())
                all_predictions.append(predicted.cpu())
                all_sensitive_attrs.append(sensitive_attrs.cpu())

        accuracy = 100 * correct / total
        print(f"Evaluation Accuracy: {accuracy:.2f}%")
        if task_id is not None: # Log final accuracy only if task_id is provided
            mlflow.log_metric(f"task{task_id}_accuracy", accuracy)

        metrics = {'accuracy': accuracy}

        if sensitive_feature_info and all_targets and all_predictions and all_sensitive_attrs:
            y_true_all = torch.cat(all_targets)
            y_pred_all = torch.cat(all_predictions)
            s_attr_all = torch.cat(all_sensitive_attrs)

            num_sensitive_attrs = s_attr_all.shape[1] if len(s_attr_all.shape) > 1 else 1
            
            fairness_results = evaluate_fairness_metrics(
                y_true_all,
                y_pred_all,
                s_attr_all,
                sensitive_feature_info['privileged_groups'],
                sensitive_feature_info['unprivileged_groups'],
            )
            metrics.update(fairness_results)

            print("Fairness Metrics:")
            for metric_name, value in fairness_results.items():
                print(f"  {metric_name}: {value:.4f}")
                if task_id is not None: # Log fairness metrics only if task_id is provided
                    mlflow.log_metric(f"task{task_id}_{metric_name}", value)

        print("--- Evaluation finished ---")
        return metrics
