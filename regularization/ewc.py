import torch
import torch.nn as nn
from collections import OrderedDict

def compute_fisher_matrix(model, dummy_input_shape, num_classes, device, num_samples=100):
    """
    Computes an approximate Fisher Information Matrix (FIM) for the given model.

    The FIM estimates the importance of each parameter to the learned task.
    It is approximated by the squared gradients of the log-likelihood with respect to the parameters.
    This implementation uses dummy data for computation, as actual data is handled by DataLoaders
    in the training loop.

    Args:
        model (torch.nn.Module): The neural network model for which to compute the FIM.
        dummy_input_shape (tuple): The shape of the dummy input data to feed into the model
                                   for gradient calculation (e.g., (input_dim,) for MLP, (C, H, W) for CNN).
        num_classes (int): The number of output classes for the task.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        num_samples (int, optional): The number of dummy samples to use for approximating the FIM.
                                     Defaults to 100.

    Returns:
        collections.OrderedDict: A dictionary where keys are parameter names and values are
                                 the corresponding Fisher information values (squared gradients).
    """
    print(f"Computing Fisher matrix with dummy data for input_shape={dummy_input_shape}...")
    fisher = OrderedDict()
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p.data)

    for _ in range(num_samples):
        if len(dummy_input_shape) == 1: # Structured data
            inputs = torch.randn(1, dummy_input_shape[0]).to(device)
        elif len(dummy_input_shape) == 3: # Unstructured data
            inputs = torch.randn(1, *dummy_input_shape).to(device)
        else:
            raise ValueError("Invalid dummy_input_shape for Fisher computation.")
        
        targets = torch.randint(0, num_classes, (1,)).to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2) / num_samples

    return fisher

def update_ewc_params(model, task_id, dummy_input_shape, num_classes, optimal_params, fisher_matrix, device):
    """
    Stores the optimal parameters and computes/stores the Fisher Information Matrix (FIM)
    for the current task. These are then used as regularization for future tasks in EWC.

    Args:
        model (torch.nn.Module): The model after training on the current task.
        task_id (int): The identifier of the current task.
        dummy_input_shape (tuple): The shape of the dummy input data for FIM computation.
        num_classes (int): The number of output classes for the task.
        optimal_params (collections.OrderedDict): A dictionary to store optimal parameters for each task.
        fisher_matrix (collections.OrderedDict): A dictionary to store FIM for each task.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
    """
    print(f"Updating EWC parameters for task {task_id}...")
    optimal_params[task_id] = OrderedDict()
    for n, p in model.named_parameters():
        optimal_params[task_id][n] = p.clone().detach()

    fisher_matrix[task_id] = compute_fisher_matrix(model, dummy_input_shape, num_classes, device)
    print(f"EWC parameters for task {task_id} updated.")

def compute_ewc_loss(model, current_task_id, lambda_reg, optimal_params, fisher_matrix, device):
    """
    Computes the Elastic Weight Consolidation (EWC) regularization loss.

    EWC penalizes deviations of the current model's parameters from the parameters
    that were optimal for previous tasks, weighted by their importance (Fisher information).

    Args:
        model (torch.nn.Module): The current neural network model.
        current_task_id (int): The identifier of the current task being trained.
        lambda_reg (float): The regularization strength (hyperparameter).
        optimal_params (collections.OrderedDict): Stored optimal parameters from previous tasks.
        fisher_matrix (collections.OrderedDict): Stored Fisher Information Matrices from previous tasks.
        device (torch.device): The device (CPU or GPU) on which to perform computations.

    Returns:
        torch.Tensor: The computed EWC regularization loss.
    """
    if not optimal_params or not fisher_matrix:
        return torch.tensor(0.0).to(device)

    ewc_loss = torch.tensor(0.0).to(device)
    for prev_task_id in optimal_params:
        if prev_task_id == current_task_id:
            continue

        optimal_p = optimal_params[prev_task_id]
        fisher_m = fisher_matrix[prev_task_id]

        for n, p in model.named_parameters():
            if n in optimal_p and n in fisher_m:
                ewc_loss += (fisher_m[n] * (p - optimal_p[n]).pow(2)).sum()
    
    return lambda_reg * ewc_loss
