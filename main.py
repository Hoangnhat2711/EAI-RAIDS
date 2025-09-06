import torch
import mlflow # Import MLflow

from core.rcl_trainer import RCLTrainer
from core.simple_trainer import SimpleTrainer
from data.dataset_loaders import get_adult_census_dataloader, get_compas_dataloader

# Define a specific trainer for LwF to manage its state (previous_model)
class LwFTrainer(RCLTrainer):
    def __init__(self, learning_rate=0.001, device='cpu'):
        super().__init__(learning_rate, device)
        print(f"LwFTrainer initialized. Using device: {self.device}")

    def train_task(self, data_type, input_shape, num_classes, task_id, train_loader, epochs=10, lambda_reg=0.1, alpha_lwf=1.0, use_ewc=False, use_lwf=True):
        # LwF always uses use_lwf=True and ignores lambda_reg
        super().train_task(data_type, input_shape, num_classes, task_id, train_loader, epochs, lambda_reg=0.0, alpha_lwf=alpha_lwf, use_ewc=False, use_lwf=True)


if __name__ == "__main__":
    # Initialize the trainers
    # Each trainer instance will maintain its own model and regularization states across tasks.
    rcl_framework_ewc = RCLTrainer(learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu')
    simple_framework = SimpleTrainer(learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu')
    lwf_framework = LwFTrainer(learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu') # Initialize LwF Trainer

    # --- Task 1: Structured Data (Adult Census) ---
    print("\n--- Preparing Task 1: Adult Census Data ---")
    train_loader_adult, test_loader_adult, input_dim_adult, num_sensitive_attrs_adult = get_adult_census_dataloader(batch_size=32)
    num_classes_adult = 2  # Binary classification for income
    
    # Define sensitive feature info for Adult Census
    adult_sensitive_feature_info = {
        'names': ['sex', 'race'],
        'privileged_groups': {'sex': 0, 'race': 0},
        'unprivileged_groups': {'sex': 1, 'race': [1, 2, 3, 4]}
    }

    if train_loader_adult is not None:
        # Run with RCL-Framework (EWC)
        with mlflow.start_run(run_name="Adult_Census_Task1_RCL-EWC"):
            print("\n### Running Adult Census with RCL-Framework (EWC) ###")
            mlflow.log_param("framework", "RCL-EWC")
            rcl_framework_ewc.train_task(
                data_type='structured',
                input_shape=(input_dim_adult,),
                num_classes=num_classes_adult,
                task_id=1,
                train_loader=train_loader_adult,
                epochs=5,
                lambda_reg=0.1, # EWC regularization strength
                use_ewc=True, use_lwf=False # Explicitly enable EWC, disable LwF
            )
            rcl_framework_ewc.evaluate_task(test_loader_adult, num_classes_adult, sensitive_feature_info=adult_sensitive_feature_info)

        # Run with LwF-Framework
        with mlflow.start_run(run_name="Adult_Census_Task1_LwF"):
            print("\n### Running Adult Census with LwF-Framework ###")
            mlflow.log_param("framework", "LwF")
            lwf_framework.train_task(
                data_type='structured',
                input_shape=(input_dim_adult,),
                num_classes=num_classes_adult,
                task_id=1,
                train_loader=train_loader_adult,
                epochs=5,
                alpha_lwf=1.0, # LwF distillation strength
                use_ewc=False, use_lwf=True # Explicitly enable LwF, disable EWC
            )
            lwf_framework.evaluate_task(test_loader_adult, num_classes_adult, sensitive_feature_info=adult_sensitive_feature_info)

        # Run with SimpleTrainer (Baseline)
        with mlflow.start_run(run_name="Adult_Census_Task1_Baseline"):
            print("\n### Running Adult Census with SimpleTrainer (Baseline) ###")
            mlflow.log_param("framework", "Baseline")
            simple_framework.train_task(
                data_type='structured',
                input_shape=(input_dim_adult,),
                num_classes=num_classes_adult,
                task_id=1,
                train_loader=train_loader_adult,
                epochs=5,
                lambda_reg=0.0, alpha_lwf=0.0, use_ewc=False, use_lwf=False # No regularization
            )
            simple_framework.evaluate_task(test_loader_adult, num_classes_adult, sensitive_feature_info=adult_sensitive_feature_info)

    # --- Task 2: Another Structured Data Task (COMPAS) ---
    print("\n--- Preparing Task 2: COMPAS Data ---")
    train_loader_compas, test_loader_compas, input_dim_compas, num_sensitive_attrs_compas = get_compas_dataloader(batch_size=32)
    num_classes_compas = 2 # Binary classification for recidivism

    # Define sensitive feature info for COMPAS
    compas_sensitive_feature_info = {
        'names': ['race', 'sex', 'age_cat'],
        'privileged_groups': {'race': 0, 'sex': 0, 'age_cat': 0},
        'unprivileged_groups': {'race': [1,2,3,4,5], 'sex': 1, 'age_cat': [1,2]}
    }

    if train_loader_compas is not None:
        # Run with RCL-Framework (EWC)
        with mlflow.start_run(run_name="COMPAS_Task2_RCL-EWC"):
            print("\n### Running COMPAS with RCL-Framework (EWC) ###")
            mlflow.log_param("framework", "RCL-EWC")
            rcl_framework_ewc.train_task(
                data_type='structured',
                input_shape=(input_dim_compas,),
                num_classes=num_classes_compas,
                task_id=2,
                train_loader=train_loader_compas,
                epochs=5,
                lambda_reg=0.1, # EWC regularization strength
                use_ewc=True, use_lwf=False # Explicitly enable EWC, disable LwF
            )
            rcl_framework_ewc.evaluate_task(test_loader_compas, num_classes_compas, sensitive_feature_info=compas_sensitive_feature_info)

        # Run with LwF-Framework
        with mlflow.start_run(run_name="COMPAS_Task2_LwF"):
            print("\n### Running COMPAS with LwF-Framework ###")
            mlflow.log_param("framework", "LwF")
            lwf_framework.train_task(
                data_type='structured',
                input_shape=(input_dim_compas,),
                num_classes=num_classes_compas,
                task_id=2,
                train_loader=train_loader_compas,
                epochs=5,
                alpha_lwf=1.0, # LwF distillation strength
                use_ewc=False, use_lwf=True # Explicitly enable LwF, disable EWC
            )
            lwf_framework.evaluate_task(test_loader_compas, num_classes_compas, sensitive_feature_info=compas_sensitive_feature_info)

        # Run with SimpleTrainer (Baseline)
        with mlflow.start_run(run_name="COMPAS_Task2_Baseline"):
            print("\n### Running COMPAS with SimpleTrainer (Baseline) ###")
            mlflow.log_param("framework", "Baseline")
            simple_framework.train_task(
                data_type='structured',
                input_shape=(input_dim_compas,),
                num_classes=num_classes_compas,
                task_id=2,
                train_loader=train_loader_compas,
                epochs=5,
                lambda_reg=0.0, alpha_lwf=0.0, use_ewc=False, use_lwf=False # No regularization
            )
            simple_framework.evaluate_task(test_loader_compas, num_classes_compas, sensitive_feature_info=compas_sensitive_feature_info)

    print("\nFramework demonstration complete.")
