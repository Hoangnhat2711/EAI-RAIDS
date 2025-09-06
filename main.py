import torch
import mlflow # Import MLflow
import yaml # Import yaml

from core.rcl_trainer import RCLTrainer
from core.simple_trainer import SimpleTrainer
from data.dataset_loaders import get_adult_census_dataloader, get_compas_dataloader, get_fairface_dataloader # Import FairFace dataloader
# from config import Config # Remove the Config class import


if __name__ == "__main__":
    # Load configuration from config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device based on config and availability
    device = config['device']
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        device = "cpu"
    elif device == "cuda" and torch.cuda.is_available():
        print("Using CUDA device.")
    else:
        print("Using CPU device.")

    # Initialize the trainers using values from config
    rcl_framework_ewc = RCLTrainer(learning_rate=config['learning_rate'], device=device)
    simple_framework = SimpleTrainer(learning_rate=config['learning_rate'], device=device)
    lwf_framework = RCLTrainer(learning_rate=config['learning_rate'], device=device) 

    # --- Task 1: Structured Data (Adult Census) ---
    print("\n--- Preparing Task 1: Adult Census Data ---")
    train_loader_adult, test_loader_adult, input_dim_adult, num_sensitive_attrs_adult = get_adult_census_dataloader(
        batch_size=config['batch_size'], 
        test_size=config['adult_test_size'], 
        random_state=config['random_state']
    )
    num_classes_adult = config['num_classes_adult']  # Binary classification for income
    
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
                epochs=config['epochs_per_task'],
                lambda_reg=config['lambda_reg'], # EWC regularization strength from config
                use_ewc=True, use_lwf=False # Explicitly enable EWC, disable LwF
            )
            rcl_framework_ewc.evaluate_task(test_loader_adult, num_classes_adult, sensitive_feature_info=adult_sensitive_feature_info, task_id=1)

        # Run with LwF-Framework (using RCLTrainer with LwF enabled)
        with mlflow.start_run(run_name="Adult_Census_Task1_LwF"):
            print("\n### Running Adult Census with LwF-Framework ###")
            mlflow.log_param("framework", "LwF")
            lwf_framework.train_task(
                data_type='structured',
                input_shape=(input_dim_adult,),
                num_classes=num_classes_adult,
                task_id=1,
                train_loader=train_loader_adult,
                epochs=config['epochs_per_task'],
                alpha_lwf=config['alpha_lwf'], # LwF distillation strength from config
                use_ewc=False, use_lwf=True # Explicitly enable LwF, disable EWC
            )
            lwf_framework.evaluate_task(test_loader_adult, num_classes_adult, sensitive_feature_info=adult_sensitive_feature_info, task_id=1)

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
                epochs=config['epochs_per_task'],
                lambda_reg=0.0, alpha_lwf=0.0, use_ewc=False, use_lwf=False # No regularization
            )
            simple_framework.evaluate_task(test_loader_adult, num_classes_adult, sensitive_feature_info=adult_sensitive_feature_info, task_id=1)

    # --- Task 2: Another Structured Data Task (COMPAS) ---
    print("\n--- Preparing Task 2: COMPAS Data ---")
    train_loader_compas, test_loader_compas, input_dim_compas, num_sensitive_attrs_compas = get_compas_dataloader(
        batch_size=config['batch_size'], 
        test_size=config['compas_test_size'], 
        random_state=config['random_state']
    )
    num_classes_compas = config['num_classes_compas'] # Binary classification for recidivism

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
                epochs=config['epochs_per_task'],
                lambda_reg=config['lambda_reg'], # EWC regularization strength from config
                use_ewc=True, use_lwf=False # Explicitly enable EWC, disable LwF
            )
            rcl_framework_ewc.evaluate_task(test_loader_compas, num_classes_compas, sensitive_feature_info=compas_sensitive_feature_info, task_id=2)

        # Run with LwF-Framework (using RCLTrainer with LwF enabled)
        with mlflow.start_run(run_name="COMPAS_Task2_LwF"):
            print("\n### Running COMPAS with LwF-Framework ###")
            mlflow.log_param("framework", "LwF")
            lwf_framework.train_task(
                data_type='structured',
                input_shape=(input_dim_compas,),
                num_classes=num_classes_compas,
                task_id=2,
                train_loader=train_loader_compas,
                epochs=config['epochs_per_task'],
                alpha_lwf=config['alpha_lwf'], # LwF distillation strength from config
                use_ewc=False, use_lwf=True # Explicitly enable LwF, disable EWC
            )
            lwf_framework.evaluate_task(test_loader_compas, num_classes_compas, sensitive_feature_info=compas_sensitive_feature_info, task_id=2)

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
                epochs=config['epochs_per_task'],
                lambda_reg=0.0, alpha_lwf=0.0, use_ewc=False, use_lwf=False # No regularization
            )
            simple_framework.evaluate_task(test_loader_compas, num_classes_compas, sensitive_feature_info=compas_sensitive_feature_info, task_id=2)

    # --- Task 3: Unstructured Data (FairFace - Simulated) ---
    print("\n--- Preparing Task 3: FairFace Data (Simulated) ---")
    train_loader_fairface, test_loader_fairface, input_shape_fairface, num_sensitive_attrs_fairface = get_fairface_dataloader(
        batch_size=config['batch_size'],
        image_size=config['fairface_image_size'],
        num_channels=config['fairface_num_channels'],
        num_classes=config['num_classes_fairface'],
        num_sensitive_groups=config['fairface_num_sensitive_groups'],
        num_samples=config['fairface_num_samples'],
        test_size=config['fairface_test_size'],
        random_state=config['random_state']
    )
    num_classes_fairface = config['num_classes_fairface']

    # Define sensitive feature info for FairFace (simulated, adjust based on actual data if integrated)
    # Assuming sensitive groups are one-hot encoded and correspond to simple indices.
    # For a real FairFace dataset, this would need to map actual race categories to indices.
    fairface_sensitive_feature_info = {
        'names': ['race_group'], # Generic name for simulated sensitive attribute
        'privileged_groups': {'race_group': 0}, # Example: Group 0 is privileged
        'unprivileged_groups': {'race_group': [1,2,3,4,5,6]} # Example: Other groups are unprivileged
    }

    if train_loader_fairface is not None:
        # Run with RCL-Framework (EWC)
        with mlflow.start_run(run_name="FairFace_Task3_RCL-EWC"):
            print("\n### Running FairFace with RCL-Framework (EWC) ###")
            mlflow.log_param("framework", "RCL-EWC")
            rcl_framework_ewc.train_task(
                data_type='unstructured',
                input_shape=input_shape_fairface,
                num_classes=num_classes_fairface,
                task_id=3,
                train_loader=train_loader_fairface,
                epochs=config['epochs_per_task'],
                lambda_reg=config['lambda_reg'],
                use_ewc=True, use_lwf=False
            )
            rcl_framework_ewc.evaluate_task(test_loader_fairface, num_classes_fairface, sensitive_feature_info=fairface_sensitive_feature_info, task_id=3)

        # Run with LwF-Framework
        with mlflow.start_run(run_name="FairFace_Task3_LwF"):
            print("\n### Running FairFace with LwF-Framework ###")
            mlflow.log_param("framework", "LwF")
            lwf_framework.train_task(
                data_type='unstructured',
                input_shape=input_shape_fairface,
                num_classes=num_classes_fairface,
                task_id=3,
                train_loader=train_loader_fairface,
                epochs=config['epochs_per_task'],
                alpha_lwf=config['alpha_lwf'],
                use_ewc=False, use_lwf=True
            )
            lwf_framework.evaluate_task(test_loader_fairface, num_classes_fairface, sensitive_feature_info=fairface_sensitive_feature_info, task_id=3)

        # Run with SimpleTrainer (Baseline)
        with mlflow.start_run(run_name="FairFace_Task3_Baseline"):
            print("\n### Running FairFace with SimpleTrainer (Baseline) ###")
            mlflow.log_param("framework", "Baseline")
            simple_framework.train_task(
                data_type='unstructured',
                input_shape=input_shape_fairface,
                num_classes=num_classes_fairface,
                task_id=3,
                train_loader=train_loader_fairface,
                epochs=config['epochs_per_task'],
                lambda_reg=0.0, alpha_lwf=0.0, use_ewc=False, use_lwf=False
            )
            simple_framework.evaluate_task(test_loader_fairface, num_classes_fairface, sensitive_feature_info=fairface_sensitive_feature_info, task_id=3)

    print("\nFramework demonstration complete.")
