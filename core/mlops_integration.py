"""
MLOps Integration - MLflow & DVC

Tích hợp với các công cụ MLOps chuyên nghiệp:
- MLflow: Experiment tracking, model registry, deployment
- DVC: Data version control, pipeline management
- Weights & Biases: Alternative experiment tracking

Đảm bảo:
- Reproducibility (khả năng tái tạo)
- Traceability (khả năng truy vết)
- Collaboration (hợp tác)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
import warnings
import json
import os
from pathlib import Path


class MLflowIntegration:
    """
    Integration với MLflow
    
    MLflow provides:
    - Experiment tracking
    - Model registry
    - Model serving
    - Artifact storage
    """
    
    def __init__(self, experiment_name: str = "responsible-ai",
                 tracking_uri: Optional[str] = None):
        """
        Khởi tạo MLflow Integration
        
        Args:
            experiment_name: Tên experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        
        self.mlflow_available = False
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow"""
        try:
            import mlflow
            self.mlflow = mlflow
            
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            
            self.mlflow_available = True
            print(f"✓ MLflow initialized: {self.tracking_uri}")
        except ImportError:
            warnings.warn(
                "MLflow not installed. Install: pip install mlflow\n"
                "Logging will be saved locally instead."
            )
    
    def start_run(self, run_name: Optional[str] = None,
                 tags: Optional[Dict] = None) -> Any:
        """
        Start MLflow run
        
        Args:
            run_name: Run name
            tags: Run tags
        
        Returns:
            MLflow run context
        """
        if not self.mlflow_available:
            warnings.warn("MLflow not available")
            return None
        
        return self.mlflow.start_run(run_name=run_name, tags=tags)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if not self.mlflow_available:
            return
        
        for key, value in params.items():
            self.mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if not self.mlflow_available:
            return
        
        for key, value in metrics.items():
            self.mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model: Any, artifact_path: str = "model",
                 signature: Optional[Any] = None):
        """
        Log model to MLflow using Adapter Pattern
        
        Ensures consistency with framework's BaseModelAdapter architecture
        
        Args:
            model: Trained model or ResponsibleModelWrapper
            artifact_path: Path trong artifact store
            signature: Model signature (input/output schema)
        """
        if not self.mlflow_available:
            return
        
        # Check if it's a ResponsibleModelWrapper
        try:
            from core.model_wrapper import ResponsibleModelWrapper
            
            if isinstance(model, ResponsibleModelWrapper):
                # Extract underlying model
                actual_model = model.model
                
                # Log metadata
                self.log_params({
                    'wrapped_model': True,
                    'model_type': type(actual_model).__name__
                })
                
                # Use adapter if available
                if hasattr(model, 'adapter') and model.adapter is not None:
                    self._log_via_adapter(model.adapter, artifact_path, signature)
                    return
                else:
                    # Fall back to unwrapped model
                    model = actual_model
        except ImportError:
            pass
        
        # Try to use BaseModelAdapter
        try:
            from core.adapters import BaseModelAdapter, SklearnAdapter, PyTorchAdapter, TensorFlowAdapter
            
            # Check if model is already an adapter
            if isinstance(model, BaseModelAdapter):
                self._log_via_adapter(model, artifact_path, signature)
                return
            
            # Try to create adapter
            adapter = self._create_adapter_for_model(model)
            if adapter is not None:
                self._log_via_adapter(adapter, artifact_path, signature)
                return
        except ImportError:
            pass
        
        # Fallback: Use MLflow's built-in logging
        self._log_via_mlflow_builtin(model, artifact_path, signature)
    
    def _create_adapter_for_model(self, model: Any) -> Optional[Any]:
        """Create appropriate adapter for model"""
        try:
            from core.adapters import SklearnAdapter, PyTorchAdapter, TensorFlowAdapter
            
            model_type = str(type(model))
            
            if 'sklearn' in model_type:
                return SklearnAdapter(model)
            elif 'torch' in model_type:
                # PyTorch adapter needs loss_fn and optimizer
                # For logging, we can create without them
                warnings.warn("PyTorch adapter created without loss_fn/optimizer for logging")
                return PyTorchAdapter(model, loss_fn=None, optimizer=None)
            elif 'tensorflow' in model_type or 'keras' in model_type:
                return TensorFlowAdapter(model)
        except Exception as e:
            warnings.warn(f"Could not create adapter: {e}")
        
        return None
    
    def _log_via_adapter(self, adapter: Any, artifact_path: str, signature: Optional[Any]):
        """
        Log model via BaseModelAdapter
        
        Uses adapter's save/load methods for consistency
        """
        import tempfile
        import os
        
        # Save model using adapter
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            
            # Use adapter's save method if available
            if hasattr(adapter, 'save_model'):
                adapter.save_model(model_path)
            else:
                # Fallback to pickle
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(adapter.model, f)
            
            # Log to MLflow
            self.mlflow.log_artifact(model_path, artifact_path)
            
            # Log adapter metadata
            self.log_params({
                'adapter_type': type(adapter).__name__,
                'framework': adapter.get_framework_info()['framework']
            })
        
        print(f"✓ Model logged via {type(adapter).__name__}")
    
    def _log_via_mlflow_builtin(self, model: Any, artifact_path: str, signature: Optional[Any]):
        """
        Fallback: Use MLflow's built-in logging
        
        Only used when adapters are not available
        """
        model_type = type(model).__name__
        
        warnings.warn(
            f"Logging {model_type} without adapter. "
            f"Consider using ResponsibleModelWrapper for consistency."
        )
        
        if 'sklearn' in str(type(model)):
            self.mlflow.sklearn.log_model(model, artifact_path, signature=signature)
        elif 'torch' in str(type(model)):
            self.mlflow.pytorch.log_model(model, artifact_path, signature=signature)
        elif 'tensorflow' in str(type(model)) or 'keras' in str(type(model)):
            self.mlflow.tensorflow.log_model(model, artifact_path, signature=signature)
        else:
            # Generic logging
            import pickle
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.mlflow.log_artifact(model_path, artifact_path)
    
    def log_fairness_metrics(self, fairness_results: Dict[str, Any]):
        """Log fairness metrics"""
        if not self.mlflow_available:
            return
        
        metrics = {}
        for metric_name, value in fairness_results.items():
            if isinstance(value, (int, float)):
                metrics[f"fairness_{metric_name}"] = value
        
        self.log_metrics(metrics)
    
    def log_robustness_metrics(self, robustness_results: Dict[str, Any]):
        """Log robustness metrics"""
        if not self.mlflow_available:
            return
        
        metrics = {}
        for metric_name, value in robustness_results.items():
            if isinstance(value, (int, float)):
                metrics[f"robustness_{metric_name}"] = value
        
        self.log_metrics(metrics)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file"""
        if not self.mlflow_available:
            return
        
        self.mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact"""
        if not self.mlflow_available:
            return
        
        self.mlflow.log_dict(dictionary, filename)
    
    def register_model(self, model_uri: str, name: str) -> Optional[Any]:
        """
        Register model to Model Registry
        
        Args:
            model_uri: URI of logged model
            name: Model name in registry
        
        Returns:
            ModelVersion object
        """
        if not self.mlflow_available:
            return None
        
        return self.mlflow.register_model(model_uri, name)
    
    def end_run(self):
        """End current MLflow run"""
        if self.mlflow_available:
            self.mlflow.end_run()


class DVCIntegration:
    """
    Integration với DVC (Data Version Control)
    
    DVC provides:
    - Data versioning
    - Pipeline management
    - Experiment tracking
    - Remote storage
    """
    
    def __init__(self, repo_path: str = "."):
        """
        Khởi tạo DVC Integration
        
        Args:
            repo_path: Path to DVC repository
        """
        self.repo_path = Path(repo_path)
        self.dvc_available = False
        self._setup_dvc()
    
    def _setup_dvc(self):
        """Setup DVC"""
        try:
            import dvc.api
            self.dvc_api = dvc.api
            self.dvc_available = True
            print(f"✓ DVC initialized: {self.repo_path}")
        except ImportError:
            warnings.warn(
                "DVC not installed. Install: pip install dvc\n"
                "Data versioning will not be available."
            )
    
    def track_data(self, data_path: str, remote: Optional[str] = None):
        """
        Track data file with DVC
        
        Args:
            data_path: Path to data file
            remote: Remote storage name
        """
        if not self.dvc_available:
            warnings.warn("DVC not available")
            return
        
        import subprocess
        
        # Add file to DVC
        subprocess.run(['dvc', 'add', data_path], check=True)
        
        # Push to remote if specified
        if remote:
            subprocess.run(['dvc', 'push', '-r', remote, data_path + '.dvc'], check=True)
        
        print(f"✓ Tracked {data_path} with DVC")
    
    def get_data(self, data_path: str, version: Optional[str] = None) -> str:
        """
        Get data file (with specific version)
        
        Args:
            data_path: Path to data file
            version: Git tag/commit for specific version
        
        Returns:
            Path to data file
        """
        if not self.dvc_available:
            warnings.warn("DVC not available, returning original path")
            return data_path
        
        if version:
            # Get specific version
            data = self.dvc_api.read(
                data_path,
                repo=str(self.repo_path),
                rev=version
            )
            return data
        else:
            # Get latest version
            return data_path
    
    def create_pipeline(self, pipeline_config: Dict[str, Any],
                       pipeline_file: str = "dvc.yaml"):
        """
        Create DVC pipeline
        
        Args:
            pipeline_config: Pipeline configuration
            pipeline_file: Pipeline file path
        """
        if not self.dvc_available:
            warnings.warn("DVC not available")
            return
        
        import yaml
        
        pipeline_path = self.repo_path / pipeline_file
        
        with open(pipeline_path, 'w') as f:
            yaml.dump(pipeline_config, f, default_flow_style=False)
        
        print(f"✓ Created pipeline: {pipeline_path}")
    
    def run_pipeline(self, pipeline_file: str = "dvc.yaml"):
        """Run DVC pipeline"""
        if not self.dvc_available:
            warnings.warn("DVC not available")
            return
        
        import subprocess
        
        subprocess.run(['dvc', 'repro', pipeline_file], check=True)
        print(f"✓ Pipeline executed: {pipeline_file}")


class ExperimentTracker:
    """
    Unified experiment tracker
    
    Combines MLflow, DVC, and local logging
    """
    
    def __init__(self, experiment_name: str = "responsible-ai",
                 use_mlflow: bool = True,
                 use_dvc: bool = True,
                 local_log_dir: str = "./experiments"):
        """
        Khởi tạo Experiment Tracker
        
        Args:
            experiment_name: Experiment name
            use_mlflow: Enable MLflow
            use_dvc: Enable DVC
            local_log_dir: Local log directory
        """
        self.experiment_name = experiment_name
        self.local_log_dir = Path(local_log_dir)
        self.local_log_dir.mkdir(exist_ok=True)
        
        # Initialize integrations
        self.mlflow = MLflowIntegration(experiment_name) if use_mlflow else None
        self.dvc = DVCIntegration() if use_dvc else None
        
        self.current_run_id = None
    
    def start_run(self, run_name: Optional[str] = None,
                 tags: Optional[Dict] = None) -> str:
        """
        Start experiment run
        
        Returns:
            Run ID
        """
        import time
        import uuid
        
        # Generate run ID
        self.current_run_id = f"run_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Start MLflow run
        if self.mlflow:
            self.mlflow.start_run(run_name=run_name or self.current_run_id, tags=tags)
        
        # Create local run directory
        run_dir = self.local_log_dir / self.current_run_id
        run_dir.mkdir(exist_ok=True)
        
        print(f"✓ Started run: {self.current_run_id}")
        return self.current_run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        # MLflow
        if self.mlflow:
            self.mlflow.log_params(params)
        
        # Local
        if self.current_run_id:
            run_dir = self.local_log_dir / self.current_run_id
            with open(run_dir / "params.json", 'w') as f:
                json.dump(params, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        # MLflow
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)
        
        # Local
        if self.current_run_id:
            run_dir = self.local_log_dir / self.current_run_id
            metrics_file = run_dir / "metrics.jsonl"
            
            with open(metrics_file, 'a') as f:
                record = {'step': step, **metrics}
                f.write(json.dumps(record) + '\n')
    
    def log_model(self, model: Any, model_name: str = "model"):
        """Log trained model"""
        # MLflow
        if self.mlflow:
            self.mlflow.log_model(model, artifact_path=model_name)
        
        # Local - save with pickle
        if self.current_run_id:
            import pickle
            run_dir = self.local_log_dir / self.current_run_id
            
            with open(run_dir / f"{model_name}.pkl", 'wb') as f:
                pickle.dump(model, f)
    
    def log_responsible_ai_metrics(self, metrics: Dict[str, Any]):
        """
        Log Responsible AI specific metrics
        
        Args:
            metrics: Dict containing fairness, robustness, privacy, etc.
        """
        # Flatten nested metrics
        flat_metrics = {}
        
        for category, values in metrics.items():
            if isinstance(values, dict):
                for metric_name, value in values.items():
                    if isinstance(value, (int, float)):
                        flat_metrics[f"{category}_{metric_name}"] = value
            elif isinstance(values, (int, float)):
                flat_metrics[category] = values
        
        self.log_metrics(flat_metrics)
    
    def log_dataset(self, dataset_path: str, track_with_dvc: bool = True):
        """Log dataset"""
        # DVC
        if self.dvc and track_with_dvc:
            self.dvc.track_data(dataset_path)
        
        # MLflow
        if self.mlflow:
            self.mlflow.log_artifact(dataset_path)
    
    def end_run(self):
        """End experiment run"""
        if self.mlflow:
            self.mlflow.end_run()
        
        print(f"✓ Ended run: {self.current_run_id}")
        self.current_run_id = None
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs
        
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for run_id in run_ids:
            run_dir = self.local_log_dir / run_id
            
            if not run_dir.exists():
                warnings.warn(f"Run {run_id} not found")
                continue
            
            # Load params
            with open(run_dir / "params.json", 'r') as f:
                params = json.load(f)
            
            # Load latest metrics
            metrics = {}
            metrics_file = run_dir / "metrics.jsonl"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        metrics.update({k: v for k, v in record.items() if k != 'step'})
            
            results.append({
                'run_id': run_id,
                **params,
                **metrics
            })
        
        return pd.DataFrame(results)


def create_reproducible_experiment(experiment_name: str,
                                   model: Any,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create fully reproducible experiment
    
    Args:
        experiment_name: Experiment name
        model: Model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        params: Model parameters
    
    Returns:
        Experiment results
    """
    tracker = ExperimentTracker(experiment_name)
    
    # Start run
    run_id = tracker.start_run()
    
    try:
        # Log params
        tracker.log_params(params)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        metrics = {
            'train_score': train_score,
            'test_score': test_score
        }
        
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(model)
        
        results = {
            'run_id': run_id,
            'params': params,
            'metrics': metrics
        }
        
    finally:
        tracker.end_run()
    
    return results

