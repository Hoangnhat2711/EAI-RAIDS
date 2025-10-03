"""
Gradient Wrapper - Unified interface cho gradient computation

Đảm bảo tính nhất quán giữa:
- Adversarial attacks (FGSM, PGD)
- Counterfactual explanations
- Causal analysis

Sử dụng analytical gradients từ adapters thay vì numerical approximation
"""

import numpy as np
from typing import Any, Optional
import warnings


class GradientWrapper:
    """
    Unified gradient computation wrapper
    
    Provides consistent interface cho:
    - AttackGenerator (adversarial attacks)
    - CounterfactualExplainer (causal explanations)
    - Any gradient-based method
    
    Uses analytical gradients from model adapters
    """
    
    def __init__(self, model_or_adapter: Any):
        """
        Khởi tạo Gradient Wrapper
        
        Args:
            model_or_adapter: Model hoặc BaseModelAdapter instance
        """
        self.model = model_or_adapter
        self.adapter = None
        self.framework = self._detect_framework()
        
        # Try to get adapter methods
        if hasattr(model_or_adapter, 'compute_gradients'):
            self.adapter = model_or_adapter
        else:
            # Try to create adapter
            self._try_create_adapter()
    
    def _detect_framework(self) -> str:
        """Detect ML framework"""
        model_type = type(self.model).__name__
        
        if 'torch' in str(type(self.model)).lower():
            return 'pytorch'
        elif 'tensorflow' in str(type(self.model)).lower() or 'keras' in str(type(self.model)).lower():
            return 'tensorflow'
        else:
            return 'sklearn'
    
    def _try_create_adapter(self):
        """Try to create appropriate adapter"""
        try:
            from core.adapters import SklearnAdapter, PyTorchAdapter, TensorFlowAdapter
            
            if self.framework == 'pytorch':
                # Need loss_fn and optimizer for PyTorch
                warnings.warn(
                    "PyTorchAdapter requires loss_fn and optimizer. "
                    "Gradients will use numerical approximation."
                )
                self.adapter = None
            elif self.framework == 'tensorflow':
                self.adapter = TensorFlowAdapter(self.model)
            else:
                self.adapter = SklearnAdapter(self.model)
        except Exception as e:
            warnings.warn(f"Could not create adapter: {e}")
            self.adapter = None
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray,
                         loss_fn: Optional[Any] = None) -> np.ndarray:
        """
        Compute gradients of loss w.r.t. input
        
        Uses analytical gradients if available, falls back to numerical
        
        Args:
            X: Input features
            y: Target labels
            loss_fn: Loss function (for PyTorch/TensorFlow)
        
        Returns:
            Gradients array
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        # Try analytical gradients first
        if self.adapter and hasattr(self.adapter, 'compute_gradients'):
            try:
                if self.framework == 'pytorch' and loss_fn is None:
                    warnings.warn("PyTorch adapter needs loss_fn for gradient computation")
                    return self._numerical_gradient(X, y)
                
                gradients = self.adapter.compute_gradients(X, y)
                return gradients
            except Exception as e:
                warnings.warn(f"Analytical gradient failed: {e}, using numerical approximation")
                return self._numerical_gradient(X, y)
        else:
            # Fallback to numerical
            return self._numerical_gradient(X, y)
    
    def _numerical_gradient(self, X: np.ndarray, y: np.ndarray,
                           epsilon: float = 1e-4) -> np.ndarray:
        """
        Numerical gradient approximation (fallback)
        
        ⚠️ WARNING: This is slow and inaccurate for large models.
        Should only be used as last resort.
        
        Args:
            X: Input
            y: Target
            epsilon: Perturbation size
        
        Returns:
            Approximate gradients
        """
        gradients = np.zeros_like(X)
        
        # Get base predictions
        if hasattr(self.model, 'predict_proba'):
            base_pred = self.model.predict_proba(X)
        else:
            base_pred = self.model.predict(X)
        
        # Compute gradient for each feature
        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += epsilon
            
            if hasattr(self.model, 'predict_proba'):
                pred_plus = self.model.predict_proba(X_plus)
            else:
                pred_plus = self.model.predict(X_plus)
            
            # Finite difference
            gradients[:, i] = (pred_plus - base_pred).ravel() / epsilon
        
        return gradients
    
    def compute_loss_gradient(self, X: np.ndarray, y: np.ndarray,
                             target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute gradient for adversarial attacks
        
        For targeted attacks: gradient toward target_class
        For untargeted: gradient away from true class
        
        Args:
            X: Input
            y: True labels
            target_class: Target class for targeted attack
        
        Returns:
            Loss gradients
        """
        if target_class is not None:
            # Targeted attack: minimize distance to target
            y_target = np.full_like(y, target_class)
            return -self.compute_gradients(X, y_target)
        else:
            # Untargeted attack: maximize loss on true labels
            return self.compute_gradients(X, y)
    
    def compute_prediction_gradient(self, X: np.ndarray, 
                                   class_index: int = 1) -> np.ndarray:
        """
        Compute gradient of prediction probability w.r.t. input
        
        Useful for counterfactual explanations
        
        Args:
            X: Input
            class_index: Which class probability to maximize
        
        Returns:
            Gradients
        """
        # Create dummy labels (all target class)
        y_dummy = np.full(len(X), class_index)
        
        return self.compute_gradients(X, y_dummy)
    
    def is_analytical(self) -> bool:
        """Check if using analytical gradients"""
        return self.adapter is not None and hasattr(self.adapter, 'compute_gradients')
    
    def get_info(self) -> dict:
        """Get gradient computation info"""
        return {
            'framework': self.framework,
            'has_adapter': self.adapter is not None,
            'analytical_gradients': self.is_analytical(),
            'adapter_type': type(self.adapter).__name__ if self.adapter else None
        }
    
    def __repr__(self) -> str:
        grad_type = "analytical" if self.is_analytical() else "numerical"
        return f"GradientWrapper(framework={self.framework}, gradients={grad_type})"


class OptimizationHelper:
    """
    Helper cho gradient-based optimization
    
    Used by CounterfactualExplainer và other optimization methods
    """
    
    def __init__(self, gradient_wrapper: GradientWrapper):
        """
        Khởi tạo Optimization Helper
        
        Args:
            gradient_wrapper: GradientWrapper instance
        """
        self.grad_wrapper = gradient_wrapper
    
    def optimize_toward_target(self, X_init: np.ndarray,
                               target_class: int,
                               max_iterations: int = 1000,
                               learning_rate: float = 0.01,
                               constraints: Optional[dict] = None) -> np.ndarray:
        """
        Optimize input toward target class
        
        Args:
            X_init: Initial input
            target_class: Desired class
            max_iterations: Max iterations
            learning_rate: Step size
            constraints: Dict with 'immutable_features', 'bounds', etc.
        
        Returns:
            Optimized input
        """
        X = X_init.copy()
        constraints = constraints or {}
        
        immutable = constraints.get('immutable_features', [])
        bounds = constraints.get('bounds', (X.min(), X.max()))
        
        for iteration in range(max_iterations):
            # Get current prediction
            pred = self.grad_wrapper.model.predict(X.reshape(1, -1))[0]
            
            if pred == target_class:
                break  # Success
            
            # Compute gradient
            y_target = np.array([target_class])
            gradient = self.grad_wrapper.compute_gradients(
                X.reshape(1, -1), y_target
            )[0]
            
            # Update (gradient ascent to maximize target class probability)
            X = X - learning_rate * gradient
            
            # Apply constraints
            for idx in immutable:
                X[idx] = X_init[idx]  # Don't change immutable features
            
            # Bounds
            X = np.clip(X, bounds[0], bounds[1])
        
        return X
    
    def project_to_constraints(self, X: np.ndarray, X_original: np.ndarray,
                              constraints: dict) -> np.ndarray:
        """
        Project X to satisfy constraints
        
        Args:
            X: Current point
            X_original: Original point
            constraints: Constraint dict
        
        Returns:
            Projected X
        """
        X_proj = X.copy()
        
        # Immutable features
        if 'immutable_features' in constraints:
            for idx in constraints['immutable_features']:
                X_proj[idx] = X_original[idx]
        
        # Monotonic features
        if 'monotonic_features' in constraints:
            for idx, direction in constraints['monotonic_features'].items():
                if direction == 'increase' and X_proj[idx] < X_original[idx]:
                    X_proj[idx] = X_original[idx]
                elif direction == 'decrease' and X_proj[idx] > X_original[idx]:
                    X_proj[idx] = X_original[idx]
        
        # Bounds
        if 'bounds' in constraints:
            X_proj = np.clip(X_proj, constraints['bounds'][0], constraints['bounds'][1])
        
        return X_proj

