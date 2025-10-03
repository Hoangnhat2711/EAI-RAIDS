"""
Fairness In-Processing Techniques

Các phương pháp giảm thiểu bias trong quá trình training:
- Adversarial Debiasing (Zhang et al. 2018)
- Prejudice Remover Regularizer (Kamishima et al. 2012)
- Fair Representation Learning
- Constraint-based Optimization

References:
- Zhang et al. "Mitigating Unwanted Biases with Adversarial Learning" (AIES 2018)
- Kamishima et al. "Fairness-Aware Classifier with Prejudice Remover Regularizer" (ECML 2012)
- Agarwal et al. "A Reductions Approach to Fair Classification" (ICML 2018)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable
import warnings


class AdversarialDebiasing:
    """
    Adversarial Debiasing (Zhang et al. 2018)
    
    Train classifier và adversary đồng thời:
    - Classifier: Predict y from X
    - Adversary: Predict sensitive attribute from predictions
    
    Classifier learns to make predictions that adversary cannot use
    to infer sensitive attributes → fair predictions
    """
    
    def __init__(self, 
                 sensitive_attribute_idx: int,
                 adversary_loss_weight: float = 1.0,
                 learning_rate: float = 0.001,
                 epochs: int = 100):
        """
        Khởi tạo Adversarial Debiasing
        
        Args:
            sensitive_attribute_idx: Index của sensitive attribute
            adversary_loss_weight: Weight cho adversary loss
            learning_rate: Learning rate
            epochs: Number of epochs
        """
        self.sensitive_idx = sensitive_attribute_idx
        self.adversary_weight = adversary_loss_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.classifier = None
        self.adversary = None
        self.history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            classifier_arch: Optional[Callable] = None,
            adversary_arch: Optional[Callable] = None) -> Dict:
        """
        Train classifier với adversarial debiasing
        
        Args:
            X: Features
            y: Labels
            classifier_arch: Function returning classifier model
            adversary_arch: Function returning adversary model
        
        Returns:
            Training history
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch required for AdversarialDebiasing")
        
        # Extract sensitive attribute
        sensitive_attr = X[:, self.sensitive_idx]
        
        # Default architectures
        if classifier_arch is None:
            classifier_arch = lambda: nn.Sequential(
                nn.Linear(X.shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        if adversary_arch is None:
            adversary_arch = lambda: nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        # Initialize models
        self.classifier = classifier_arch()
        self.adversary = adversary_arch()
        
        # Optimizers
        classifier_optimizer = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        adversary_optimizer = optim.Adam(self.adversary.parameters(), lr=self.learning_rate)
        
        # Loss functions
        classifier_loss_fn = nn.BCELoss()
        adversary_loss_fn = nn.BCELoss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        sensitive_tensor = torch.FloatTensor(sensitive_attr).reshape(-1, 1)
        
        history = {
            'classifier_loss': [],
            'adversary_loss': [],
            'total_loss': []
        }
        
        for epoch in range(self.epochs):
            # =========================
            # Train Adversary
            # =========================
            self.adversary.train()
            self.classifier.eval()
            
            adversary_optimizer.zero_grad()
            
            # Get classifier predictions
            with torch.no_grad():
                pred = self.classifier(X_tensor)
            
            # Adversary tries to predict sensitive attribute
            adv_pred = self.adversary(pred)
            adv_loss = adversary_loss_fn(adv_pred, sensitive_tensor)
            
            adv_loss.backward()
            adversary_optimizer.step()
            
            # =========================
            # Train Classifier
            # =========================
            self.classifier.train()
            self.adversary.eval()
            
            classifier_optimizer.zero_grad()
            
            # Classifier prediction
            pred = self.classifier(X_tensor)
            
            # Classification loss
            clf_loss = classifier_loss_fn(pred, y_tensor)
            
            # Adversarial loss (classifier wants adversary to fail)
            with torch.no_grad():
                adv_pred = self.adversary(pred.detach())
            adv_pred_train = self.adversary(pred)
            
            # Minimize classification loss, maximize adversary confusion
            # (Adversary confusion = inverse of adversary loss)
            adversary_confusion = -adversary_loss_fn(adv_pred_train, sensitive_tensor)
            
            total_loss = clf_loss + self.adversary_weight * adversary_confusion
            
            total_loss.backward()
            classifier_optimizer.step()
            
            # Record
            history['classifier_loss'].append(clf_loss.item())
            history['adversary_loss'].append(adv_loss.item())
            history['total_loss'].append(total_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}: "
                      f"Clf Loss={clf_loss.item():.4f}, "
                      f"Adv Loss={adv_loss.item():.4f}, "
                      f"Total={total_loss.item():.4f}")
        
        self.history = history
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained classifier"""
        if self.classifier is None:
            raise ValueError("Model not trained yet")
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")
        
        self.classifier.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            pred = self.classifier(X_tensor)
            return (pred.numpy() > 0.5).astype(int).ravel()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if self.classifier is None:
            raise ValueError("Model not trained yet")
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")
        
        self.classifier.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            pred = self.classifier(X_tensor).numpy()
            return np.hstack([1 - pred, pred])


class PrejudiceRemover:
    """
    Prejudice Remover Regularizer (Kamishima et al. 2012)
    
    Add regularization term to loss function:
    L_total = L_classification + η * L_prejudice
    
    L_prejudice measures dependency between predictions and sensitive attributes
    """
    
    def __init__(self, 
                 sensitive_attribute_idx: int,
                 eta: float = 1.0,
                 fairness_measure: str = 'mutual_info'):
        """
        Khởi tạo Prejudice Remover
        
        Args:
            sensitive_attribute_idx: Index của sensitive attribute
            eta: Prejudice regularization weight
            fairness_measure: 'mutual_info' hoặc 'correlation'
        """
        self.sensitive_idx = sensitive_attribute_idx
        self.eta = eta
        self.fairness_measure = fairness_measure
    
    def compute_prejudice_loss(self, predictions: np.ndarray,
                               sensitive_attr: np.ndarray) -> float:
        """
        Compute prejudice loss
        
        Args:
            predictions: Model predictions
            sensitive_attr: Sensitive attribute values
        
        Returns:
            Prejudice loss value
        """
        if self.fairness_measure == 'mutual_info':
            # Mutual information between predictions and sensitive attribute
            return self._mutual_information(predictions, sensitive_attr)
        elif self.fairness_measure == 'correlation':
            # Pearson correlation
            return abs(np.corrcoef(predictions, sensitive_attr)[0, 1])
        else:
            raise ValueError(f"Unknown fairness measure: {self.fairness_measure}")
    
    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information I(X;Y)
        
        Uses histogram-based estimation
        """
        # Discretize
        x_discrete = (x > np.median(x)).astype(int)
        y_discrete = (y > np.median(y)).astype(int)
        
        # Joint distribution
        joint_hist = np.histogram2d(x_discrete, y_discrete, bins=2)[0]
        joint_prob = joint_hist / joint_hist.sum()
        
        # Marginal distributions
        x_prob = joint_prob.sum(axis=1)
        y_prob = joint_prob.sum(axis=0)
        
        # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
        mi = 0
        for i in range(2):
            for j in range(2):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (x_prob[i] * y_prob[j])
                    )
        
        return mi
    
    def create_regularized_loss(self, base_loss_fn: Callable) -> Callable:
        """
        Create regularized loss function
        
        Args:
            base_loss_fn: Original loss function
        
        Returns:
            Regularized loss function
        """
        def regularized_loss(predictions, y_true, X):
            # Base classification loss
            clf_loss = base_loss_fn(predictions, y_true)
            
            # Prejudice loss
            sensitive_attr = X[:, self.sensitive_idx]
            prejudice_loss = self.compute_prejudice_loss(predictions, sensitive_attr)
            
            # Total loss
            total_loss = clf_loss + self.eta * prejudice_loss
            
            return total_loss, clf_loss, prejudice_loss
        
        return regularized_loss


class FairConstrainedOptimization:
    """
    Constraint-based Fair Optimization (Agarwal et al. 2018)
    
    Formulate fairness as constrained optimization:
    
    minimize: Classification loss
    subject to: Fairness constraints (e.g., demographic parity, equal opportunity)
    
    Uses Lagrangian multipliers and reduction approach
    """
    
    def __init__(self, 
                 constraint_type: str = 'demographic_parity',
                 constraint_slack: float = 0.05,
                 max_iterations: int = 100):
        """
        Khởi tạo Fair Constrained Optimization
        
        Args:
            constraint_type: 'demographic_parity', 'equal_opportunity', 'equalized_odds'
            constraint_slack: Allowed slack in constraint (tolerance)
            max_iterations: Max iterations for optimization
        """
        self.constraint_type = constraint_type
        self.slack = constraint_slack
        self.max_iter = max_iterations
        
        self.model = None
        self.lagrange_multipliers = None
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            sensitive_features: np.ndarray,
            base_estimator: Any) -> Dict:
        """
        Train model với fairness constraints
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Sensitive attributes
            base_estimator: Base ML model (sklearn-compatible)
        
        Returns:
            Training info
        """
        try:
            from sklearn.base import clone
        except ImportError:
            raise ImportError("scikit-learn required")
        
        # Initialize Lagrange multipliers
        self.lagrange_multipliers = np.zeros(len(np.unique(sensitive_features)))
        
        history = {
            'loss': [],
            'constraint_violation': []
        }
        
        for iteration in range(self.max_iter):
            # Train model with current multipliers
            # (Simplified - in practice, use weighted training)
            model = clone(base_estimator)
            
            # Weight samples based on Lagrange multipliers
            sample_weights = self._compute_sample_weights(
                sensitive_features, self.lagrange_multipliers
            )
            
            model.fit(X, y, sample_weight=sample_weights)
            
            # Evaluate constraint violation
            predictions = model.predict(X)
            violation = self._evaluate_constraint(
                predictions, y, sensitive_features
            )
            
            # Update Lagrange multipliers
            self.lagrange_multipliers += 0.1 * violation  # Gradient ascent
            self.lagrange_multipliers = np.maximum(self.lagrange_multipliers, 0)
            
            history['loss'].append(np.mean((predictions != y).astype(float)))
            history['constraint_violation'].append(np.abs(violation).max())
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}: Violation={np.abs(violation).max():.4f}")
            
            # Check convergence
            if np.abs(violation).max() < self.slack:
                print(f"✓ Converged at iteration {iteration}")
                break
        
        self.model = model
        return history
    
    def _compute_sample_weights(self, sensitive_features: np.ndarray,
                                multipliers: np.ndarray) -> np.ndarray:
        """Compute sample weights from Lagrange multipliers"""
        weights = np.ones(len(sensitive_features))
        for i, group_val in enumerate(np.unique(sensitive_features)):
            mask = sensitive_features == group_val
            weights[mask] *= (1 + multipliers[i])
        return weights / weights.sum()
    
    def _evaluate_constraint(self, predictions: np.ndarray,
                            y_true: np.ndarray,
                            sensitive_features: np.ndarray) -> np.ndarray:
        """Evaluate fairness constraint violation"""
        groups = np.unique(sensitive_features)
        violations = []
        
        if self.constraint_type == 'demographic_parity':
            # P(Y_pred=1 | S=0) ≈ P(Y_pred=1 | S=1)
            overall_positive_rate = np.mean(predictions == 1)
            
            for group in groups:
                mask = sensitive_features == group
                group_positive_rate = np.mean(predictions[mask] == 1)
                violations.append(group_positive_rate - overall_positive_rate)
        
        elif self.constraint_type == 'equal_opportunity':
            # P(Y_pred=1 | Y=1, S=0) ≈ P(Y_pred=1 | Y=1, S=1)
            positive_mask = y_true == 1
            overall_tpr = np.mean(predictions[positive_mask] == 1)
            
            for group in groups:
                mask = (sensitive_features == group) & positive_mask
                if mask.sum() > 0:
                    group_tpr = np.mean(predictions[mask] == 1)
                    violations.append(group_tpr - overall_tpr)
                else:
                    violations.append(0)
        
        return np.array(violations)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)


def get_inprocessing_method(method: str, **kwargs) -> Any:
    """
    Factory function for in-processing methods
    
    Args:
        method: 'adversarial', 'prejudice_remover', 'constrained'
        **kwargs: Method-specific parameters
    
    Returns:
        In-processing method instance
    """
    if method == 'adversarial':
        return AdversarialDebiasing(**kwargs)
    elif method == 'prejudice_remover':
        return PrejudiceRemover(**kwargs)
    elif method == 'constrained':
        return FairConstrainedOptimization(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

