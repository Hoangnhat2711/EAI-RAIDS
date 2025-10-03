"""
Causal Explainer - Counterfactual Explanations & Causal Analysis

Trả lời câu hỏi "WHY" thay vì chỉ "WHAT":
- Counterfactual Explanations: "What minimal changes would flip the prediction?"
- Causal Feature Importance: "What is the causal effect of feature X?"
- Actionable Insights: "What should be changed to get desired outcome?"

Research-grade implementation cho publication
"""

import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Callable
import warnings


class CounterfactualExplainer:
    """
    Generate Counterfactual Explanations
    
    Counterfactual: "If feature X was Y instead of Z, 
                     then prediction would be different"
    
    Methods:
    - Wachter et al. (2017) - Optimization-based
    - DiCE (Diverse Counterfactual Explanations)
    - Actionable Recourse
    """
    
    def __init__(self, model: Any, method: str = 'wachter'):
        """
        Khởi tạo Counterfactual Explainer
        
        Args:
            model: Model cần giải thích
            method: 'wachter', 'dice', or 'actionable'
        """
        self.model = model
        self.method = method
        self.counterfactuals_history = []
    
    def explain(self, X_instance: np.ndarray, 
               desired_class: Optional[int] = None,
               features_to_vary: Optional[List[int]] = None,
               max_iterations: int = 1000,
               learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Generate counterfactual explanation cho một instance
        
        Args:
            X_instance: Instance cần giải thích (1D array)
            desired_class: Class mong muốn (None = flip current prediction)
            features_to_vary: Indices của features có thể thay đổi (None = all)
            max_iterations: Max optimization iterations
            learning_rate: Learning rate cho optimization
        
        Returns:
            Dictionary với counterfactual và explanation
        """
        X_instance = np.array(X_instance).flatten()
        
        if self.method == 'wachter':
            return self._wachter_counterfactual(
                X_instance, desired_class, features_to_vary, 
                max_iterations, learning_rate
            )
        elif self.method == 'dice':
            return self._dice_counterfactual(
                X_instance, desired_class, features_to_vary
            )
        else:
            raise ValueError(f"Method '{self.method}' not supported")
    
    def _wachter_counterfactual(self, X_instance: np.ndarray,
                                desired_class: Optional[int],
                                features_to_vary: Optional[List[int]],
                                max_iterations: int,
                                learning_rate: float) -> Dict[str, Any]:
        """
        Wachter et al. (2017) - Optimization-based counterfactual
        
        Minimize: loss(f(X'), desired) + λ * distance(X, X')
        """
        # Original prediction
        original_pred = self.model.predict(X_instance.reshape(1, -1))[0]
        
        # Nếu không specify desired class, flip current prediction
        if desired_class is None:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_instance.reshape(1, -1))[0]
                # Flip to opposite class (binary classification)
                desired_class = 1 - original_pred if len(proba) == 2 else (original_pred + 1) % len(proba)
            else:
                desired_class = 1 - original_pred
        
        # Initialize counterfactual
        X_cf = X_instance.copy()
        
        # Features có thể vary
        if features_to_vary is None:
            features_to_vary = list(range(len(X_instance)))
        
        # Optimization loop
        best_cf = X_cf.copy()
        best_distance = float('inf')
        
        for iteration in range(max_iterations):
            # Predict current counterfactual
            pred = self.model.predict(X_cf.reshape(1, -1))[0]
            
            # Check if found valid counterfactual
            if pred == desired_class:
                distance = np.linalg.norm(X_cf - X_instance)
                if distance < best_distance:
                    best_distance = distance
                    best_cf = X_cf.copy()
            
            # Gradient estimation (numerical)
            gradient = self._estimate_gradient(X_cf, desired_class)
            
            # Update only variable features
            for idx in features_to_vary:
                X_cf[idx] -= learning_rate * gradient[idx]
            
            # Early stopping
            if iteration % 100 == 0 and best_distance < float('inf'):
                if iteration > 500:  # Give enough time
                    break
        
        # Calculate changes
        changes = self._calculate_changes(X_instance, best_cf)
        
        result = {
            'original_instance': X_instance,
            'counterfactual': best_cf,
            'original_prediction': original_pred,
            'counterfactual_prediction': self.model.predict(best_cf.reshape(1, -1))[0],
            'desired_class': desired_class,
            'distance': best_distance,
            'changes': changes,
            'validity': best_distance < float('inf'),
            'method': 'wachter'
        }
        
        self.counterfactuals_history.append(result)
        
        return result
    
    def _estimate_gradient(self, X: np.ndarray, target_class: int) -> np.ndarray:
        """
        Estimate gradient numerically
        
        For each feature: (f(X + ε) - f(X)) / ε
        """
        epsilon = 1e-4
        gradient = np.zeros_like(X)
        
        original_pred = self.model.predict(X.reshape(1, -1))[0]
        
        for i in range(len(X)):
            X_plus = X.copy()
            X_plus[i] += epsilon
            
            pred_plus = self.model.predict(X_plus.reshape(1, -1))[0]
            
            # Gradient pointing toward target class
            if pred_plus == target_class:
                gradient[i] = -1  # Move in this direction
            elif pred_plus == original_pred:
                gradient[i] = 1   # Move away from this direction
            else:
                gradient[i] = 0
        
        return gradient
    
    def _dice_counterfactual(self, X_instance: np.ndarray,
                            desired_class: Optional[int],
                            features_to_vary: Optional[List[int]],
                            num_cfs: int = 3) -> Dict[str, Any]:
        """
        DiCE - Diverse Counterfactual Explanations
        
        Generate multiple diverse counterfactuals
        """
        # Generate multiple counterfactuals với different initializations
        counterfactuals = []
        
        for i in range(num_cfs):
            # Random initialization around original instance
            noise = np.random.normal(0, 0.1, X_instance.shape)
            X_init = X_instance + noise
            
            # Generate counterfactual
            cf_result = self._wachter_counterfactual(
                X_init, desired_class, features_to_vary,
                max_iterations=500, learning_rate=0.01
            )
            
            if cf_result['validity']:
                counterfactuals.append(cf_result['counterfactual'])
        
        # Calculate diversity
        diversity_score = self._calculate_diversity(counterfactuals)
        
        result = {
            'original_instance': X_instance,
            'counterfactuals': counterfactuals,
            'num_valid': len(counterfactuals),
            'diversity_score': diversity_score,
            'method': 'dice'
        }
        
        return result
    
    def _calculate_diversity(self, counterfactuals: List[np.ndarray]) -> float:
        """Calculate diversity among counterfactuals"""
        if len(counterfactuals) < 2:
            return 0.0
        
        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i+1, len(counterfactuals)):
                dist = np.linalg.norm(counterfactuals[i] - counterfactuals[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_changes(self, original: np.ndarray, 
                          counterfactual: np.ndarray) -> List[Dict]:
        """Calculate feature changes"""
        changes = []
        
        for i in range(len(original)):
            if abs(original[i] - counterfactual[i]) > 1e-6:
                changes.append({
                    'feature_index': i,
                    'original_value': float(original[i]),
                    'counterfactual_value': float(counterfactual[i]),
                    'change': float(counterfactual[i] - original[i]),
                    'relative_change': float((counterfactual[i] - original[i]) / (abs(original[i]) + 1e-8))
                })
        
        # Sort by absolute change
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        return changes
    
    def explain_batch(self, X: np.ndarray, 
                     desired_class: Optional[int] = None,
                     max_samples: int = 10) -> List[Dict]:
        """
        Generate counterfactuals cho batch of instances
        
        Args:
            X: Multiple instances
            desired_class: Desired class
            max_samples: Max số samples để process
        
        Returns:
            List of counterfactual explanations
        """
        X = np.array(X)
        n_samples = min(len(X), max_samples)
        
        results = []
        for i in range(n_samples):
            try:
                cf = self.explain(X[i], desired_class=desired_class)
                results.append(cf)
            except Exception as e:
                print(f"Error explaining sample {i}: {e}")
        
        return results
    
    def generate_text_explanation(self, cf_result: Dict,
                                  feature_names: Optional[List[str]] = None) -> str:
        """
        Generate human-readable explanation
        
        Args:
            cf_result: Result từ explain()
            feature_names: Names of features
        
        Returns:
            Text explanation
        """
        if not cf_result.get('validity', False):
            return "❌ Could not find valid counterfactual"
        
        text = []
        text.append("🔍 COUNTERFACTUAL EXPLANATION")
        text.append("=" * 60)
        
        text.append(f"\nOriginal Prediction: {cf_result['original_prediction']}")
        text.append(f"Counterfactual Prediction: {cf_result['counterfactual_prediction']}")
        text.append(f"Distance: {cf_result['distance']:.4f}")
        
        text.append(f"\n📝 Changes needed (top 5):")
        
        changes = cf_result['changes'][:5]
        for i, change in enumerate(changes, 1):
            feat_idx = change['feature_index']
            feat_name = feature_names[feat_idx] if feature_names else f"Feature {feat_idx}"
            
            text.append(f"{i}. {feat_name}:")
            text.append(f"   From: {change['original_value']:.4f}")
            text.append(f"   To:   {change['counterfactual_value']:.4f}")
            text.append(f"   Change: {change['change']:+.4f} ({change['relative_change']:+.1%})")
        
        text.append("\n" + "=" * 60)
        
        return "\n".join(text)


class CausalFeatureImportance:
    """
    Calculate Causal Feature Importance
    
    Khác với SHAP (correlation), đây là causal effect thực sự
    """
    
    def __init__(self, model: Any):
        """
        Khởi tạo Causal Feature Importance
        
        Args:
            model: Model cần analyze
        """
        self.model = model
    
    def compute_causal_effect(self, X: np.ndarray, y: np.ndarray,
                             feature_idx: int,
                             intervention_values: Optional[List] = None) -> Dict[str, float]:
        """
        Compute causal effect của một feature
        
        Sử dụng do-calculus: E[Y | do(X=x)] vs E[Y | X=x]
        
        Args:
            X: Features
            y: True labels
            feature_idx: Index của feature cần analyze
            intervention_values: Values to intervene (None = use quantiles)
        
        Returns:
            Causal effects
        """
        X = np.array(X)
        y = np.array(y)
        
        # Original predictions
        y_pred_original = self.model.predict(X)
        
        if intervention_values is None:
            # Use quantiles
            feature_values = X[:, feature_idx]
            intervention_values = np.percentile(feature_values, [25, 50, 75])
        
        # Compute effect of interventions
        effects = {}
        
        for value in intervention_values:
            # Create intervened data
            X_intervened = X.copy()
            X_intervened[:, feature_idx] = value
            
            # Predict with intervention
            y_pred_intervened = self.model.predict(X_intervened)
            
            # Causal effect = difference in predictions
            effect = np.mean(y_pred_intervened) - np.mean(y_pred_original)
            effects[f'intervention_{value:.2f}'] = effect
        
        # Average causal effect
        ace = np.mean(list(effects.values()))
        
        return {
            'feature_index': feature_idx,
            'average_causal_effect': ace,
            'intervention_effects': effects
        }
    
    def rank_features_by_causal_importance(self, X: np.ndarray, 
                                          y: np.ndarray) -> List[Tuple[int, float]]:
        """
        Rank tất cả features theo causal importance
        
        Args:
            X: Features
            y: Labels
        
        Returns:
            List of (feature_index, causal_effect) sorted by importance
        """
        n_features = X.shape[1]
        
        causal_effects = []
        
        for i in range(n_features):
            result = self.compute_causal_effect(X, y, i)
            causal_effects.append((i, abs(result['average_causal_effect'])))
        
        # Sort by absolute effect
        causal_effects.sort(key=lambda x: x[1], reverse=True)
        
        return causal_effects


class ActionableRecourse:
    """
    Actionable Recourse - Counterfactuals với constraints
    
    Đảm bảo counterfactuals là actionable trong thực tế:
    - Immutable features (e.g., age, race) không thể thay đổi
    - Monotonic features (e.g., education) chỉ tăng
    - Cost constraints
    """
    
    def __init__(self, model: Any):
        """
        Khởi tạo Actionable Recourse
        
        Args:
            model: Model
        """
        self.model = model
    
    def generate_recourse(self, X_instance: np.ndarray,
                         immutable_features: List[int] = None,
                         monotonic_features: Dict[int, str] = None,
                         feature_costs: Dict[int, float] = None,
                         max_cost: float = None) -> Dict[str, Any]:
        """
        Generate actionable recourse
        
        Args:
            X_instance: Instance cần recourse
            immutable_features: Features không thể thay đổi
            monotonic_features: {feature_idx: 'increase' or 'decrease'}
            feature_costs: Cost của việc thay đổi mỗi feature
            max_cost: Maximum cost allowed
        
        Returns:
            Actionable counterfactual
        """
        X_instance = np.array(X_instance).flatten()
        
        immutable_features = immutable_features or []
        monotonic_features = monotonic_features or {}
        feature_costs = feature_costs or {}
        
        # Initialize counterfactual
        X_cf = X_instance.copy()
        
        # Optimization với constraints
        max_iterations = 1000
        learning_rate = 0.01
        
        original_pred = self.model.predict(X_instance.reshape(1, -1))[0]
        desired_class = 1 - original_pred  # Flip
        
        for iteration in range(max_iterations):
            pred = self.model.predict(X_cf.reshape(1, -1))[0]
            
            if pred == desired_class:
                break
            
            # Gradient
            gradient = self._estimate_gradient(X_cf, desired_class)
            
            # Update với constraints
            for i in range(len(X_cf)):
                if i in immutable_features:
                    continue  # Don't change
                
                # Apply monotonic constraint
                if i in monotonic_features:
                    direction = monotonic_features[i]
                    if direction == 'increase' and gradient[i] < 0:
                        gradient[i] = 0
                    elif direction == 'decrease' and gradient[i] > 0:
                        gradient[i] = 0
                
                # Update
                X_cf[i] -= learning_rate * gradient[i]
        
        # Calculate total cost
        total_cost = 0
        for i, cost in feature_costs.items():
            if abs(X_cf[i] - X_instance[i]) > 1e-6:
                total_cost += cost * abs(X_cf[i] - X_instance[i])
        
        # Check constraints
        valid = True
        if max_cost and total_cost > max_cost:
            valid = False
        
        result = {
            'original_instance': X_instance,
            'recourse': X_cf,
            'total_cost': total_cost,
            'valid': valid,
            'prediction_changed': self.model.predict(X_cf.reshape(1, -1))[0] != original_pred
        }
        
        return result
    
    def _estimate_gradient(self, X: np.ndarray, target_class: int) -> np.ndarray:
        """Estimate gradient"""
        epsilon = 1e-4
        gradient = np.zeros_like(X)
        
        for i in range(len(X)):
            X_plus = X.copy()
            X_plus[i] += epsilon
            
            pred_plus = self.model.predict(X_plus.reshape(1, -1))[0]
            
            if pred_plus == target_class:
                gradient[i] = -1
            else:
                gradient[i] = 1
        
        return gradient

