"""
SHAP Explainer - Sử dụng SHAP (SHapley Additive exPlanations)
để giải thích predictions
"""

import numpy as np
from typing import Any, Optional, Dict
import warnings


class SHAPExplainer:
    """
    SHAP-based explainer cho ML models
    
    SHAP (SHapley Additive exPlanations) là một phương pháp giải thích
    dựa trên game theory để hiểu contribution của mỗi feature
    """
    
    def __init__(self, model: Any):
        """
        Khởi tạo SHAP Explainer
        
        Args:
            model: ML model cần giải thích
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        
        # Kiểm tra xem shap có được cài đặt không
        try:
            import shap
            self.shap = shap
            self.is_available = True
        except ImportError:
            warnings.warn(
                "SHAP chưa được cài đặt. Chạy: pip install shap",
                ImportWarning
            )
            self.is_available = False
    
    def explain(self, X, sample_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Tạo SHAP explanation cho predictions
        
        Args:
            X: Features để giải thích
            sample_index: Index của sample cụ thể (None = tất cả)
        
        Returns:
            Dictionary chứa SHAP values và explanation
        """
        if not self.is_available:
            return {
                'error': 'SHAP không khả dụng. Cài đặt: pip install shap',
                'explanation': None
            }
        
        X = np.array(X)
        
        # Khởi tạo explainer phù hợp với model type
        self._initialize_explainer(X)
        
        # Tính SHAP values
        self.shap_values = self.explainer.shap_values(X)
        
        explanation = {
            'method': 'SHAP',
            'shap_values': self.shap_values,
            'feature_importance': self._get_feature_importance(),
            'base_value': getattr(self.explainer, 'expected_value', None)
        }
        
        # Nếu giải thích cho 1 sample cụ thể
        if sample_index is not None:
            explanation['sample_explanation'] = self._explain_single_sample(
                X, sample_index
            )
        
        return explanation
    
    def _initialize_explainer(self, X):
        """Khởi tạo SHAP explainer phù hợp với model type"""
        if self.explainer is not None:
            return  # Đã khởi tạo rồi
        
        model_type = type(self.model).__name__
        
        try:
            # Tree-based models
            if any(tree_type in model_type for tree_type in [
                'RandomForest', 'GradientBoosting', 'XGBoost', 
                'LightGBM', 'CatBoost', 'DecisionTree'
            ]):
                self.explainer = self.shap.TreeExplainer(self.model)
            
            # Linear models
            elif any(linear_type in model_type for linear_type in [
                'Linear', 'Logistic', 'Ridge', 'Lasso'
            ]):
                self.explainer = self.shap.LinearExplainer(self.model, X)
            
            # Deep learning models
            elif any(dl_type in model_type for dl_type in [
                'Sequential', 'Model', 'Network'
            ]):
                self.explainer = self.shap.DeepExplainer(self.model, X[:100])
            
            # Fallback: Kernel explainer (model-agnostic nhưng chậm hơn)
            else:
                self.explainer = self.shap.KernelExplainer(
                    self.model.predict, 
                    self.shap.sample(X, 100)
                )
        
        except Exception as e:
            warnings.warn(f"Không thể khởi tạo specialized explainer: {e}. Sử dụng KernelExplainer.")
            self.explainer = self.shap.KernelExplainer(
                self.model.predict,
                self.shap.sample(X, min(100, len(X)))
            )
    
    def _get_feature_importance(self) -> np.ndarray:
        """Tính feature importance từ SHAP values"""
        if self.shap_values is None:
            return np.array([])
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            # Average across classes
            importance = np.mean([
                np.abs(sv).mean(axis=0) 
                for sv in self.shap_values
            ], axis=0)
        else:
            importance = np.abs(self.shap_values).mean(axis=0)
        
        return importance
    
    def _explain_single_sample(self, X, sample_index: int) -> Dict[str, Any]:
        """Giải thích chi tiết cho một sample"""
        if self.shap_values is None:
            return {}
        
        # Get SHAP values cho sample này
        if isinstance(self.shap_values, list):
            # Multi-class: lấy class đầu tiên
            sample_shap = self.shap_values[0][sample_index]
        else:
            sample_shap = self.shap_values[sample_index]
        
        # Sắp xếp features theo importance
        feature_indices = np.argsort(np.abs(sample_shap))[::-1]
        
        explanation = {
            'sample_index': sample_index,
            'top_features': []
        }
        
        # Lấy top 10 features quan trọng nhất
        for idx in feature_indices[:10]:
            explanation['top_features'].append({
                'feature_index': int(idx),
                'feature_value': float(X[sample_index][idx]),
                'shap_value': float(sample_shap[idx]),
                'contribution': 'positive' if sample_shap[idx] > 0 else 'negative'
            })
        
        return explanation
    
    def plot_summary(self, X, max_display: int = 20):
        """
        Tạo SHAP summary plot
        
        Args:
            X: Features
            max_display: Số lượng features tối đa để hiển thị
        """
        if not self.is_available:
            print("SHAP không khả dụng")
            return
        
        if self.shap_values is None:
            self.explain(X)
        
        try:
            self.shap.summary_plot(
                self.shap_values, 
                X, 
                max_display=max_display
            )
        except Exception as e:
            print(f"Không thể tạo summary plot: {e}")
    
    def plot_force(self, X, sample_index: int = 0):
        """
        Tạo SHAP force plot cho một sample
        
        Args:
            X: Features
            sample_index: Index của sample cần plot
        """
        if not self.is_available:
            print("SHAP không khả dụng")
            return
        
        if self.shap_values is None:
            self.explain(X)
        
        try:
            # Handle multi-class
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[0][sample_index]
            else:
                shap_vals = self.shap_values[sample_index]
            
            self.shap.force_plot(
                self.explainer.expected_value,
                shap_vals,
                X[sample_index]
            )
        except Exception as e:
            print(f"Không thể tạo force plot: {e}")
    
    def get_feature_importance_ranking(self, feature_names=None) -> Dict[str, float]:
        """
        Lấy ranking của feature importance
        
        Args:
            feature_names: Tên các features
        
        Returns:
            Dictionary với feature names và importance scores
        """
        importance = self._get_feature_importance()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Sắp xếp theo importance
        sorted_indices = np.argsort(importance)[::-1]
        
        ranking = {}
        for idx in sorted_indices:
            ranking[feature_names[idx]] = float(importance[idx])
        
        return ranking
    
    def generate_explanation_text(self, X, sample_index: int, 
                                 feature_names=None) -> str:
        """
        Tạo explanation dạng text cho một sample
        
        Args:
            X: Features
            sample_index: Index của sample
            feature_names: Tên các features
        
        Returns:
            Explanation text
        """
        if self.shap_values is None:
            self.explain(X)
        
        sample_exp = self._explain_single_sample(X, sample_index)
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        text = []
        text.append(f"Giải thích cho Sample #{sample_index}:")
        text.append("\nCác features quan trọng nhất:")
        
        for i, feature_info in enumerate(sample_exp['top_features'][:5], 1):
            feat_idx = feature_info['feature_index']
            feat_name = feature_names[feat_idx]
            feat_val = feature_info['feature_value']
            shap_val = feature_info['shap_value']
            contribution = feature_info['contribution']
            
            impact = "tăng" if contribution == 'positive' else "giảm"
            
            text.append(
                f"  {i}. {feat_name} = {feat_val:.4f} "
                f"(SHAP: {shap_val:+.4f}, {impact} prediction)"
            )
        
        return "\n".join(text)

