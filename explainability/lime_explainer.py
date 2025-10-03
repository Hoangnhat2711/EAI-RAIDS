"""
LIME Explainer - Sử dụng LIME (Local Interpretable Model-agnostic Explanations)
để giải thích predictions
"""

import numpy as np
from typing import Any, Optional, Dict
import warnings


class LIMEExplainer:
    """
    LIME-based explainer cho ML models
    
    LIME giải thích predictions bằng cách approximate model 
    với một interpretable model (như linear model) locally
    """
    
    def __init__(self, model: Any):
        """
        Khởi tạo LIME Explainer
        
        Args:
            model: ML model cần giải thích
        """
        self.model = model
        self.explainer = None
        
        # Kiểm tra xem lime có được cài đặt không
        try:
            import lime
            import lime.lime_tabular
            self.lime = lime
            self.is_available = True
        except ImportError:
            warnings.warn(
                "LIME chưa được cài đặt. Chạy: pip install lime",
                ImportWarning
            )
            self.is_available = False
    
    def explain(self, X, sample_index: int = 0, 
               num_features: int = 10) -> Dict[str, Any]:
        """
        Tạo LIME explanation cho một prediction
        
        Args:
            X: Features (training data để làm reference)
            sample_index: Index của sample cần giải thích
            num_features: Số lượng features quan trọng nhất
        
        Returns:
            Dictionary chứa explanation
        """
        if not self.is_available:
            return {
                'error': 'LIME không khả dụng. Cài đặt: pip install lime',
                'explanation': None
            }
        
        X = np.array(X)
        
        # Khởi tạo LIME explainer nếu chưa có
        if self.explainer is None:
            self._initialize_explainer(X)
        
        # Lấy sample cần giải thích
        instance = X[sample_index]
        
        # Tạo explanation
        try:
            # Xác định predict function
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            exp = self.explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features
            )
            
            explanation = {
                'method': 'LIME',
                'sample_index': sample_index,
                'feature_weights': dict(exp.as_list()),
                'prediction': exp.predict_proba,
                'local_pred': exp.local_pred,
                'score': exp.score
            }
            
            # Thêm top features
            explanation['top_features'] = self._parse_top_features(exp, num_features)
            
            return explanation
        
        except Exception as e:
            return {
                'error': f'Lỗi khi tạo LIME explanation: {str(e)}',
                'explanation': None
            }
    
    def _initialize_explainer(self, X):
        """Khởi tạo LIME explainer"""
        if not self.is_available:
            return
        
        # Khởi tạo LimeTabularExplainer
        self.explainer = self.lime.lime_tabular.LimeTabularExplainer(
            X,
            mode='classification',  # Hoặc 'regression'
            verbose=False
        )
    
    def _parse_top_features(self, exp, num_features: int) -> list:
        """Parse top features từ LIME explanation"""
        top_features = []
        
        for feature, weight in exp.as_list()[:num_features]:
            top_features.append({
                'feature': feature,
                'weight': float(weight),
                'contribution': 'positive' if weight > 0 else 'negative'
            })
        
        return top_features
    
    def explain_batch(self, X, sample_indices: list, 
                     num_features: int = 10) -> Dict[int, Dict]:
        """
        Tạo explanations cho nhiều samples
        
        Args:
            X: Features
            sample_indices: List các indices cần giải thích
            num_features: Số lượng features quan trọng nhất
        
        Returns:
            Dictionary mapping sample index -> explanation
        """
        explanations = {}
        
        for idx in sample_indices:
            explanations[idx] = self.explain(X, idx, num_features)
        
        return explanations
    
    def generate_explanation_text(self, X, sample_index: int,
                                 num_features: int = 5) -> str:
        """
        Tạo explanation dạng text
        
        Args:
            X: Features
            sample_index: Index của sample
            num_features: Số lượng features để hiển thị
        
        Returns:
            Explanation text
        """
        exp = self.explain(X, sample_index, num_features)
        
        if 'error' in exp:
            return f"Lỗi: {exp['error']}"
        
        text = []
        text.append(f"Giải thích LIME cho Sample #{sample_index}:")
        
        if 'local_pred' in exp:
            text.append(f"Prediction: {exp['local_pred']}")
        
        text.append("\nCác features quan trọng nhất:")
        
        for i, feature_info in enumerate(exp['top_features'], 1):
            feature = feature_info['feature']
            weight = feature_info['weight']
            contribution = feature_info['contribution']
            
            impact = "tăng" if contribution == 'positive' else "giảm"
            
            text.append(
                f"  {i}. {feature} "
                f"(weight: {weight:+.4f}, {impact} prediction)"
            )
        
        return "\n".join(text)
    
    def visualize(self, X, sample_index: int, num_features: int = 10):
        """
        Visualize LIME explanation (nếu có matplotlib)
        
        Args:
            X: Features
            sample_index: Index của sample
            num_features: Số lượng features
        """
        if not self.is_available:
            print("LIME không khả dụng")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib không khả dụng để visualization")
            return
        
        X = np.array(X)
        instance = X[sample_index]
        
        if self.explainer is None:
            self._initialize_explainer(X)
        
        try:
            predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
            
            exp = self.explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features
            )
            
            # Show in notebook hoặc save figure
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Không thể visualize: {e}")
    
    def get_feature_importance_ranking(self, X, sample_index: int,
                                      feature_names=None) -> Dict[str, float]:
        """
        Lấy ranking của feature importance cho một sample
        
        Args:
            X: Features
            sample_index: Index của sample
            feature_names: Tên các features
        
        Returns:
            Dictionary với feature names và importance scores
        """
        exp = self.explain(X, sample_index)
        
        if 'error' in exp:
            return {}
        
        return exp['feature_weights']
    
    def compare_samples(self, X, sample_indices: list,
                       num_features: int = 5) -> str:
        """
        So sánh explanations giữa nhiều samples
        
        Args:
            X: Features
            sample_indices: List các indices
            num_features: Số lượng features
        
        Returns:
            Comparison text
        """
        explanations = self.explain_batch(X, sample_indices, num_features)
        
        text = []
        text.append("So sánh LIME Explanations:")
        text.append("=" * 60)
        
        for idx, exp in explanations.items():
            if 'error' in exp:
                text.append(f"\nSample #{idx}: Lỗi - {exp['error']}")
                continue
            
            text.append(f"\nSample #{idx}:")
            for feature_info in exp['top_features'][:3]:
                text.append(f"  - {feature_info['feature']}: {feature_info['weight']:+.4f}")
        
        text.append("\n" + "=" * 60)
        
        return "\n".join(text)

