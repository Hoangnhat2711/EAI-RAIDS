"""
Model Wrapper để tích hợp Responsible AI vào các ML models
"""

import numpy as np
from typing import Any, Optional, Dict
from datetime import datetime


class ResponsibleModelWrapper:
    """
    Wrapper cho ML models để đảm bảo tính trách nhiệm
    
    Wrapper này bọc bất kỳ ML model nào và thêm các tính năng:
    - Fairness monitoring
    - Explainability
    - Audit logging
    - Privacy protection
    """
    
    def __init__(self, model: Any, responsible_ai_framework: Any):
        """
        Khởi tạo Responsible Model Wrapper
        
        Args:
            model: Model ML cần wrap (sklearn, pytorch, tensorflow, etc.)
            responsible_ai_framework: Instance của ResponsibleAI
        """
        self.model = model
        self.rai = responsible_ai_framework
        self.is_fitted = False
        self.training_metadata = {}
        self.sensitive_features_names = []
        
        self.rai.logger.info(f"Đã wrap model: {type(model).__name__}")
    
    def fit(self, X, y, sensitive_features=None, **kwargs):
        """
        Train model với các kiểm tra trách nhiệm
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Các features nhạy cảm để kiểm tra fairness
            **kwargs: Các tham số khác cho model.fit()
        
        Returns:
            self
        """
        self.rai.logger.info("Bắt đầu training với Responsible AI checks...")
        
        # Lưu metadata
        self.training_metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]),
            'has_sensitive_features': sensitive_features is not None
        }
        
        # Kiểm tra privacy
        if self.rai.is_principle_active('privacy'):
            X = self._apply_privacy_protection(X)
        
        # Train model
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        # Ghi log
        if self.rai.is_principle_active('accountability'):
            self.rai.log_decision('model_training', self.training_metadata)
        
        self.rai.logger.info("Training hoàn tất!")
        
        return self
    
    def predict(self, X, log_predictions=True):
        """
        Dự đoán với audit logging
        
        Args:
            X: Features
            log_predictions: Có ghi log predictions không
        
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Hãy gọi fit() trước.")
        
        predictions = self.model.predict(X)
        
        # Log predictions
        if log_predictions and self.rai.is_principle_active('accountability'):
            self.rai.log_decision('prediction', {
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(X),
                'prediction_summary': {
                    'unique_values': len(np.unique(predictions)),
                    'shape': predictions.shape if hasattr(predictions, 'shape') else len(predictions)
                }
            })
        
        return predictions
    
    def predict_proba(self, X):
        """Dự đoán xác suất (nếu model hỗ trợ)"""
        if not self.is_fitted:
            raise ValueError("Model chưa được train. Hãy gọi fit() trước.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("Model không hỗ trợ predict_proba")
        
        return self.model.predict_proba(X)
    
    def evaluate_fairness(self, X, y_true, y_pred, sensitive_features):
        """
        Đánh giá fairness của model
        
        Args:
            X: Features
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Các features nhạy cảm
        
        Returns:
            Dictionary chứa các fairness metrics
        """
        if not self.rai.is_principle_active('fairness'):
            self.rai.logger.warning("Fairness principle không được kích hoạt")
            return {}
        
        from fairness.metrics import FairnessMetrics
        
        fairness_metrics = FairnessMetrics(self.rai)
        results = fairness_metrics.evaluate(y_true, y_pred, sensitive_features)
        
        # Log kết quả
        self.rai.log_decision('fairness_evaluation', {
            'timestamp': datetime.now().isoformat(),
            'metrics': results
        })
        
        return results
    
    def explain_prediction(self, X, method='shap', sample_index=0):
        """
        Giải thích predictions
        
        Args:
            X: Features
            method: Phương pháp giải thích ('shap' hoặc 'lime')
            sample_index: Index của sample cần giải thích
        
        Returns:
            Explanation object
        """
        if not self.rai.is_principle_active('explainability'):
            self.rai.logger.warning("Explainability principle không được kích hoạt")
            return None
        
        self.rai.logger.info(f"Tạo explanation bằng phương pháp: {method}")
        
        if method.lower() == 'shap':
            from explainability.shap_explainer import SHAPExplainer
            explainer = SHAPExplainer(self.model)
            return explainer.explain(X, sample_index)
        elif method.lower() == 'lime':
            from explainability.lime_explainer import LIMEExplainer
            explainer = LIMEExplainer(self.model)
            return explainer.explain(X, sample_index)
        else:
            raise ValueError(f"Phương pháp '{method}' không được hỗ trợ")
    
    def _apply_privacy_protection(self, X):
        """
        Áp dụng bảo vệ privacy cho dữ liệu
        
        Args:
            X: Features
        
        Returns:
            Protected features
        """
        privacy_config = self.rai.get_config('privacy')
        
        if privacy_config.get('differential_privacy', {}).get('enabled', False):
            self.rai.logger.info("Áp dụng Differential Privacy...")
            
            # Thêm noise theo differential privacy
            epsilon = privacy_config['differential_privacy'].get('epsilon', 1.0)
            
            # Chuyển đổi sang numpy array nếu cần
            X_array = np.array(X) if not isinstance(X, np.ndarray) else X
            
            # Thêm Laplace noise
            sensitivity = 1.0  # Giả định sensitivity = 1
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, X_array.shape)
            
            X_protected = X_array + noise
            
            self.rai.logger.info(f"Đã thêm noise với epsilon={epsilon}")
            return X_protected
        
        return X
    
    def generate_responsibility_report(self, X_test=None, y_test=None, sensitive_features=None) -> str:
        """
        Tạo báo cáo đầy đủ về tính trách nhiệm của model
        
        Args:
            X_test: Test features (optional)
            y_test: Test labels (optional)
            sensitive_features: Sensitive features (optional)
        
        Returns:
            Báo cáo dạng string
        """
        report = []
        report.append("=" * 70)
        report.append("BÁO CÁO TRÁCH NHIỆM MODEL")
        report.append("=" * 70)
        
        # Thông tin model
        report.append(f"\nModel: {type(self.model).__name__}")
        report.append(f"Trạng thái: {'Đã train' if self.is_fitted else 'Chưa train'}")
        
        if self.training_metadata:
            report.append("\nThông tin training:")
            report.append(f"  - Số lượng samples: {self.training_metadata.get('n_samples', 'N/A')}")
            report.append(f"  - Số lượng features: {self.training_metadata.get('n_features', 'N/A')}")
            report.append(f"  - Thời gian train: {self.training_metadata.get('timestamp', 'N/A')}")
        
        # Đánh giá fairness nếu có dữ liệu test
        if X_test is not None and y_test is not None and sensitive_features is not None:
            report.append("\nĐánh giá Fairness:")
            try:
                y_pred = self.predict(X_test, log_predictions=False)
                fairness_results = self.evaluate_fairness(X_test, y_test, y_pred, sensitive_features)
                
                for metric, value in fairness_results.items():
                    report.append(f"  - {metric}: {value:.4f}")
            except Exception as e:
                report.append(f"  Lỗi khi đánh giá fairness: {str(e)}")
        
        # Thông tin về principles
        report.append("\nCác nguyên tắc được áp dụng:")
        for principle in self.rai.get_active_principles():
            report.append(f"  ✓ {principle.capitalize()}")
        
        # Audit logs
        logs = self.rai.get_audit_logs(limit=5)
        if logs:
            report.append(f"\nAudit Logs (5 gần nhất):")
            for log in logs:
                report.append(f"  - {log['decision_type']} at {log['timestamp']}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def __repr__(self) -> str:
        return f"ResponsibleModelWrapper(model={type(self.model).__name__}, fitted={self.is_fitted})"

