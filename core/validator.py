"""
Compliance Validator - Xác thực tuân thủ các nguyên tắc AI có trách nhiệm
"""

from typing import Dict, List, Any, Tuple
import numpy as np


class ComplianceValidator:
    """
    Validator để kiểm tra tuân thủ các nguyên tắc Responsible AI
    """
    
    def __init__(self, responsible_ai_framework: Any):
        """
        Khởi tạo Compliance Validator
        
        Args:
            responsible_ai_framework: Instance của ResponsibleAI
        """
        self.rai = responsible_ai_framework
        self.validation_results = {}
    
    def validate_all(self, model_wrapper=None, X=None, y=None, 
                     sensitive_features=None) -> Dict[str, Any]:
        """
        Thực hiện tất cả các kiểm tra tuân thủ
        
        Args:
            model_wrapper: ResponsibleModelWrapper instance
            X: Features
            y: Labels
            sensitive_features: Sensitive features
        
        Returns:
            Dictionary chứa kết quả validation
        """
        results = {
            'overall_compliance': True,
            'checks': {}
        }
        
        # Kiểm tra Fairness
        if self.rai.is_principle_active('fairness'):
            fairness_check = self._validate_fairness(
                model_wrapper, X, y, sensitive_features
            )
            results['checks']['fairness'] = fairness_check
            if not fairness_check['passed']:
                results['overall_compliance'] = False
        
        # Kiểm tra Transparency
        if self.rai.is_principle_active('transparency'):
            transparency_check = self._validate_transparency(model_wrapper)
            results['checks']['transparency'] = transparency_check
            if not transparency_check['passed']:
                results['overall_compliance'] = False
        
        # Kiểm tra Accountability
        if self.rai.is_principle_active('accountability'):
            accountability_check = self._validate_accountability()
            results['checks']['accountability'] = accountability_check
            if not accountability_check['passed']:
                results['overall_compliance'] = False
        
        # Kiểm tra Explainability
        if self.rai.is_principle_active('explainability'):
            explainability_check = self._validate_explainability(model_wrapper)
            results['checks']['explainability'] = explainability_check
            if not explainability_check['passed']:
                results['overall_compliance'] = False
        
        # Kiểm tra Privacy
        if self.rai.is_principle_active('privacy'):
            privacy_check = self._validate_privacy()
            results['checks']['privacy'] = privacy_check
            if not privacy_check['passed']:
                results['overall_compliance'] = False
        
        self.validation_results = results
        return results
    
    def _validate_fairness(self, model_wrapper, X, y, sensitive_features) -> Dict[str, Any]:
        """Kiểm tra tuân thủ nguyên tắc Fairness"""
        result = {
            'principle': 'fairness',
            'passed': True,
            'score': 1.0,
            'details': [],
            'warnings': []
        }
        
        if model_wrapper is None or X is None or y is None:
            result['warnings'].append("Không đủ dữ liệu để kiểm tra fairness")
            return result
        
        if sensitive_features is None:
            result['warnings'].append("Không có sensitive features để kiểm tra")
            result['score'] = 0.5
            return result
        
        try:
            # Kiểm tra có bias không
            if hasattr(model_wrapper, 'is_fitted') and model_wrapper.is_fitted:
                y_pred = model_wrapper.predict(X, log_predictions=False)
                
                # Kiểm tra distribution của predictions
                unique_vals, counts = np.unique(y_pred, return_counts=True)
                distribution_balance = min(counts) / max(counts) if len(counts) > 0 else 1.0
                
                result['score'] = distribution_balance
                result['details'].append(f"Distribution balance: {distribution_balance:.4f}")
                
                fairness_threshold = self.rai.get_config('fairness').get('threshold', 0.8)
                if distribution_balance < fairness_threshold:
                    result['passed'] = False
                    result['warnings'].append(
                        f"Distribution balance ({distribution_balance:.4f}) thấp hơn ngưỡng ({fairness_threshold})"
                    )
            else:
                result['warnings'].append("Model chưa được train")
                result['score'] = 0.0
        
        except Exception as e:
            result['passed'] = False
            result['warnings'].append(f"Lỗi khi kiểm tra fairness: {str(e)}")
        
        return result
    
    def _validate_transparency(self, model_wrapper) -> Dict[str, Any]:
        """Kiểm tra tuân thủ nguyên tắc Transparency"""
        result = {
            'principle': 'transparency',
            'passed': True,
            'score': 1.0,
            'details': [],
            'warnings': []
        }
        
        if model_wrapper is None:
            result['warnings'].append("Không có model để kiểm tra")
            return result
        
        # Kiểm tra có metadata không
        if hasattr(model_wrapper, 'training_metadata'):
            if model_wrapper.training_metadata:
                result['details'].append("Model có training metadata đầy đủ")
                result['score'] = 1.0
            else:
                result['warnings'].append("Model thiếu training metadata")
                result['score'] = 0.5
        
        # Kiểm tra model có thể giải thích được không
        model_type = type(model_wrapper.model).__name__
        interpretable_models = ['LogisticRegression', 'DecisionTreeClassifier', 
                               'LinearRegression', 'Ridge', 'Lasso']
        
        if model_type in interpretable_models:
            result['details'].append(f"Model type '{model_type}' có tính interpretable cao")
        else:
            result['warnings'].append(f"Model type '{model_type}' có thể khó giải thích")
            result['score'] *= 0.8
        
        return result
    
    def _validate_accountability(self) -> Dict[str, Any]:
        """Kiểm tra tuân thủ nguyên tắc Accountability"""
        result = {
            'principle': 'accountability',
            'passed': True,
            'score': 1.0,
            'details': [],
            'warnings': []
        }
        
        # Kiểm tra audit logging có được bật không
        audit_config = self.rai.get_config('audit')
        if not audit_config.get('enabled', False):
            result['passed'] = False
            result['warnings'].append("Audit logging không được bật")
            result['score'] = 0.0
            return result
        
        # Kiểm tra có audit logs không
        logs = self.rai.get_audit_logs()
        if len(logs) > 0:
            result['details'].append(f"Có {len(logs)} audit logs được ghi nhận")
            result['score'] = 1.0
        else:
            result['warnings'].append("Chưa có audit logs nào")
            result['score'] = 0.5
        
        return result
    
    def _validate_explainability(self, model_wrapper) -> Dict[str, Any]:
        """Kiểm tra tuân thủ nguyên tắc Explainability"""
        result = {
            'principle': 'explainability',
            'passed': True,
            'score': 1.0,
            'details': [],
            'warnings': []
        }
        
        if model_wrapper is None:
            result['warnings'].append("Không có model để kiểm tra")
            return result
        
        # Kiểm tra explainability methods có được cấu hình không
        explainability_config = self.rai.get_config('explainability')
        methods = explainability_config.get('methods', [])
        
        if len(methods) > 0:
            result['details'].append(f"Các phương pháp explainability: {', '.join(methods)}")
            result['score'] = 1.0
        else:
            result['warnings'].append("Không có explainability methods nào được cấu hình")
            result['score'] = 0.5
        
        return result
    
    def _validate_privacy(self) -> Dict[str, Any]:
        """Kiểm tra tuân thủ nguyên tắc Privacy"""
        result = {
            'principle': 'privacy',
            'passed': True,
            'score': 1.0,
            'details': [],
            'warnings': []
        }
        
        privacy_config = self.rai.get_config('privacy')
        
        # Kiểm tra differential privacy
        dp_config = privacy_config.get('differential_privacy', {})
        if dp_config.get('enabled', False):
            epsilon = dp_config.get('epsilon', 0)
            result['details'].append(f"Differential Privacy được bật với epsilon={epsilon}")
            
            # Kiểm tra epsilon có trong ngưỡng an toàn không
            if epsilon > 10:
                result['warnings'].append(f"Epsilon ({epsilon}) cao, có thể ảnh hưởng privacy")
                result['score'] *= 0.8
        else:
            result['warnings'].append("Differential Privacy không được bật")
            result['score'] = 0.5
        
        # Kiểm tra anonymization
        if privacy_config.get('data_anonymization', False):
            result['details'].append("Data anonymization được bật")
        
        # Kiểm tra PII detection
        if privacy_config.get('pii_detection', False):
            result['details'].append("PII detection được bật")
        
        return result
    
    def generate_compliance_report(self) -> str:
        """
        Tạo báo cáo tuân thủ
        
        Returns:
            Báo cáo dạng string
        """
        if not self.validation_results:
            return "Chưa có kết quả validation. Hãy gọi validate_all() trước."
        
        report = []
        report.append("=" * 70)
        report.append("BÁO CÁO TUÂN THỦ RESPONSIBLE AI")
        report.append("=" * 70)
        
        overall = self.validation_results.get('overall_compliance', False)
        status = "ĐẠT" if overall else "KHÔNG ĐẠT"
        icon = "✓" if overall else "✗"
        
        report.append(f"\nTrạng thái tổng thể: {icon} {status}")
        report.append("\nChi tiết từng nguyên tắc:")
        report.append("-" * 70)
        
        for principle, result in self.validation_results.get('checks', {}).items():
            status = "✓ ĐẠT" if result['passed'] else "✗ KHÔNG ĐẠT"
            score = result['score']
            
            report.append(f"\n{principle.upper()}: {status} (Score: {score:.2f})")
            
            if result['details']:
                report.append("  Chi tiết:")
                for detail in result['details']:
                    report.append(f"    • {detail}")
            
            if result['warnings']:
                report.append("  Cảnh báo:")
                for warning in result['warnings']:
                    report.append(f"    ⚠ {warning}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def get_recommendations(self) -> List[str]:
        """
        Lấy các đề xuất cải thiện
        
        Returns:
            Danh sách các đề xuất
        """
        recommendations = []
        
        if not self.validation_results:
            return ["Hãy chạy validate_all() để nhận đề xuất"]
        
        for principle, result in self.validation_results.get('checks', {}).items():
            if not result['passed'] or result['score'] < 0.8:
                if principle == 'fairness':
                    recommendations.append(
                        "Cải thiện Fairness: Xem xét rebalancing dataset hoặc áp dụng fairness constraints"
                    )
                elif principle == 'transparency':
                    recommendations.append(
                        "Cải thiện Transparency: Thêm documentation và metadata cho model"
                    )
                elif principle == 'accountability':
                    recommendations.append(
                        "Cải thiện Accountability: Bật audit logging và ghi nhận tất cả quyết định"
                    )
                elif principle == 'explainability':
                    recommendations.append(
                        "Cải thiện Explainability: Cấu hình SHAP hoặc LIME explainers"
                    )
                elif principle == 'privacy':
                    recommendations.append(
                        "Cải thiện Privacy: Bật Differential Privacy hoặc data anonymization"
                    )
        
        if not recommendations:
            recommendations.append("Tuyệt vời! Tất cả các nguyên tắc đều được tuân thủ tốt.")
        
        return recommendations

