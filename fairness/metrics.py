"""
Fairness Metrics - Các metrics để đánh giá tính công bằng
"""

import numpy as np
from typing import Dict, Any, List
from collections import Counter


class FairnessMetrics:
    """
    Tính toán các metrics về fairness/công bằng
    
    Các metrics được hỗ trợ:
    - Demographic Parity: Tỷ lệ dự đoán positive ngang nhau giữa các nhóm
    - Equalized Odds: True positive rate và false positive rate ngang nhau
    - Equal Opportunity: True positive rate ngang nhau
    - Disparate Impact: Tỷ lệ giữa positive rate của nhóm protected và unprivileged
    """
    
    def __init__(self, responsible_ai_framework: Any):
        """
        Khởi tạo Fairness Metrics
        
        Args:
            responsible_ai_framework: Instance của ResponsibleAI
        """
        self.rai = responsible_ai_framework
        self.fairness_config = self.rai.get_config('fairness')
    
    def evaluate(self, y_true, y_pred, sensitive_features) -> Dict[str, float]:
        """
        Đánh giá tất cả các fairness metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive features (e.g., gender, race)
        
        Returns:
            Dictionary chứa các fairness metrics
        """
        results = {}
        
        # Chuyển đổi sang numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_features = np.array(sensitive_features)
        
        configured_metrics = self.fairness_config.get('metrics', [])
        
        if 'demographic_parity' in configured_metrics:
            results['demographic_parity'] = self.demographic_parity(y_pred, sensitive_features)
        
        if 'equalized_odds' in configured_metrics:
            results['equalized_odds'] = self.equalized_odds(y_true, y_pred, sensitive_features)
        
        if 'equal_opportunity' in configured_metrics:
            results['equal_opportunity'] = self.equal_opportunity(y_true, y_pred, sensitive_features)
        
        if 'disparate_impact' in configured_metrics:
            results['disparate_impact'] = self.disparate_impact(y_pred, sensitive_features)
        
        # Tính overall fairness score
        if results:
            results['overall_fairness_score'] = np.mean(list(results.values()))
        
        return results
    
    def demographic_parity(self, y_pred, sensitive_features) -> float:
        """
        Demographic Parity: Tỷ lệ predictions positive phải gần bằng nhau
        giữa các nhóm sensitive
        
        Args:
            y_pred: Predicted labels
            sensitive_features: Sensitive features
        
        Returns:
            Score (1.0 = perfect parity, 0.0 = complete disparity)
        """
        unique_groups = np.unique(sensitive_features)
        positive_rates = []
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_preds = y_pred[group_mask]
            
            if len(group_preds) > 0:
                positive_rate = np.mean(group_preds > 0.5 if group_preds.dtype == float else group_preds == 1)
                positive_rates.append(positive_rate)
        
        if len(positive_rates) < 2:
            return 1.0  # Không thể đánh giá với 1 nhóm
        
        # Tính sự chênh lệch
        max_rate = max(positive_rates)
        min_rate = min(positive_rates)
        
        if max_rate == 0:
            return 1.0
        
        # Trả về ratio (1.0 = perfect parity)
        parity_score = min_rate / max_rate if max_rate > 0 else 0.0
        
        return parity_score
    
    def equalized_odds(self, y_true, y_pred, sensitive_features) -> float:
        """
        Equalized Odds: TPR và FPR phải gần bằng nhau giữa các nhóm
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive features
        
        Returns:
            Score (1.0 = perfect equality)
        """
        unique_groups = np.unique(sensitive_features)
        tpr_list = []
        fpr_list = []
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            if len(y_true_group) == 0:
                continue
            
            # Chuyển về binary
            y_true_binary = (y_true_group > 0.5 if y_true_group.dtype == float else y_true_group == 1)
            y_pred_binary = (y_pred_group > 0.5 if y_pred_group.dtype == float else y_pred_group == 1)
            
            # True Positive Rate
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_list.append(tpr)
            
            # False Positive Rate
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_list.append(fpr)
        
        if len(tpr_list) < 2:
            return 1.0
        
        # Tính difference
        tpr_diff = max(tpr_list) - min(tpr_list)
        fpr_diff = max(fpr_list) - min(fpr_list)
        
        # Score càng cao càng fair (1 - difference)
        avg_diff = (tpr_diff + fpr_diff) / 2
        score = max(0, 1 - avg_diff)
        
        return score
    
    def equal_opportunity(self, y_true, y_pred, sensitive_features) -> float:
        """
        Equal Opportunity: TPR phải gần bằng nhau giữa các nhóm
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive features
        
        Returns:
            Score (1.0 = perfect equality)
        """
        unique_groups = np.unique(sensitive_features)
        tpr_list = []
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            if len(y_true_group) == 0:
                continue
            
            # Chuyển về binary
            y_true_binary = (y_true_group > 0.5 if y_true_group.dtype == float else y_true_group == 1)
            y_pred_binary = (y_pred_group > 0.5 if y_pred_group.dtype == float else y_pred_group == 1)
            
            # True Positive Rate
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_list.append(tpr)
        
        if len(tpr_list) < 2:
            return 1.0
        
        # Tính ratio
        max_tpr = max(tpr_list)
        min_tpr = min(tpr_list)
        
        score = min_tpr / max_tpr if max_tpr > 0 else 1.0
        
        return score
    
    def disparate_impact(self, y_pred, sensitive_features) -> float:
        """
        Disparate Impact: Tỷ lệ giữa positive rate của unprivileged group
        và privileged group
        
        Args:
            y_pred: Predicted labels
            sensitive_features: Sensitive features
        
        Returns:
            Score (0.8-1.25 thường được coi là acceptable)
        """
        unique_groups = np.unique(sensitive_features)
        
        if len(unique_groups) < 2:
            return 1.0
        
        positive_rates = []
        
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_preds = y_pred[group_mask]
            
            if len(group_preds) > 0:
                positive_rate = np.mean(group_preds > 0.5 if group_preds.dtype == float else group_preds == 1)
                positive_rates.append(positive_rate)
        
        if len(positive_rates) < 2:
            return 1.0
        
        # Disparate impact = min_rate / max_rate
        max_rate = max(positive_rates)
        min_rate = min(positive_rates)
        
        if max_rate == 0:
            return 1.0
        
        di_score = min_rate / max_rate
        
        return di_score
    
    def check_fairness_compliance(self, results: Dict[str, float]) -> Dict[str, Any]:
        """
        Kiểm tra xem metrics có đạt ngưỡng fairness không
        
        Args:
            results: Dictionary chứa fairness metrics
        
        Returns:
            Dictionary với compliance status
        """
        threshold = self.fairness_config.get('threshold', 0.8)
        
        compliance = {
            'passed': True,
            'threshold': threshold,
            'failed_metrics': []
        }
        
        for metric, value in results.items():
            if metric == 'overall_fairness_score':
                continue
            
            if value < threshold:
                compliance['passed'] = False
                compliance['failed_metrics'].append({
                    'metric': metric,
                    'value': value,
                    'threshold': threshold
                })
        
        return compliance
    
    def generate_fairness_report(self, y_true, y_pred, sensitive_features) -> str:
        """
        Tạo báo cáo chi tiết về fairness
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive features
        
        Returns:
            Báo cáo dạng string
        """
        results = self.evaluate(y_true, y_pred, sensitive_features)
        compliance = self.check_fairness_compliance(results)
        
        report = []
        report.append("=" * 60)
        report.append("BÁO CÁO FAIRNESS")
        report.append("=" * 60)
        
        status = "✓ ĐẠT" if compliance['passed'] else "✗ KHÔNG ĐẠT"
        report.append(f"\nTrạng thái: {status}")
        report.append(f"Ngưỡng fairness: {compliance['threshold']}")
        
        report.append("\nCác metrics:")
        for metric, value in results.items():
            icon = "✓" if value >= compliance['threshold'] else "✗"
            report.append(f"  {icon} {metric}: {value:.4f}")
        
        if compliance['failed_metrics']:
            report.append("\nCác metrics không đạt:")
            for failed in compliance['failed_metrics']:
                report.append(
                    f"  • {failed['metric']}: {failed['value']:.4f} "
                    f"(ngưỡng: {failed['threshold']:.4f})"
                )
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

