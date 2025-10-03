"""
Bias Detector - Phát hiện bias trong data và model
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter


class BiasDetector:
    """
    Phát hiện và phân tích bias trong dữ liệu và model predictions
    """
    
    def __init__(self, responsible_ai_framework: Any):
        """
        Khởi tạo Bias Detector
        
        Args:
            responsible_ai_framework: Instance của ResponsibleAI
        """
        self.rai = responsible_ai_framework
        self.bias_reports = []
    
    def detect_data_bias(self, X, y, sensitive_features, 
                        feature_names=None) -> Dict[str, Any]:
        """
        Phát hiện bias trong training data
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Sensitive features (dict hoặc array)
            feature_names: Tên các features
        
        Returns:
            Dictionary chứa kết quả phát hiện bias
        """
        X = np.array(X)
        y = np.array(y)
        
        bias_report = {
            'timestamp': self.rai.creation_time.isoformat(),
            'bias_detected': False,
            'biases': []
        }
        
        # 1. Kiểm tra class imbalance
        class_imbalance = self._check_class_imbalance(y)
        if class_imbalance['is_imbalanced']:
            bias_report['bias_detected'] = True
            bias_report['biases'].append({
                'type': 'class_imbalance',
                'severity': 'medium',
                'details': class_imbalance
            })
        
        # 2. Kiểm tra representation bias trong sensitive features
        if sensitive_features is not None:
            sensitive_features = np.array(sensitive_features)
            representation_bias = self._check_representation_bias(
                sensitive_features, y
            )
            if representation_bias['is_biased']:
                bias_report['bias_detected'] = True
                bias_report['biases'].append({
                    'type': 'representation_bias',
                    'severity': 'high',
                    'details': representation_bias
                })
        
        # 3. Kiểm tra label bias
        if sensitive_features is not None:
            label_bias = self._check_label_bias(y, sensitive_features)
            if label_bias['is_biased']:
                bias_report['bias_detected'] = True
                bias_report['biases'].append({
                    'type': 'label_bias',
                    'severity': 'high',
                    'details': label_bias
                })
        
        self.bias_reports.append(bias_report)
        
        return bias_report
    
    def _check_class_imbalance(self, y) -> Dict[str, Any]:
        """Kiểm tra class imbalance"""
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        # Tính imbalance ratio
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        is_imbalanced = imbalance_ratio > 2.0  # Ngưỡng 2:1
        
        return {
            'is_imbalanced': is_imbalanced,
            'imbalance_ratio': imbalance_ratio,
            'class_distribution': class_distribution,
            'recommendation': 'Xem xét resampling hoặc class weights' if is_imbalanced else None
        }
    
    def _check_representation_bias(self, sensitive_features, y) -> Dict[str, Any]:
        """Kiểm tra representation bias trong sensitive groups"""
        unique_groups, group_counts = np.unique(sensitive_features, return_counts=True)
        group_distribution = dict(zip(unique_groups, group_counts))
        
        # Kiểm tra sự chênh lệch về representation
        max_count = max(group_counts)
        min_count = min(group_counts)
        representation_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        is_biased = representation_ratio > 3.0  # Ngưỡng 3:1
        
        # Phân tích label distribution cho mỗi group
        group_label_dist = {}
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_labels = y[group_mask]
            unique_labels, label_counts = np.unique(group_labels, return_counts=True)
            group_label_dist[str(group)] = dict(zip(
                [str(l) for l in unique_labels], 
                [int(c) for c in label_counts]
            ))
        
        return {
            'is_biased': is_biased,
            'representation_ratio': representation_ratio,
            'group_distribution': {str(k): int(v) for k, v in group_distribution.items()},
            'group_label_distribution': group_label_dist,
            'recommendation': 'Cân bằng representation giữa các nhóm' if is_biased else None
        }
    
    def _check_label_bias(self, y, sensitive_features) -> Dict[str, Any]:
        """Kiểm tra label bias giữa các sensitive groups"""
        unique_groups = np.unique(sensitive_features)
        
        # Tính positive rate cho mỗi group
        positive_rates = {}
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_labels = y[group_mask]
            
            # Chuyển về binary nếu cần
            if group_labels.dtype == float:
                positive_rate = np.mean(group_labels > 0.5)
            else:
                positive_rate = np.mean(group_labels == 1)
            
            positive_rates[str(group)] = positive_rate
        
        # Kiểm tra sự chênh lệch
        rates = list(positive_rates.values())
        max_rate = max(rates) if rates else 0
        min_rate = min(rates) if rates else 0
        
        rate_diff = max_rate - min_rate
        is_biased = rate_diff > 0.2  # Ngưỡng 20%
        
        return {
            'is_biased': is_biased,
            'positive_rates': positive_rates,
            'rate_difference': rate_diff,
            'recommendation': 'Kiểm tra labeling process cho bias' if is_biased else None
        }
    
    def detect_prediction_bias(self, y_true, y_pred, 
                              sensitive_features) -> Dict[str, Any]:
        """
        Phát hiện bias trong model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Sensitive features
        
        Returns:
            Dictionary chứa kết quả phát hiện bias
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_features = np.array(sensitive_features)
        
        bias_report = {
            'timestamp': self.rai.creation_time.isoformat(),
            'bias_detected': False,
            'biases': []
        }
        
        # 1. Kiểm tra prediction disparity
        prediction_disparity = self._check_prediction_disparity(
            y_pred, sensitive_features
        )
        if prediction_disparity['is_biased']:
            bias_report['bias_detected'] = True
            bias_report['biases'].append({
                'type': 'prediction_disparity',
                'severity': 'high',
                'details': prediction_disparity
            })
        
        # 2. Kiểm tra error rate disparity
        error_disparity = self._check_error_disparity(
            y_true, y_pred, sensitive_features
        )
        if error_disparity['is_biased']:
            bias_report['bias_detected'] = True
            bias_report['biases'].append({
                'type': 'error_rate_disparity',
                'severity': 'high',
                'details': error_disparity
            })
        
        self.bias_reports.append(bias_report)
        
        return bias_report
    
    def _check_prediction_disparity(self, y_pred, 
                                   sensitive_features) -> Dict[str, Any]:
        """Kiểm tra sự chênh lệch trong predictions giữa các groups"""
        unique_groups = np.unique(sensitive_features)
        
        prediction_rates = {}
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_preds = y_pred[group_mask]
            
            if group_preds.dtype == float:
                pred_rate = np.mean(group_preds > 0.5)
            else:
                pred_rate = np.mean(group_preds == 1)
            
            prediction_rates[str(group)] = pred_rate
        
        rates = list(prediction_rates.values())
        max_rate = max(rates) if rates else 0
        min_rate = min(rates) if rates else 0
        
        disparity = max_rate - min_rate
        is_biased = disparity > 0.2  # Ngưỡng 20%
        
        return {
            'is_biased': is_biased,
            'prediction_rates': prediction_rates,
            'disparity': disparity,
            'recommendation': 'Áp dụng fairness constraints hoặc post-processing' if is_biased else None
        }
    
    def _check_error_disparity(self, y_true, y_pred, 
                              sensitive_features) -> Dict[str, Any]:
        """Kiểm tra sự chênh lệch về error rates giữa các groups"""
        unique_groups = np.unique(sensitive_features)
        
        error_rates = {}
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_true = y_true[group_mask]
            group_pred = y_pred[group_mask]
            
            # Chuyển về binary
            if group_true.dtype == float:
                group_true_binary = group_true > 0.5
                group_pred_binary = group_pred > 0.5
            else:
                group_true_binary = group_true == 1
                group_pred_binary = group_pred == 1
            
            error_rate = np.mean(group_true_binary != group_pred_binary)
            error_rates[str(group)] = error_rate
        
        rates = list(error_rates.values())
        max_rate = max(rates) if rates else 0
        min_rate = min(rates) if rates else 0
        
        disparity = max_rate - min_rate
        is_biased = disparity > 0.15  # Ngưỡng 15%
        
        return {
            'is_biased': is_biased,
            'error_rates': error_rates,
            'disparity': disparity,
            'recommendation': 'Cải thiện performance cho nhóm có error rate cao' if is_biased else None
        }
    
    def generate_bias_report(self) -> str:
        """
        Tạo báo cáo tổng hợp về bias
        
        Returns:
            Báo cáo dạng string
        """
        if not self.bias_reports:
            return "Chưa có báo cáo bias nào. Hãy chạy detect_data_bias() hoặc detect_prediction_bias() trước."
        
        report = []
        report.append("=" * 70)
        report.append("BÁO CÁO PHÁT HIỆN BIAS")
        report.append("=" * 70)
        
        total_biases = sum(1 for r in self.bias_reports if r['bias_detected'])
        report.append(f"\nTổng số lần phát hiện bias: {total_biases}/{len(self.bias_reports)}")
        
        for idx, bias_report in enumerate(self.bias_reports, 1):
            report.append(f"\n--- Báo cáo #{idx} ---")
            report.append(f"Thời gian: {bias_report['timestamp']}")
            
            if bias_report['bias_detected']:
                report.append(f"Trạng thái: ⚠ Phát hiện {len(bias_report['biases'])} loại bias")
                
                for bias in bias_report['biases']:
                    report.append(f"\n  Loại: {bias['type']}")
                    report.append(f"  Mức độ: {bias['severity'].upper()}")
                    
                    details = bias['details']
                    if 'recommendation' in details and details['recommendation']:
                        report.append(f"  Đề xuất: {details['recommendation']}")
            else:
                report.append("Trạng thái: ✓ Không phát hiện bias")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)

