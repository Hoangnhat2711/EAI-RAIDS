"""
Mitigation Engine - Tự động phát hiện và xử lý các vấn đề về Bias, Drift

Đây là tính năng tiên tiến giúp framework không chỉ phát hiện mà còn 
TỰ ĐỘNG HÀNH ĐỘNG để giảm thiểu các vấn đề
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from sklearn.utils.class_weight import compute_class_weight
import warnings


class MitigationEngine:
    """
    Engine tự động áp dụng các kỹ thuật mitigation
    
    Các kỹ thuật được hỗ trợ:
    - Class Reweighting
    - Resampling (Over/Under sampling)
    - Fairness Constraints
    - Prediction Post-processing
    - Auto-retraining on drift
    """
    
    def __init__(self, responsible_ai_framework: Any):
        """
        Khởi tạo Mitigation Engine
        
        Args:
            responsible_ai_framework: Instance của ResponsibleAI
        """
        self.rai = responsible_ai_framework
        self.mitigation_history = []
        
        # Configuration
        self.config = self.rai.get_config('fairness')
        self.auto_mitigate = self.config.get('auto_mitigate', True)
        
        self.rai.logger.info("Mitigation Engine initialized")
    
    def analyze_and_mitigate_bias(self, X, y, sensitive_features=None) -> Tuple[Any, Any, Dict]:
        """
        Phân tích và tự động mitigate bias trong data
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Sensitive attributes
        
        Returns:
            (X_mitigated, y_mitigated, mitigation_report)
        """
        from fairness.bias_detector import BiasDetector
        
        X_processed = np.array(X)
        y_processed = np.array(y)
        
        mitigation_report = {
            'techniques_applied': [],
            'original_distribution': self._get_class_distribution(y),
            'bias_detected': False
        }
        
        # Phát hiện bias
        bias_detector = BiasDetector(self.rai)
        bias_report = bias_detector.detect_data_bias(X, y, sensitive_features)
        
        if not bias_report['bias_detected']:
            mitigation_report['message'] = "No significant bias detected"
            return X_processed, y_processed, mitigation_report
        
        mitigation_report['bias_detected'] = True
        mitigation_report['biases'] = bias_report['biases']
        
        # Áp dụng mitigation techniques
        for bias in bias_report['biases']:
            bias_type = bias['type']
            
            if bias_type == 'class_imbalance':
                self.rai.logger.info("Applying mitigation for class imbalance...")
                X_processed, y_processed = self._mitigate_class_imbalance(
                    X_processed, y_processed, bias['details']
                )
                mitigation_report['techniques_applied'].append('class_rebalancing')
            
            elif bias_type == 'representation_bias':
                if sensitive_features is not None:
                    self.rai.logger.info("Applying mitigation for representation bias...")
                    X_processed, y_processed = self._mitigate_representation_bias(
                        X_processed, y_processed, sensitive_features
                    )
                    mitigation_report['techniques_applied'].append('representation_reweighting')
        
        mitigation_report['final_distribution'] = self._get_class_distribution(y_processed)
        
        # Log mitigation
        self.mitigation_history.append(mitigation_report)
        
        return X_processed, y_processed, mitigation_report
    
    def _mitigate_class_imbalance(self, X, y, imbalance_details: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Xử lý class imbalance bằng resampling
        
        Args:
            X: Features
            y: Labels
            imbalance_details: Chi tiết về imbalance
        
        Returns:
            (X_resampled, y_resampled)
        """
        imbalance_ratio = imbalance_details.get('imbalance_ratio', 1.0)
        
        # Nếu imbalance không nghiêm trọng, không cần resample
        if imbalance_ratio < 2.0:
            return X, y
        
        # Chọn strategy dựa trên mức độ imbalance
        if imbalance_ratio < 5.0:
            # Moderate imbalance: Random oversampling
            return self._random_oversample(X, y)
        else:
            # Severe imbalance: SMOTE hoặc combined sampling
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                self.rai.logger.info(f"Applied SMOTE: {len(y)} -> {len(y_resampled)} samples")
                return X_resampled, y_resampled
            except ImportError:
                self.rai.logger.warning("imblearn not installed, using random oversampling")
                return self._random_oversample(X, y)
    
    def _random_oversample(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Random oversampling của minority class"""
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_count = max(class_counts)
        
        X_resampled = []
        y_resampled = []
        
        for cls in unique_classes:
            # Get samples của class này
            class_indices = np.where(y == cls)[0]
            class_X = X[class_indices]
            class_y = y[class_indices]
            
            # Oversample đến max_count
            n_samples_needed = max_count - len(class_indices)
            if n_samples_needed > 0:
                oversample_indices = np.random.choice(
                    len(class_indices), 
                    size=n_samples_needed, 
                    replace=True
                )
                class_X = np.vstack([class_X, class_X[oversample_indices]])
                class_y = np.hstack([class_y, class_y[oversample_indices]])
            
            X_resampled.append(class_X)
            y_resampled.append(class_y)
        
        X_final = np.vstack(X_resampled)
        y_final = np.hstack(y_resampled)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_final))
        
        self.rai.logger.info(f"Applied random oversampling: {len(y)} -> {len(y_final)} samples")
        
        return X_final[shuffle_idx], y_final[shuffle_idx]
    
    def _mitigate_representation_bias(self, X, y, sensitive_features) -> Tuple[np.ndarray, np.ndarray]:
        """
        Xử lý representation bias trong sensitive groups
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Sensitive attributes
        
        Returns:
            (X_balanced, y_balanced)
        """
        sensitive_features = np.array(sensitive_features)
        unique_groups = np.unique(sensitive_features)
        
        # Tính target count (average của tất cả groups)
        group_counts = [np.sum(sensitive_features == g) for g in unique_groups]
        target_count = int(np.mean(group_counts))
        
        X_balanced = []
        y_balanced = []
        
        for group in unique_groups:
            group_indices = np.where(sensitive_features == group)[0]
            group_X = X[group_indices]
            group_y = y[group_indices]
            
            current_count = len(group_indices)
            
            if current_count < target_count:
                # Oversample
                n_needed = target_count - current_count
                oversample_idx = np.random.choice(current_count, size=n_needed, replace=True)
                group_X = np.vstack([group_X, group_X[oversample_idx]])
                group_y = np.hstack([group_y, group_y[oversample_idx]])
            elif current_count > target_count:
                # Undersample
                undersample_idx = np.random.choice(current_count, size=target_count, replace=False)
                group_X = group_X[undersample_idx]
                group_y = group_y[undersample_idx]
            
            X_balanced.append(group_X)
            y_balanced.append(group_y)
        
        X_final = np.vstack(X_balanced)
        y_final = np.hstack(y_balanced)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_final))
        
        self.rai.logger.info(f"Balanced representation: {len(y)} -> {len(y_final)} samples")
        
        return X_final[shuffle_idx], y_final[shuffle_idx]
    
    def compute_sample_weights(self, y, sensitive_features=None) -> np.ndarray:
        """
        Tính sample weights để xử lý imbalance
        
        Args:
            y: Labels
            sensitive_features: Sensitive attributes (optional)
        
        Returns:
            Sample weights array
        """
        y = np.array(y)
        
        # Base weights từ class imbalance
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        weights = np.array([class_weight_dict[label] for label in y])
        
        # Nếu có sensitive features, adjust weights
        if sensitive_features is not None:
            sensitive_features = np.array(sensitive_features)
            unique_groups = np.unique(sensitive_features)
            
            # Tính representation weights
            group_counts = np.array([np.sum(sensitive_features == g) for g in unique_groups])
            avg_count = np.mean(group_counts)
            group_weights = avg_count / group_counts
            group_weight_dict = dict(zip(unique_groups, group_weights))
            
            # Combine weights
            representation_weights = np.array([
                group_weight_dict[group] for group in sensitive_features
            ])
            
            weights = weights * representation_weights
        
        # Normalize
        weights = weights / np.mean(weights)
        
        self.rai.logger.info(f"Computed sample weights (mean: {np.mean(weights):.3f}, std: {np.std(weights):.3f})")
        
        return weights
    
    def postprocess_predictions(self, y_pred, sensitive_features, 
                               fairness_constraint: str = 'demographic_parity') -> np.ndarray:
        """
        Post-process predictions để đảm bảo fairness
        
        Args:
            y_pred: Raw predictions
            sensitive_features: Sensitive attributes
            fairness_constraint: Loại fairness constraint
        
        Returns:
            Adjusted predictions
        """
        y_pred = np.array(y_pred)
        sensitive_features = np.array(sensitive_features)
        
        if fairness_constraint == 'demographic_parity':
            return self._enforce_demographic_parity(y_pred, sensitive_features)
        elif fairness_constraint == 'equal_opportunity':
            # Cần thêm y_true để implement
            self.rai.logger.warning("Equal opportunity requires y_true, using demographic parity")
            return self._enforce_demographic_parity(y_pred, sensitive_features)
        else:
            self.rai.logger.warning(f"Unknown constraint: {fairness_constraint}")
            return y_pred
    
    def _enforce_demographic_parity(self, y_pred, sensitive_features) -> np.ndarray:
        """
        Điều chỉnh predictions để đạt demographic parity
        
        Strategy: Equalize positive rates across groups
        """
        unique_groups = np.unique(sensitive_features)
        
        # Tính positive rate cho mỗi group
        positive_rates = []
        for group in unique_groups:
            group_mask = sensitive_features == group
            group_preds = y_pred[group_mask]
            pos_rate = np.mean(group_preds > 0.5) if group_preds.dtype == float else np.mean(group_preds == 1)
            positive_rates.append(pos_rate)
        
        # Target: average positive rate
        target_rate = np.mean(positive_rates)
        
        y_pred_adjusted = y_pred.copy()
        
        # Adjust mỗi group
        for group, current_rate in zip(unique_groups, positive_rates):
            group_mask = sensitive_features == group
            
            if abs(current_rate - target_rate) < 0.05:  # Đã đủ fair
                continue
            
            group_preds = y_pred_adjusted[group_mask]
            
            if current_rate > target_rate:
                # Giảm positive rate: flip một số positives thành negatives
                n_to_flip = int((current_rate - target_rate) * len(group_preds))
                positive_indices = np.where(group_preds > 0.5)[0]
                
                if len(positive_indices) > n_to_flip:
                    flip_indices = np.random.choice(positive_indices, size=n_to_flip, replace=False)
                    group_preds[flip_indices] = 0 if group_preds.dtype == int else 0.0
            else:
                # Tăng positive rate: flip một số negatives thành positives
                n_to_flip = int((target_rate - current_rate) * len(group_preds))
                negative_indices = np.where(group_preds <= 0.5)[0]
                
                if len(negative_indices) > n_to_flip:
                    flip_indices = np.random.choice(negative_indices, size=n_to_flip, replace=False)
                    group_preds[flip_indices] = 1 if group_preds.dtype == int else 1.0
            
            y_pred_adjusted[group_mask] = group_preds
        
        self.rai.logger.info(f"Applied demographic parity post-processing (target rate: {target_rate:.3f})")
        
        return y_pred_adjusted
    
    def _get_class_distribution(self, y) -> Dict[str, int]:
        """Lấy distribution của classes"""
        unique, counts = np.unique(y, return_counts=True)
        return {str(cls): int(count) for cls, count in zip(unique, counts)}
    
    def get_mitigation_summary(self) -> str:
        """
        Lấy summary về các mitigation đã áp dụng
        
        Returns:
            Summary string
        """
        if not self.mitigation_history:
            return "No mitigation applied yet"
        
        report = []
        report.append("=" * 60)
        report.append("MITIGATION SUMMARY")
        report.append("=" * 60)
        
        report.append(f"\nTotal mitigations: {len(self.mitigation_history)}")
        
        # Count techniques
        all_techniques = []
        for m in self.mitigation_history:
            all_techniques.extend(m.get('techniques_applied', []))
        
        from collections import Counter
        technique_counts = Counter(all_techniques)
        
        report.append("\nTechniques applied:")
        for technique, count in technique_counts.items():
            report.append(f"  • {technique}: {count} times")
        
        # Recent mitigation
        if self.mitigation_history:
            recent = self.mitigation_history[-1]
            report.append("\nMost recent mitigation:")
            report.append(f"  Bias detected: {recent['bias_detected']}")
            if recent['bias_detected']:
                report.append(f"  Techniques: {', '.join(recent['techniques_applied'])}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def __repr__(self) -> str:
        return f"MitigationEngine(mitigations_applied={len(self.mitigation_history)})"

