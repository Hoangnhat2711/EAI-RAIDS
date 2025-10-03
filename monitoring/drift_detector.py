"""
Drift Detector - Phát hiện model drift và data drift
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import stats


class DriftDetector:
    """
    Phát hiện drift trong model và data
    
    Types of drift:
    - Data Drift: Input distribution thay đổi
    - Concept Drift: Relationship giữa input và output thay đổi
    - Prediction Drift: Output distribution thay đổi
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Khởi tạo Drift Detector
        
        Args:
            threshold: Significance threshold cho statistical tests
        """
        self.threshold = threshold
        self.reference_data = None
        self.reference_predictions = None
        self.drift_history = []
    
    def set_reference(self, X, y=None, predictions=None):
        """
        Set reference data để so sánh
        
        Args:
            X: Reference features
            y: Reference labels (optional)
            predictions: Reference predictions (optional)
        """
        self.reference_data = np.array(X)
        
        if predictions is not None:
            self.reference_predictions = np.array(predictions)
        
        print("Reference data đã được set")
    
    def detect_data_drift(self, X_new, method='ks') -> Dict[str, Any]:
        """
        Phát hiện data drift
        
        Args:
            X_new: New data để kiểm tra
            method: Statistical test method ('ks', 'chi2')
        
        Returns:
            Dictionary với kết quả drift detection
        """
        if self.reference_data is None:
            raise ValueError("Chưa set reference data. Gọi set_reference() trước.")
        
        X_new = np.array(X_new)
        
        drift_results = {
            'method': method,
            'drift_detected': False,
            'features_with_drift': [],
            'p_values': []
        }
        
        # Kiểm tra từng feature
        for feature_idx in range(self.reference_data.shape[1]):
            ref_feature = self.reference_data[:, feature_idx]
            new_feature = X_new[:, feature_idx]
            
            # Chọn statistical test
            if method.lower() == 'ks':
                statistic, p_value = stats.ks_2samp(ref_feature, new_feature)
            elif method.lower() == 'chi2':
                # Chi-squared test cho categorical
                # Đơn giản hóa: bin continuous data
                ref_hist, bin_edges = np.histogram(ref_feature, bins=10)
                new_hist, _ = np.histogram(new_feature, bins=bin_edges)
                
                # Avoid zero frequencies
                ref_hist = ref_hist + 1
                new_hist = new_hist + 1
                
                statistic, p_value = stats.chisquare(new_hist, ref_hist)
            else:
                raise ValueError(f"Method '{method}' không được hỗ trợ")
            
            drift_results['p_values'].append(p_value)
            
            # Nếu p_value < threshold, có drift
            if p_value < self.threshold:
                drift_results['drift_detected'] = True
                drift_results['features_with_drift'].append({
                    'feature_index': feature_idx,
                    'p_value': p_value,
                    'statistic': statistic
                })
        
        # Lưu vào history
        self.drift_history.append({
            'type': 'data_drift',
            'result': drift_results
        })
        
        return drift_results
    
    def detect_prediction_drift(self, predictions_new) -> Dict[str, Any]:
        """
        Phát hiện prediction drift
        
        Args:
            predictions_new: New predictions
        
        Returns:
            Dictionary với kết quả
        """
        if self.reference_predictions is None:
            raise ValueError("Chưa set reference predictions")
        
        predictions_new = np.array(predictions_new)
        
        # KS test cho predictions
        statistic, p_value = stats.ks_2samp(
            self.reference_predictions, 
            predictions_new
        )
        
        drift_detected = p_value < self.threshold
        
        # Tính distribution stats
        ref_mean = np.mean(self.reference_predictions)
        new_mean = np.mean(predictions_new)
        mean_shift = abs(new_mean - ref_mean)
        
        ref_std = np.std(self.reference_predictions)
        new_std = np.std(predictions_new)
        std_shift = abs(new_std - ref_std)
        
        result = {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic,
            'mean_shift': mean_shift,
            'std_shift': std_shift,
            'reference_mean': ref_mean,
            'new_mean': new_mean
        }
        
        self.drift_history.append({
            'type': 'prediction_drift',
            'result': result
        })
        
        return result
    
    def detect_concept_drift(self, X_new, y_new, model) -> Dict[str, Any]:
        """
        Phát hiện concept drift bằng cách so sánh model performance
        
        Args:
            X_new: New features
            y_new: New labels
            model: Model để evaluate
        
        Returns:
            Dictionary với kết quả
        """
        if self.reference_data is None:
            raise ValueError("Chưa set reference data")
        
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        # Predictions trên reference và new data
        ref_predictions = model.predict(self.reference_data)
        new_predictions = model.predict(X_new)
        
        # Tính accuracy/error trên reference (nếu có labels)
        # Simplified: so sánh prediction distributions
        
        # Sử dụng KS test
        statistic, p_value = stats.ks_2samp(ref_predictions, new_predictions)
        
        drift_detected = p_value < self.threshold
        
        result = {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic,
            'interpretation': 'Concept drift detected - relationship thay đổi' if drift_detected else 'Không có concept drift'
        }
        
        self.drift_history.append({
            'type': 'concept_drift',
            'result': result
        })
        
        return result
    
    def detect_all_drifts(self, X_new, y_new=None, 
                         predictions_new=None, 
                         model=None) -> Dict[str, Any]:
        """
        Phát hiện tất cả các loại drift
        
        Args:
            X_new: New features
            y_new: New labels (optional)
            predictions_new: New predictions (optional)
            model: Model (optional, cho concept drift)
        
        Returns:
            Dictionary với tất cả kết quả
        """
        results = {
            'data_drift': None,
            'prediction_drift': None,
            'concept_drift': None,
            'overall_drift_detected': False
        }
        
        # Data drift
        try:
            results['data_drift'] = self.detect_data_drift(X_new)
            if results['data_drift']['drift_detected']:
                results['overall_drift_detected'] = True
        except Exception as e:
            results['data_drift'] = {'error': str(e)}
        
        # Prediction drift
        if predictions_new is not None and self.reference_predictions is not None:
            try:
                results['prediction_drift'] = self.detect_prediction_drift(predictions_new)
                if results['prediction_drift']['drift_detected']:
                    results['overall_drift_detected'] = True
            except Exception as e:
                results['prediction_drift'] = {'error': str(e)}
        
        # Concept drift
        if model is not None and y_new is not None:
            try:
                results['concept_drift'] = self.detect_concept_drift(X_new, y_new, model)
                if results['concept_drift']['drift_detected']:
                    results['overall_drift_detected'] = True
            except Exception as e:
                results['concept_drift'] = {'error': str(e)}
        
        return results
    
    def get_drift_summary(self) -> str:
        """
        Lấy summary về drift history
        
        Returns:
            Summary string
        """
        if not self.drift_history:
            return "Chưa có drift checks nào"
        
        report = []
        report.append("=" * 60)
        report.append("DRIFT DETECTION SUMMARY")
        report.append("=" * 60)
        
        report.append(f"\nTổng số checks: {len(self.drift_history)}")
        
        # Count by type
        drift_types = {}
        for entry in self.drift_history:
            drift_type = entry['type']
            drift_types[drift_type] = drift_types.get(drift_type, 0) + 1
        
        report.append("\nPhân loại:")
        for drift_type, count in drift_types.items():
            report.append(f"  {drift_type}: {count}")
        
        # Count detections
        detections = sum(
            1 for entry in self.drift_history
            if entry['result'].get('drift_detected', False)
        )
        
        report.append(f"\nPhát hiện drift: {detections}/{len(self.drift_history)}")
        
        # Recent detections
        recent_drifts = [
            entry for entry in self.drift_history[-5:]
            if entry['result'].get('drift_detected', False)
        ]
        
        if recent_drifts:
            report.append("\nDrift gần đây:")
            for entry in recent_drifts:
                drift_type = entry['type']
                p_value = entry['result'].get('p_value', 'N/A')
                report.append(f"  • {drift_type} (p-value: {p_value})")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def calculate_psi(self, reference: np.ndarray, 
                     current: np.ndarray, 
                     bins: int = 10) -> float:
        """
        Tính Population Stability Index (PSI)
        
        PSI là một metric phổ biến để đo lường drift
        PSI < 0.1: No significant drift
        0.1 <= PSI < 0.25: Moderate drift
        PSI >= 0.25: Significant drift
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Số lượng bins
        
        Returns:
            PSI value
        """
        reference = np.array(reference)
        current = np.array(current)
        
        # Tạo bins từ reference
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Tính distribution cho cả hai
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / len(reference)
        curr_props = curr_counts / len(current)
        
        # Avoid log(0)
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        curr_props = np.where(curr_props == 0, 0.0001, curr_props)
        
        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return psi
    
    def monitor_features_psi(self, X_new) -> Dict[str, float]:
        """
        Tính PSI cho tất cả features
        
        Args:
            X_new: New data
        
        Returns:
            Dictionary với PSI cho từng feature
        """
        if self.reference_data is None:
            raise ValueError("Chưa set reference data")
        
        X_new = np.array(X_new)
        psi_scores = {}
        
        for feature_idx in range(self.reference_data.shape[1]):
            ref_feature = self.reference_data[:, feature_idx]
            new_feature = X_new[:, feature_idx]
            
            psi = self.calculate_psi(ref_feature, new_feature)
            psi_scores[f'feature_{feature_idx}'] = psi
        
        return psi_scores
    
    def __repr__(self) -> str:
        return f"DriftDetector(threshold={self.threshold}, checks={len(self.drift_history)})"

