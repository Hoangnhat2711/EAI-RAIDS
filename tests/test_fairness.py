"""
Unit Tests cho Fairness Module
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from core.responsible_ai import ResponsibleAI
from fairness.metrics import FairnessMetrics
from fairness.bias_detector import BiasDetector


class TestFairnessMetrics:
    """Test FairnessMetrics class"""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data"""
        np.random.seed(42)
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        sensitive = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        rai = ResponsibleAI()
        metrics = FairnessMetrics(rai)
        
        return metrics, y_true, y_pred, sensitive
    
    def test_demographic_parity(self, setup_data):
        """Test demographic parity calculation"""
        metrics, _, y_pred, sensitive = setup_data
        
        score = metrics.demographic_parity(y_pred, sensitive)
        
        assert 0 <= score <= 1, "Score should be between 0 and 1"
        assert isinstance(score, (float, np.floating))
    
    def test_equalized_odds(self, setup_data):
        """Test equalized odds calculation"""
        metrics, y_true, y_pred, sensitive = setup_data
        
        score = metrics.equalized_odds(y_true, y_pred, sensitive)
        
        assert 0 <= score <= 1
        assert isinstance(score, (float, np.floating))
    
    def test_equal_opportunity(self, setup_data):
        """Test equal opportunity calculation"""
        metrics, y_true, y_pred, sensitive = setup_data
        
        score = metrics.equal_opportunity(y_true, y_pred, sensitive)
        
        assert 0 <= score <= 1
        assert isinstance(score, (float, np.floating))
    
    def test_disparate_impact(self, setup_data):
        """Test disparate impact calculation"""
        metrics, _, y_pred, sensitive = setup_data
        
        score = metrics.disparate_impact(y_pred, sensitive)
        
        assert score >= 0  # Can be > 1
        assert isinstance(score, (float, np.floating))
    
    def test_evaluate_all_metrics(self, setup_data):
        """Test comprehensive evaluation"""
        metrics, y_true, y_pred, sensitive = setup_data
        
        results = metrics.evaluate(y_true, y_pred, sensitive)
        
        assert isinstance(results, dict)
        assert 'overall_fairness_score' in results
        assert all(isinstance(v, (float, np.floating)) for v in results.values())


class TestBiasDetector:
    """Test BiasDetector class"""
    
    @pytest.fixture
    def setup_detector(self):
        """Setup bias detector"""
        rai = ResponsibleAI()
        detector = BiasDetector(rai)
        return detector
    
    def test_detect_class_imbalance(self, setup_detector):
        """Test class imbalance detection"""
        # Create imbalanced data
        y = np.array([0]*80 + [1]*20)
        
        result = setup_detector._check_class_imbalance(y)
        
        assert result['is_imbalanced'] == True
        assert result['imbalance_ratio'] == 4.0
        assert 'recommendation' in result
    
    def test_detect_data_bias(self, setup_detector):
        """Test comprehensive data bias detection"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.array([0]*70 + [1]*30)
        sensitive = np.random.randint(0, 2, 100)
        
        report = setup_detector.detect_data_bias(X, y, sensitive)
        
        assert isinstance(report, dict)
        assert 'bias_detected' in report
        assert 'biases' in report
    
    def test_detect_prediction_bias(self, setup_detector):
        """Test prediction bias detection"""
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        sensitive = np.random.randint(0, 2, 100)
        
        report = setup_detector.detect_prediction_bias(y_true, y_pred, sensitive)
        
        assert isinstance(report, dict)
        assert 'bias_detected' in report


def test_fairness_integration():
    """Integration test for fairness module"""
    # Create dataset
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    sensitive = np.random.randint(0, 2, 200)
    
    # Initialize
    rai = ResponsibleAI()
    metrics = FairnessMetrics(rai)
    detector = BiasDetector(rai)
    
    # Detect bias
    bias_report = detector.detect_data_bias(X, y, sensitive)
    assert isinstance(bias_report, dict)
    
    # Evaluate fairness (using y as predictions for simplicity)
    fairness_results = metrics.evaluate(y, y, sensitive)
    assert 'overall_fairness_score' in fairness_results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

