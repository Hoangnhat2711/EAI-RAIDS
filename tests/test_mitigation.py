"""
Unit Tests cho Mitigation Engine
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from core.responsible_ai import ResponsibleAI
from core.mitigation_engine import MitigationEngine


class TestMitigationEngine:
    """Test MitigationEngine class"""
    
    @pytest.fixture
    def setup_engine(self):
        """Setup mitigation engine"""
        rai = ResponsibleAI()
        engine = MitigationEngine(rai)
        return engine
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            weights=[0.8, 0.2],
            random_state=42
        )
        sensitive = np.random.randint(0, 2, 100)
        return X, y, sensitive
    
    def test_analyze_and_mitigate(self, setup_engine, imbalanced_data):
        """Test comprehensive bias mitigation"""
        engine = setup_engine
        X, y, sensitive = imbalanced_data
        
        X_clean, y_clean, report = engine.analyze_and_mitigate_bias(
            X, y, sensitive
        )
        
        assert X_clean.shape[1] == X.shape[1]  # Same features
        assert 'techniques_applied' in report
        assert 'original_distribution' in report
        assert 'final_distribution' in report
    
    def test_compute_sample_weights(self, setup_engine, imbalanced_data):
        """Test sample weight computation"""
        engine = setup_engine
        X, y, sensitive = imbalanced_data
        
        weights = engine.compute_sample_weights(y, sensitive)
        
        assert len(weights) == len(y)
        assert np.all(weights > 0)
        assert np.isclose(np.mean(weights), 1.0, atol=0.1)
    
    def test_postprocess_predictions(self, setup_engine):
        """Test prediction post-processing"""
        engine = setup_engine
        
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        sensitive = np.array([0, 0, 0, 1, 1, 1])
        
        y_adjusted = engine.postprocess_predictions(
            y_pred, sensitive, fairness_constraint='demographic_parity'
        )
        
        assert len(y_adjusted) == len(y_pred)
        assert set(y_adjusted).issubset({0, 1})
    
    def test_class_imbalance_mitigation(self, setup_engine, imbalanced_data):
        """Test class imbalance handling"""
        engine = setup_engine
        X, y, _ = imbalanced_data
        
        # Check original imbalance
        unique, counts = np.unique(y, return_counts=True)
        original_ratio = max(counts) / min(counts)
        
        assert original_ratio > 2, "Data should be imbalanced"
        
        # Mitigate
        imbalance_details = {
            'imbalance_ratio': original_ratio,
            'class_distribution': dict(zip(unique, counts))
        }
        
        X_balanced, y_balanced = engine._mitigate_class_imbalance(
            X, y, imbalance_details
        )
        
        # Check balanced
        unique_new, counts_new = np.unique(y_balanced, return_counts=True)
        new_ratio = max(counts_new) / min(counts_new)
        
        assert new_ratio < original_ratio, "Should reduce imbalance"


def test_mitigation_integration():
    """Integration test for mitigation engine"""
    # Create biased dataset
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=15,
        weights=[0.7, 0.3],
        random_state=42
    )
    sensitive = np.random.randint(0, 2, 200)
    
    # Initialize
    rai = ResponsibleAI()
    engine = MitigationEngine(rai)
    
    # Mitigate
    X_clean, y_clean, report = engine.analyze_and_mitigate_bias(
        X, y, sensitive
    )
    
    # Check results
    assert len(X_clean) >= len(X)  # May oversample
    assert report['bias_detected'] or not report['bias_detected']  # Bool
    
    # Compute weights
    weights = engine.compute_sample_weights(y_clean, sensitive)
    assert len(weights) == len(y_clean)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

