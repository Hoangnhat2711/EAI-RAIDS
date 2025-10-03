"""
Unit Tests cho Adversarial Robustness Module
"""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from robustness.attack_generator import AttackGenerator
from robustness.defense_trainer import DefenseTrainer


class TestAttackGenerator:
    """Test AttackGenerator class"""
    
    @pytest.fixture
    def setup_attack(self):
        """Setup attack generator"""
        # Create simple model
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        attack_gen = AttackGenerator(model)
        
        return attack_gen, X[:20], y[:20]
    
    def test_fgsm_attack(self, setup_attack):
        """Test FGSM attack generation"""
        attack_gen, X, y = setup_attack
        
        X_adv, report = attack_gen.fgsm_attack(X, y, epsilon=0.3)
        
        assert X_adv.shape == X.shape
        assert 'success_rate' in report
        assert 'attack_type' in report
        assert report['attack_type'] == 'FGSM'
        assert 0 <= report['success_rate'] <= 1
    
    def test_pgd_attack(self, setup_attack):
        """Test PGD attack generation"""
        attack_gen, X, y = setup_attack
        
        X_adv, report = attack_gen.pgd_attack(X, y, epsilon=0.3, num_iter=10)
        
        assert X_adv.shape == X.shape
        assert report['attack_type'] == 'PGD'
        assert 0 <= report['success_rate'] <= 1
    
    def test_deepfool_attack(self, setup_attack):
        """Test DeepFool attack"""
        attack_gen, X, y = setup_attack
        
        X_adv, report = attack_gen.deepfool_attack(X[:10], max_iter=20)
        
        assert X_adv.shape == X[:10].shape
        assert report['attack_type'] == 'DeepFool'
    
    def test_robustness_evaluation(self, setup_attack):
        """Test comprehensive robustness evaluation"""
        attack_gen, X, y = setup_attack
        
        report = attack_gen.evaluate_robustness(X, y, attacks=['fgsm'])
        
        assert 'baseline_accuracy' in report
        assert 'attacks' in report
        assert isinstance(report['attacks'], dict)


class TestDefenseTrainer:
    """Test DefenseTrainer class"""
    
    @pytest.fixture
    def setup_defense(self):
        """Setup defense trainer"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        attack_gen = AttackGenerator(model)
        defense = DefenseTrainer(model, attack_gen)
        
        return defense, X[:50], y[:50], X[50:], y[50:]
    
    def test_adversarial_training(self, setup_defense):
        """Test adversarial training"""
        defense, X_train, y_train, X_test, y_test = setup_defense
        
        report = defense.adversarial_training(
            X_train, y_train,
            attack_config={'attack_type': 'fgsm', 'epsilon': 0.1},
            ratio=0.5,
            epochs=1
        )
        
        assert 'iterations' in report
        assert len(report['iterations']) == 1
        assert 'train_accuracy' in report['iterations'][0]
    
    def test_defense_effectiveness(self, setup_defense):
        """Test defense effectiveness evaluation"""
        defense, X_train, y_train, X_test, y_test = setup_defense
        
        # First train
        defense.adversarial_training(X_train, y_train, epochs=1)
        
        # Evaluate
        report = defense.evaluate_defense_effectiveness(X_test, y_test)
        
        assert 'baseline_accuracy' in report
        assert 'adversarial_accuracy' in report
        assert 'defense_effectiveness' in report
    
    def test_input_transformation(self, setup_defense):
        """Test input transformation defense"""
        defense, X_train, _, _, _ = setup_defense
        
        X_transformed = defense.input_transformation_defense(
            X_train, transformation='quantization'
        )
        
        assert X_transformed.shape == X_train.shape


def test_robustness_integration():
    """Integration test for robustness module"""
    # Create dataset
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test attacks
    attack_gen = AttackGenerator(model)
    robustness = attack_gen.evaluate_robustness(X_test, y_test, attacks=['fgsm'])
    
    assert robustness['baseline_accuracy'] > 0
    
    # Test defense
    defense = DefenseTrainer(model, attack_gen)
    defense.adversarial_training(X_train, y_train, epochs=1)
    
    effectiveness = defense.evaluate_defense_effectiveness(X_test, y_test)
    assert 'defense_effectiveness' in effectiveness


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

