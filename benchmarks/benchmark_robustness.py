"""
Benchmark cho Adversarial Robustness

So sánh model có và không có adversarial training
"""

import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

from robustness.attack_generator import AttackGenerator
from robustness.defense_trainer import DefenseTrainer


class RobustnessBenchmark:
    """
    Benchmark adversarial robustness
    
    Compare:
    - Vanilla model vs Adversarially trained model
    - Different attack methods
    - Different defense strategies
    """
    
    def __init__(self):
        self.results = {}
    
    def run_benchmark(self, n_samples: int = 1000, n_features: int = 20):
        """
        Run comprehensive robustness benchmark
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
        """
        print("=" * 80)
        print("ADVERSARIAL ROBUSTNESS BENCHMARK")
        print("=" * 80)
        
        # Create dataset
        print(f"\n📊 Creating dataset: {n_samples} samples, {n_features} features")
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Benchmark 1: Vanilla Model
        print("\n" + "="*80)
        print("1. VANILLA MODEL (No Defense)")
        print("="*80)
        vanilla_results = self._benchmark_vanilla_model(X_train, y_train, X_test, y_test)
        self.results['vanilla'] = vanilla_results
        
        # Benchmark 2: Adversarially Trained Model
        print("\n" + "="*80)
        print("2. ADVERSARIALLY TRAINED MODEL")
        print("="*80)
        adv_trained_results = self._benchmark_adversarial_training(
            X_train, y_train, X_test, y_test
        )
        self.results['adversarial_trained'] = adv_trained_results
        
        # Comparison
        print("\n" + "="*80)
        print("3. COMPARISON & ANALYSIS")
        print("="*80)
        self._print_comparison()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _benchmark_vanilla_model(self, X_train, y_train, X_test, y_test):
        """Benchmark vanilla model"""
        # Train
        print("\n🔧 Training vanilla model...")
        start_time = time.time()
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Baseline accuracy
        baseline_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  ✓ Baseline Accuracy: {baseline_acc:.4f}")
        print(f"  ⏱ Training Time: {train_time:.2f}s")
        
        # Test against attacks
        print("\n🔍 Testing against adversarial attacks...")
        attack_gen = AttackGenerator(model)
        
        # FGSM
        print("  Testing FGSM...")
        X_adv_fgsm, fgsm_report = attack_gen.fgsm_attack(X_test, y_test, epsilon=0.3)
        fgsm_acc = accuracy_score(y_test, model.predict(X_adv_fgsm))
        print(f"    Attack Success Rate: {fgsm_report['success_rate']:.1%}")
        print(f"    Adversarial Accuracy: {fgsm_acc:.4f}")
        
        # PGD
        print("  Testing PGD...")
        X_adv_pgd, pgd_report = attack_gen.pgd_attack(X_test, y_test, epsilon=0.3, num_iter=20)
        pgd_acc = accuracy_score(y_test, model.predict(X_adv_pgd))
        print(f"    Attack Success Rate: {pgd_report['success_rate']:.1%}")
        print(f"    Adversarial Accuracy: {pgd_acc:.4f}")
        
        return {
            'baseline_accuracy': baseline_acc,
            'train_time': train_time,
            'fgsm': {
                'success_rate': fgsm_report['success_rate'],
                'adversarial_accuracy': fgsm_acc
            },
            'pgd': {
                'success_rate': pgd_report['success_rate'],
                'adversarial_accuracy': pgd_acc
            }
        }
    
    def _benchmark_adversarial_training(self, X_train, y_train, X_test, y_test):
        """Benchmark adversarially trained model"""
        # Train vanilla first
        print("\n🔧 Training base model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Adversarial training
        print("\n🛡️ Applying adversarial training...")
        start_time = time.time()
        
        attack_gen = AttackGenerator(model)
        defense = DefenseTrainer(model, attack_gen)
        
        defense.adversarial_training(
            X_train[:200], y_train[:200],  # Subset for speed
            attack_config={'attack_type': 'fgsm', 'epsilon': 0.3},
            ratio=0.5,
            epochs=3
        )
        
        train_time = time.time() - start_time
        
        # Baseline accuracy
        baseline_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  ✓ Baseline Accuracy: {baseline_acc:.4f}")
        print(f"  ⏱ Adversarial Training Time: {train_time:.2f}s")
        
        # Test against attacks
        print("\n🔍 Testing robustness after adversarial training...")
        
        # FGSM
        print("  Testing FGSM...")
        X_adv_fgsm, fgsm_report = attack_gen.fgsm_attack(X_test, y_test, epsilon=0.3)
        fgsm_acc = accuracy_score(y_test, model.predict(X_adv_fgsm))
        print(f"    Attack Success Rate: {fgsm_report['success_rate']:.1%}")
        print(f"    Adversarial Accuracy: {fgsm_acc:.4f}")
        
        # PGD
        print("  Testing PGD...")
        X_adv_pgd, pgd_report = attack_gen.pgd_attack(X_test, y_test, epsilon=0.3, num_iter=20)
        pgd_acc = accuracy_score(y_test, model.predict(X_adv_pgd))
        print(f"    Attack Success Rate: {pgd_report['success_rate']:.1%}")
        print(f"    Adversarial Accuracy: {pgd_acc:.4f}")
        
        return {
            'baseline_accuracy': baseline_acc,
            'train_time': train_time,
            'fgsm': {
                'success_rate': fgsm_report['success_rate'],
                'adversarial_accuracy': fgsm_acc
            },
            'pgd': {
                'success_rate': pgd_report['success_rate'],
                'adversarial_accuracy': pgd_acc
            }
        }
    
    def _print_comparison(self):
        """Print comparison results"""
        vanilla = self.results['vanilla']
        adv_trained = self.results['adversarial_trained']
        
        print("\n📊 RESULTS COMPARISON")
        print("-" * 80)
        
        # Baseline accuracy
        print(f"\nBaseline Accuracy:")
        print(f"  Vanilla:              {vanilla['baseline_accuracy']:.4f}")
        print(f"  Adversarial Trained:  {adv_trained['baseline_accuracy']:.4f}")
        print(f"  Difference:           {adv_trained['baseline_accuracy'] - vanilla['baseline_accuracy']:+.4f}")
        
        # FGSM
        print(f"\nFGSM Attack (ε=0.3):")
        print(f"  Vanilla Success Rate:             {vanilla['fgsm']['success_rate']:.1%}")
        print(f"  Adversarial Trained Success Rate: {adv_trained['fgsm']['success_rate']:.1%}")
        print(f"  Improvement:                      {(vanilla['fgsm']['success_rate'] - adv_trained['fgsm']['success_rate']):.1%}")
        
        print(f"\n  Vanilla Adv Accuracy:             {vanilla['fgsm']['adversarial_accuracy']:.4f}")
        print(f"  Adversarial Trained Adv Accuracy: {adv_trained['fgsm']['adversarial_accuracy']:.4f}")
        print(f"  Improvement:                      {adv_trained['fgsm']['adversarial_accuracy'] - vanilla['fgsm']['adversarial_accuracy']:+.4f}")
        
        # PGD
        print(f"\nPGD Attack (ε=0.3, 20 iterations):")
        print(f"  Vanilla Success Rate:             {vanilla['pgd']['success_rate']:.1%}")
        print(f"  Adversarial Trained Success Rate: {adv_trained['pgd']['success_rate']:.1%}")
        print(f"  Improvement:                      {(vanilla['pgd']['success_rate'] - adv_trained['pgd']['success_rate']):.1%}")
        
        print(f"\n  Vanilla Adv Accuracy:             {vanilla['pgd']['adversarial_accuracy']:.4f}")
        print(f"  Adversarial Trained Adv Accuracy: {adv_trained['pgd']['adversarial_accuracy']:.4f}")
        print(f"  Improvement:                      {adv_trained['pgd']['adversarial_accuracy'] - vanilla['pgd']['adversarial_accuracy']:+.4f}")
        
        # Training time
        print(f"\nTraining Time:")
        print(f"  Vanilla:              {vanilla['train_time']:.2f}s")
        print(f"  Adversarial Training: {adv_trained['train_time']:.2f}s (additional)")
        
        # Overall assessment
        print("\n" + "="*80)
        print("📋 ASSESSMENT")
        print("="*80)
        
        fgsm_improvement = adv_trained['fgsm']['adversarial_accuracy'] - vanilla['fgsm']['adversarial_accuracy']
        pgd_improvement = adv_trained['pgd']['adversarial_accuracy'] - vanilla['pgd']['adversarial_accuracy']
        
        if fgsm_improvement > 0.1 and pgd_improvement > 0.05:
            print("✅ EXCELLENT: Adversarial training significantly improved robustness")
        elif fgsm_improvement > 0.05:
            print("✓ GOOD: Adversarial training improved robustness")
        else:
            print("⚠ MODERATE: Limited improvement from adversarial training")
        
        print(f"\nKey Findings:")
        print(f"  • Adversarial training reduced FGSM success rate by {(vanilla['fgsm']['success_rate'] - adv_trained['fgsm']['success_rate'])*100:.1f}%")
        print(f"  • Adversarial training reduced PGD success rate by {(vanilla['pgd']['success_rate'] - adv_trained['pgd']['success_rate'])*100:.1f}%")
        print(f"  • Model maintains {adv_trained['baseline_accuracy']:.1%} clean accuracy")
    
    def _save_results(self):
        """Save results to JSON"""
        filename = 'benchmark_results_robustness.json'
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


def run_benchmark():
    """Run benchmark"""
    benchmark = RobustnessBenchmark()
    results = benchmark.run_benchmark(n_samples=500, n_features=20)
    return results


if __name__ == '__main__':
    run_benchmark()

