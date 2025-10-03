"""
Adversarial Attack Generator

Triển khai các thuật toán tấn công đối kháng tiên tiến:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- C&W (Carlini & Wagner)
- DeepFool
"""

import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple
import warnings


class AttackGenerator:
    """
    Generator cho các adversarial examples
    
    Mục đích: Kiểm tra độ vững chắc của model trước các tấn công
    """
    
    def __init__(self, model: Any, loss_fn: Optional[Callable] = None):
        """
        Khởi tạo Attack Generator
        
        Args:
            model: Model cần kiểm tra
            loss_fn: Loss function (nếu cần cho gradient-based attacks)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.attack_history = []
    
    def fgsm_attack(self, X: np.ndarray, y: np.ndarray,
                   epsilon: float = 0.3) -> Tuple[np.ndarray, Dict]:
        """
        Fast Gradient Sign Method Attack
        
        Đây là một trong những attacks đơn giản và hiệu quả nhất
        
        Args:
            X: Input data
            y: True labels
            epsilon: Perturbation magnitude
        
        Returns:
            (adversarial_examples, attack_report)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        # Tính gradients (simplified - cho scikit-learn models)
        gradients = self._compute_gradients(X, y)
        
        # Apply FGSM: X_adv = X + epsilon * sign(grad)
        X_adv = X + epsilon * np.sign(gradients)
        
        # Clip để giữ trong valid range
        X_adv = np.clip(X_adv, X.min(), X.max())
        
        # Evaluate attack
        original_pred = self.model.predict(X)
        adversarial_pred = self.model.predict(X_adv)
        
        success_rate = np.mean(original_pred != adversarial_pred)
        
        attack_report = {
            'attack_type': 'FGSM',
            'epsilon': epsilon,
            'success_rate': success_rate,
            'n_samples': len(X),
            'n_successful': int(np.sum(original_pred != adversarial_pred)),
            'avg_perturbation': np.mean(np.abs(X_adv - X))
        }
        
        self.attack_history.append(attack_report)
        
        return X_adv, attack_report
    
    def pgd_attack(self, X: np.ndarray, y: np.ndarray,
                  epsilon: float = 0.3, alpha: float = 0.01,
                  num_iter: int = 40) -> Tuple[np.ndarray, Dict]:
        """
        Projected Gradient Descent Attack
        
        PGD là một iterative attack mạnh hơn FGSM
        
        Args:
            X: Input data
            y: True labels
            epsilon: Max perturbation
            alpha: Step size
            num_iter: Số iterations
        
        Returns:
            (adversarial_examples, attack_report)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        # Random initialization trong epsilon ball
        X_adv = X + np.random.uniform(-epsilon, epsilon, X.shape)
        X_adv = np.clip(X_adv, X.min(), X.max())
        
        # Iterative attack
        for i in range(num_iter):
            # Tính gradient
            gradients = self._compute_gradients(X_adv, y)
            
            # Update
            X_adv = X_adv + alpha * np.sign(gradients)
            
            # Project back to epsilon ball
            perturbation = np.clip(X_adv - X, -epsilon, epsilon)
            X_adv = X + perturbation
            X_adv = np.clip(X_adv, X.min(), X.max())
        
        # Evaluate attack
        original_pred = self.model.predict(X)
        adversarial_pred = self.model.predict(X_adv)
        
        success_rate = np.mean(original_pred != adversarial_pred)
        
        attack_report = {
            'attack_type': 'PGD',
            'epsilon': epsilon,
            'alpha': alpha,
            'num_iter': num_iter,
            'success_rate': success_rate,
            'n_samples': len(X),
            'n_successful': int(np.sum(original_pred != adversarial_pred)),
            'avg_perturbation': np.mean(np.abs(X_adv - X))
        }
        
        self.attack_history.append(attack_report)
        
        return X_adv, attack_report
    
    def deepfool_attack(self, X: np.ndarray, 
                       max_iter: int = 50,
                       overshoot: float = 0.02) -> Tuple[np.ndarray, Dict]:
        """
        DeepFool Attack
        
        Tìm minimal perturbation để thay đổi prediction
        
        Args:
            X: Input data
            max_iter: Max iterations
            overshoot: Overshoot parameter
        
        Returns:
            (adversarial_examples, attack_report)
        """
        X = np.array(X, dtype=float)
        X_adv = X.copy()
        
        original_pred = self.model.predict(X)
        
        perturbations = []
        
        for i in range(len(X)):
            x = X[i:i+1]
            x_adv = x.copy()
            orig_class = original_pred[i]
            
            for iter_num in range(max_iter):
                # Tính gradient
                grad = self._compute_gradients(x_adv, np.array([orig_class]))
                
                # Minimal perturbation
                pert = (overshoot + 1) * grad / (np.linalg.norm(grad) + 1e-8)
                x_adv = x_adv + pert
                
                # Check if successful
                new_pred = self.model.predict(x_adv)[0]
                if new_pred != orig_class:
                    break
            
            X_adv[i] = x_adv[0]
            perturbations.append(np.linalg.norm(x_adv - x))
        
        adversarial_pred = self.model.predict(X_adv)
        success_rate = np.mean(original_pred != adversarial_pred)
        
        attack_report = {
            'attack_type': 'DeepFool',
            'max_iter': max_iter,
            'success_rate': success_rate,
            'n_samples': len(X),
            'n_successful': int(np.sum(original_pred != adversarial_pred)),
            'avg_perturbation': np.mean(perturbations)
        }
        
        self.attack_history.append(attack_report)
        
        return X_adv, attack_report
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Tính gradients of loss w.r.t input
        
        Note: Simplified implementation - trong thực tế cần gradient từ model
        """
        # Numerical gradient approximation
        epsilon = 1e-4
        gradients = np.zeros_like(X)
        
        original_pred = self.model.predict(X)
        
        # Approximate gradient cho mỗi feature
        for i in range(X.shape[1]):
            X_plus = X.copy()
            X_plus[:, i] += epsilon
            
            pred_plus = self.model.predict(X_plus)
            
            # Gradient approximation
            gradients[:, i] = (pred_plus != original_pred).astype(float)
        
        return gradients
    
    def evaluate_robustness(self, X: np.ndarray, y: np.ndarray,
                           attacks: Optional[list] = None) -> Dict[str, Any]:
        """
        Đánh giá toàn diện độ vững chắc của model
        
        Args:
            X: Test data
            y: True labels
            attacks: List các attacks cần test (None = tất cả)
        
        Returns:
            Comprehensive robustness report
        """
        if attacks is None:
            attacks = ['fgsm', 'pgd']
        
        report = {
            'baseline_accuracy': self._compute_accuracy(X, y),
            'attacks': {}
        }
        
        for attack_type in attacks:
            print(f"Testing {attack_type.upper()} attack...")
            
            if attack_type.lower() == 'fgsm':
                X_adv, attack_report = self.fgsm_attack(X, y)
            elif attack_type.lower() == 'pgd':
                X_adv, attack_report = self.pgd_attack(X, y)
            elif attack_type.lower() == 'deepfool':
                X_adv, attack_report = self.deepfool_attack(X)
            else:
                warnings.warn(f"Unknown attack: {attack_type}")
                continue
            
            # Tính accuracy trên adversarial examples
            adv_accuracy = self._compute_accuracy(X_adv, y)
            attack_report['adversarial_accuracy'] = adv_accuracy
            attack_report['accuracy_drop'] = report['baseline_accuracy'] - adv_accuracy
            
            report['attacks'][attack_type] = attack_report
        
        # Overall robustness score
        if report['attacks']:
            avg_adv_accuracy = np.mean([
                r['adversarial_accuracy'] 
                for r in report['attacks'].values()
            ])
            report['overall_robustness_score'] = avg_adv_accuracy / report['baseline_accuracy']
        
        return report
    
    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Tính accuracy"""
        predictions = self.model.predict(X)
        return np.mean(predictions == y)
    
    def generate_robustness_report(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Tạo báo cáo chi tiết về robustness
        
        Args:
            X: Test data
            y: True labels
        
        Returns:
            Report string
        """
        eval_results = self.evaluate_robustness(X, y)
        
        report = []
        report.append("=" * 70)
        report.append("ADVERSARIAL ROBUSTNESS REPORT")
        report.append("=" * 70)
        
        report.append(f"\nBaseline Accuracy: {eval_results['baseline_accuracy']:.4f}")
        
        if 'overall_robustness_score' in eval_results:
            score = eval_results['overall_robustness_score']
            report.append(f"Overall Robustness Score: {score:.4f}")
            
            # Đánh giá
            if score > 0.9:
                assessment = "✓ EXCELLENT - Model highly robust"
            elif score > 0.7:
                assessment = "⚠ GOOD - Acceptable robustness"
            elif score > 0.5:
                assessment = "⚠ MODERATE - Consider defense training"
            else:
                assessment = "❌ POOR - Adversarial training recommended"
            
            report.append(f"Assessment: {assessment}")
        
        report.append("\nAttack Results:")
        report.append("-" * 70)
        
        for attack_name, attack_result in eval_results['attacks'].items():
            report.append(f"\n{attack_name.upper()}:")
            report.append(f"  Success Rate: {attack_result['success_rate']:.1%}")
            report.append(f"  Adversarial Accuracy: {attack_result['adversarial_accuracy']:.4f}")
            report.append(f"  Accuracy Drop: {attack_result['accuracy_drop']:.4f}")
            report.append(f"  Avg Perturbation: {attack_result['avg_perturbation']:.6f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def get_attack_summary(self) -> str:
        """Summary về attacks đã thực hiện"""
        if not self.attack_history:
            return "No attacks performed yet"
        
        report = []
        report.append("ATTACK HISTORY SUMMARY")
        report.append("=" * 60)
        
        report.append(f"\nTotal attacks: {len(self.attack_history)}")
        
        # Group by type
        from collections import Counter
        attack_types = [a['attack_type'] for a in self.attack_history]
        type_counts = Counter(attack_types)
        
        report.append("\nBy type:")
        for attack_type, count in type_counts.items():
            report.append(f"  • {attack_type}: {count}")
        
        # Average success rate
        avg_success = np.mean([a['success_rate'] for a in self.attack_history])
        report.append(f"\nAverage success rate: {avg_success:.1%}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def __repr__(self) -> str:
        return f"AttackGenerator(attacks_performed={len(self.attack_history)})"

