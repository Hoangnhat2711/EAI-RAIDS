"""
ADVANCED DEMO - Responsible AI Framework với tính năng nâng cao

Demo này showcase các tính năng tiên tiến:
1. Mitigation Engine - Tự động xử lý bias
2. Adversarial Robustness Testing
3. Multi-handler Audit Logging
4. Alert Manager
5. Framework Adapters
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import Responsible AI Framework
from core.responsible_ai import ResponsibleAI
from core.model_wrapper import ResponsibleModelWrapper
from core.mitigation_engine import MitigationEngine
from core.alert_manager import AlertManager, AlertLevel
from core.adapters import SklearnAdapter
from fairness.bias_detector import BiasDetector
from robustness.attack_generator import AttackGenerator
from robustness.defense_trainer import DefenseTrainer
from audit.logger import AuditLogger
from audit.handlers import FileLogHandler, ElasticsearchHandler, PostgreSQLHandler
from monitoring.drift_detector import DriftDetector


def print_section(title):
    """Helper để print section headers"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def create_biased_data():
    """Tạo dataset có bias để demo mitigation"""
    print("📊 Tạo biased dataset...")
    
    # Generate imbalanced data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced
        random_state=42
    )
    
    # Tạo sensitive feature (gender: 0/1)
    sensitive_features = np.random.randint(0, 2, size=1000)
    
    # Introduce bias: class 1 overrepresented trong group 0
    for i in range(len(y)):
        if sensitive_features[i] == 0 and np.random.random() < 0.3:
            y[i] = 1
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    sensitive_train = sensitive_features[:800]
    sensitive_test = sensitive_features[800:]
    
    print(f"✓ Dataset created: {len(X_train)} train, {len(X_test)} test samples")
    print(f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test


def main():
    """Main advanced demo"""
    
    print("\n" + "🚀" * 40)
    print("ADVANCED RESPONSIBLE AI FRAMEWORK DEMO")
    print("🚀" * 40)
    
    # ===== 1. SETUP =====
    print_section("1️⃣  FRAMEWORK INITIALIZATION")
    
    # Initialize framework
    rai = ResponsibleAI(config_path='config.yaml')
    print(f"✓ Framework initialized with principles: {', '.join(rai.get_active_principles())}")
    
    # Initialize Alert Manager
    alert_config = {
        'channels': {
            'console': {'enabled': True},
            # 'email': {'enabled': False},
            # 'slack': {'enabled': False}
        },
        'fairness_threshold': 0.8,
        'drift_threshold': 0.1
    }
    alert_manager = AlertManager(alert_config)
    print("✓ Alert Manager initialized")
    
    # Initialize Audit Logger với multiple handlers
    handlers = [
        FileLogHandler({'log_dir': 'audit_logs', 'log_file': 'advanced_demo.jsonl'}),
        # ElasticsearchHandler({'hosts': ['http://localhost:9200']}),  # Optional
        # PostgreSQLHandler({'database': 'responsible_ai'})  # Optional
    ]
    audit_logger = AuditLogger(log_dir='audit_logs', handlers=handlers)
    print(f"✓ Audit Logger with {len(handlers)} handler(s)")
    
    # ===== 2. DATA & BIAS DETECTION =====
    print_section("2️⃣  DATA LOADING & BIAS DETECTION")
    
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = create_biased_data()
    
    # Detect bias
    bias_detector = BiasDetector(rai)
    bias_report = bias_detector.detect_data_bias(X_train, y_train, sensitive_train)
    
    if bias_report['bias_detected']:
        print(f"⚠️  Phát hiện {len(bias_report['biases'])} loại bias!")
        for bias in bias_report['biases']:
            print(f"   • {bias['type']} (severity: {bias['severity']})")
        
        # Send alert
        alert_manager.alert_on_bias(bias_report)
    
    # ===== 3. MITIGATION =====
    print_section("3️⃣  AUTOMATIC BIAS MITIGATION")
    
    mitigation_engine = MitigationEngine(rai)
    X_train_mitigated, y_train_mitigated, mitigation_report = mitigation_engine.analyze_and_mitigate_bias(
        X_train, y_train, sensitive_train
    )
    
    print(f"✓ Mitigation applied!")
    print(f"  Techniques used: {', '.join(mitigation_report['techniques_applied'])}")
    print(f"  Original distribution: {mitigation_report['original_distribution']}")
    print(f"  Final distribution: {mitigation_report['final_distribution']}")
    
    # ===== 4. MODEL TRAINING =====
    print_section("4️⃣  MODEL TRAINING với Responsible Wrapper")
    
    # Use adapter
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    model_adapter = SklearnAdapter(base_model)
    
    # Wrap với ResponsibleModelWrapper
    responsible_model = ResponsibleModelWrapper(model_adapter.model, rai)
    
    # Tính sample weights cho fairness
    sample_weights = mitigation_engine.compute_sample_weights(y_train_mitigated, sensitive_train)
    
    # Train
    responsible_model.fit(
        X_train_mitigated, 
        y_train_mitigated,
        sensitive_features=sensitive_train,
        sample_weight=sample_weights
    )
    
    print("✓ Model trained with bias mitigation!")
    
    # Log training
    audit_logger.log_training(
        model_type='RandomForestClassifier',
        dataset_info={
            'n_samples': len(X_train_mitigated),
            'n_features': X_train.shape[1],
            'mitigation_applied': True
        },
        hyperparameters={'n_estimators': 100},
        metrics={'sample_weights_used': True}
    )
    
    # ===== 5. FAIRNESS EVALUATION =====
    print_section("5️⃣  FAIRNESS EVALUATION")
    
    y_pred = responsible_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    fairness_results = responsible_model.evaluate_fairness(
        X_test, y_test, y_pred, sensitive_test
    )
    
    print("\nFairness Metrics:")
    for metric, value in fairness_results.items():
        if metric != 'overall_fairness_score':
            icon = "✓" if value >= 0.8 else "⚠️"
            print(f"  {icon} {metric}: {value:.4f}")
    
    # Alert if violations
    alert_manager.alert_on_fairness_violation(fairness_results)
    
    # ===== 6. ADVERSARIAL ROBUSTNESS =====
    print_section("6️⃣  ADVERSARIAL ROBUSTNESS TESTING")
    
    attack_generator = AttackGenerator(responsible_model.model)
    
    # Test FGSM attack
    print("Testing FGSM Attack...")
    X_adv_fgsm, fgsm_report = attack_generator.fgsm_attack(X_test[:100], y_test[:100], epsilon=0.3)
    print(f"  FGSM Success Rate: {fgsm_report['success_rate']:.1%}")
    
    # Test PGD attack
    print("Testing PGD Attack...")
    X_adv_pgd, pgd_report = attack_generator.pgd_attack(X_test[:100], y_test[:100], epsilon=0.3)
    print(f"  PGD Success Rate: {pgd_report['success_rate']:.1%}")
    
    # Comprehensive robustness evaluation
    robustness_report = attack_generator.evaluate_robustness(X_test[:200], y_test[:200])
    print(f"\n✓ Overall Robustness Score: {robustness_report.get('overall_robustness_score', 0):.4f}")
    
    # Alert if vulnerable
    alert_manager.alert_on_adversarial_attack({
        'success_rate': fgsm_report['success_rate'],
        'attack_type': 'FGSM'
    })
    
    # ===== 7. ADVERSARIAL TRAINING =====
    print_section("7️⃣  ADVERSARIAL TRAINING (Defense)")
    
    defense_trainer = DefenseTrainer(responsible_model.model, attack_generator)
    
    # Train với adversarial examples
    print("Applying adversarial training...")
    defense_report = defense_trainer.adversarial_training(
        X_train[:200], y_train[:200],
        attack_config={'attack_type': 'fgsm', 'epsilon': 0.3},
        ratio=0.5,
        epochs=2
    )
    
    # Evaluate defense
    defense_effectiveness = defense_trainer.evaluate_defense_effectiveness(
        X_test[:100], y_test[:100]
    )
    print(f"\n✓ Defense Effectiveness: {defense_effectiveness['defense_effectiveness']:.1%}")
    
    # ===== 8. DRIFT DETECTION =====
    print_section("8️⃣  MODEL DRIFT DETECTION")
    
    drift_detector = DriftDetector(threshold=0.05)
    drift_detector.set_reference(X_train, y_train, predictions=responsible_model.predict(X_train))
    
    # Simulate new data với drift
    X_new = X_test + np.random.normal(0, 0.1, X_test.shape)  # Add noise
    
    # Detect drift
    drift_results = drift_detector.detect_all_drifts(
        X_new[:100],
        y_test[:100],
        predictions_new=responsible_model.predict(X_new[:100]),
        model=responsible_model.model
    )
    
    if drift_results['overall_drift_detected']:
        print("⚠️  Drift detected!")
        if drift_results['data_drift']:
            print(f"  • Data drift: {len(drift_results['data_drift'].get('features_with_drift', []))} features")
        if drift_results['prediction_drift']:
            print(f"  • Prediction drift: p-value = {drift_results['prediction_drift'].get('p_value', 0):.4f}")
        
        # Alert
        alert_manager.alert_on_drift(drift_results)
    else:
        print("✓ No significant drift detected")
    
    # ===== 9. REPORTS =====
    print_section("9️⃣  COMPREHENSIVE REPORTS")
    
    # Model responsibility report
    print(responsible_model.generate_responsibility_report(X_test, y_test, sensitive_test))
    
    # Mitigation summary
    print("\n" + mitigation_engine.get_mitigation_summary())
    
    # Alert summary
    print("\n" + alert_manager.get_alert_summary())
    
    # Robustness report
    print("\n" + attack_generator.generate_robustness_report(X_test[:200], y_test[:200]))
    
    # ===== 10. SUMMARY =====
    print_section("🎉  ADVANCED DEMO COMPLETED!")
    
    print("✅ Features demonstrated:")
    print("  ✓ Automatic bias detection and mitigation")
    print("  ✓ Sample weighting for fairness")
    print("  ✓ Adversarial robustness testing (FGSM, PGD)")
    print("  ✓ Adversarial training for defense")
    print("  ✓ Model drift detection")
    print("  ✓ Multi-handler audit logging")
    print("  ✓ Intelligent alert management")
    print("  ✓ Framework adapters (sklearn)")
    print("  ✓ Comprehensive reporting")
    
    print("\n💡 Next steps:")
    print("  • Integrate with production ML pipeline")
    print("  • Enable PostgreSQL/Elasticsearch handlers")
    print("  • Configure email/Slack alerts")
    print("  • Try PyTorch/TensorFlow adapters")
    print("  • Implement continuous monitoring")
    
    print("\n" + "🚀" * 40 + "\n")


if __name__ == '__main__':
    main()

