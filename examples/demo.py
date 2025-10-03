"""
Demo sử dụng Responsible AI Framework

Ví dụ này minh họa cách sử dụng framework để:
1. Train một model với các kiểm tra trách nhiệm
2. Đánh giá fairness
3. Giải thích predictions
4. Audit logging
5. Kiểm tra compliance
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import Responsible AI Framework
from core.responsible_ai import ResponsibleAI
from core.model_wrapper import ResponsibleModelWrapper
from core.validator import ComplianceValidator
from fairness.metrics import FairnessMetrics
from fairness.bias_detector import BiasDetector
from audit.logger import AuditLogger
from audit.reporter import AuditReporter


def create_sample_data():
    """Tạo sample data cho demo"""
    print("\n📊 Tạo sample dataset...")
    
    # Tạo synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    # Tạo sensitive feature (giả lập gender: 0 hoặc 1)
    sensitive_features = np.random.randint(0, 2, size=1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    sensitive_train = sensitive_features[:800]
    sensitive_test = sensitive_features[800:]
    
    print(f"✓ Dataset tạo xong: {len(X_train)} training samples, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test


def main():
    """Main demo function"""
    
    print("=" * 70)
    print("🚀 DEMO: RESPONSIBLE AI FRAMEWORK")
    print("=" * 70)
    
    # 1. Khởi tạo Responsible AI Framework
    print("\n📋 Bước 1: Khởi tạo Responsible AI Framework")
    rai = ResponsibleAI(config_path='config.yaml')
    print(f"✓ Framework khởi tạo với các nguyên tắc: {', '.join(rai.get_active_principles())}")
    
    # 2. Tạo data
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = create_sample_data()
    
    # 3. Khởi tạo Audit Logger
    print("\n📝 Bước 2: Khởi tạo Audit Logger")
    audit_logger = AuditLogger(log_dir='audit_logs', responsible_ai_framework=rai)
    print(f"✓ Audit Logger sẵn sàng (Session: {audit_logger.session_id})")
    
    # 4. Kiểm tra bias trong data
    print("\n🔍 Bước 3: Kiểm tra bias trong training data")
    bias_detector = BiasDetector(rai)
    data_bias_report = bias_detector.detect_data_bias(
        X_train, y_train, sensitive_train
    )
    
    if data_bias_report['bias_detected']:
        print(f"⚠ Phát hiện {len(data_bias_report['biases'])} loại bias trong data!")
        for bias in data_bias_report['biases']:
            print(f"  - {bias['type']} (Severity: {bias['severity']})")
    else:
        print("✓ Không phát hiện bias nghiêm trọng trong training data")
    
    # Log bias detection
    audit_logger.log_bias_detection(
        bias_type='data_bias',
        detected=data_bias_report['bias_detected'],
        severity='medium' if data_bias_report['bias_detected'] else 'low',
        details=data_bias_report
    )
    
    # 5. Train model với Responsible AI wrapper
    print("\n🤖 Bước 4: Training model với Responsible AI wrapper")
    
    # Tạo base model
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Wrap với ResponsibleModelWrapper
    responsible_model = ResponsibleModelWrapper(base_model, rai)
    
    # Train
    responsible_model.fit(X_train, y_train, sensitive_features=sensitive_train)
    print("✓ Model training hoàn tất!")
    
    # Log training
    audit_logger.log_training(
        model_type='RandomForestClassifier',
        dataset_info={
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        },
        hyperparameters={'n_estimators': 100},
        metrics={'trained': True}
    )
    
    # 6. Predictions
    print("\n🎯 Bước 5: Making predictions")
    y_pred = responsible_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    
    # Log predictions
    audit_logger.log_prediction(
        input_data=X_test,
        prediction=y_pred,
        model_info={'type': 'RandomForestClassifier'},
        confidence=accuracy
    )
    
    # 7. Đánh giá Fairness
    print("\n⚖️ Bước 6: Đánh giá Fairness")
    fairness_results = responsible_model.evaluate_fairness(
        X_test, y_test, y_pred, sensitive_test
    )
    
    print("Fairness Metrics:")
    for metric, value in fairness_results.items():
        icon = "✓" if value >= 0.8 else "⚠"
        print(f"  {icon} {metric}: {value:.4f}")
    
    # Log fairness check
    fairness_passed = all(v >= 0.8 for v in fairness_results.values())
    audit_logger.log_fairness_check(
        metrics=fairness_results,
        passed=fairness_passed,
        threshold=0.8
    )
    
    # 8. Kiểm tra prediction bias
    print("\n🔍 Bước 7: Kiểm tra bias trong predictions")
    prediction_bias = bias_detector.detect_prediction_bias(
        y_test, y_pred, sensitive_test
    )
    
    if prediction_bias['bias_detected']:
        print(f"⚠ Phát hiện bias trong predictions!")
        for bias in prediction_bias['biases']:
            print(f"  - {bias['type']}")
    else:
        print("✓ Không phát hiện bias trong predictions")
    
    # 9. Giải thích predictions (simplified - không require SHAP/LIME install)
    print("\n💡 Bước 8: Model Explainability")
    print("✓ Framework hỗ trợ SHAP và LIME explanations")
    print("  (Cần cài đặt: pip install shap lime)")
    
    # 10. Compliance Validation
    print("\n✅ Bước 9: Kiểm tra Compliance")
    validator = ComplianceValidator(rai)
    compliance_results = validator.validate_all(
        responsible_model, X_test, y_test, sensitive_test
    )
    
    overall_status = "✓ ĐẠT" if compliance_results['overall_compliance'] else "✗ KHÔNG ĐẠT"
    print(f"Trạng thái tổng thể: {overall_status}")
    
    print("\nChi tiết:")
    for principle, result in compliance_results['checks'].items():
        status = "✓" if result['passed'] else "✗"
        print(f"  {status} {principle.capitalize()}: Score {result['score']:.2f}")
    
    # Đề xuất cải thiện
    recommendations = validator.get_recommendations()
    if recommendations:
        print("\n📌 Đề xuất cải thiện:")
        for rec in recommendations[:3]:
            print(f"  • {rec}")
    
    # 11. Tạo Audit Reports
    print("\n📊 Bước 10: Tạo Audit Reports")
    reporter = AuditReporter(audit_logger)
    
    # Compliance report
    print("\n" + reporter.generate_compliance_report())
    
    # Activity report
    print("\n" + reporter.generate_activity_report())
    
    # 12. Tạo Model Responsibility Report
    print("\n📄 Bước 11: Model Responsibility Report")
    print(responsible_model.generate_responsibility_report(
        X_test, y_test, sensitive_test
    ))
    
    # 13. Framework Summary
    print("\n📋 Framework Summary")
    print(rai.generate_summary_report())
    
    # Export reports
    print("\n💾 Xuất báo cáo...")
    reporter.export_report('full', 'audit_report_full.txt')
    print("✓ Báo cáo đã được lưu: audit_report_full.txt")
    
    print("\n" + "=" * 70)
    print("✨ DEMO HOÀN TẤT!")
    print("=" * 70)
    print("\n📚 Các tính năng đã demo:")
    print("  ✓ Khởi tạo Responsible AI Framework")
    print("  ✓ Bias detection trong data và predictions")
    print("  ✓ Training với responsible checks")
    print("  ✓ Fairness evaluation")
    print("  ✓ Compliance validation")
    print("  ✓ Audit logging và reporting")
    print("  ✓ Model explainability (framework support)")
    print("\n🎉 Framework sẵn sàng để sử dụng trong production!")
    print("=" * 70)


if __name__ == '__main__':
    main()

