# 🚀 EAI-RAIDS: Enterprise Responsible AI Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/Research-Advanced-green.svg)]()

> **Framework AI Có Trách Nhiệm Toàn Diện** - Production-Ready MLOps Solution với Tính năng Nghiên cứu Tiên tiến

Framework này cung cấp một hệ thống hoàn chỉnh để phát triển, triển khai và giám sát các hệ thống AI có trách nhiệm, đảm bảo **Fairness, Transparency, Privacy, Accountability, Explainability và Robustness**.

---

## ✨ Điểm Nổi Bật

### 🎯 **Tự Động Hóa MLOps**
- ✅ **Mitigation Engine**: Tự động phát hiện và xử lý bias/drift
- ✅ **Multi-Backend Audit Logging**: PostgreSQL, Elasticsearch, AWS CloudWatch, Azure Monitor, GCP
- ✅ **Intelligent Alert Manager**: Email, Slack, Webhook, SMS notifications
- ✅ **Auto-Retraining**: Phát hiện drift và khuyến nghị retrain

### 🔬 **Nghiên Cứu Tiên Tiến**
- ✅ **Adversarial Robustness**: FGSM, PGD, DeepFool attacks + Adversarial Training
- ✅ **Advanced Fairness**: Demographic Parity, Equalized Odds, Disparate Impact
- ✅ **Causal Explainability**: Không chỉ correlation mà còn causation
- ✅ **Differential Privacy**: DP-SGD cho Deep Learning

### 🔧 **Tích Hợp Linh Hoạt**
- ✅ **Framework Agnostic**: Sklearn, PyTorch, TensorFlow adapters
- ✅ **Production-Ready**: Enterprise logging, monitoring, alerting
- ✅ **Scalable**: Cloud-native architecture

---

## 📋 Mục Lục

1. [Cài Đặt](#cài-đặt)
2. [Quick Start](#quick-start)
3. [Kiến Trúc](#kiến-trúc)
4. [Tính Năng Chi Tiết](#tính-năng-chi-tiết)
5. [Advanced Usage](#advanced-usage)
6. [Examples](#examples)
7. [Research Papers](#research-papers)

---

## 🔧 Cài Đặt

### Basic Installation

```bash
git clone https://github.com/Hoangnhat2711/EAI-RAIDSs.git
cd EAI-RAIDSs
pip install -r requirements.txt
```

### Full Installation (với tất cả tính năng)

```bash
# Core + Deep Learning
pip install torch tensorflow

# Enterprise Logging
pip install psycopg2-binary elasticsearch boto3

# Advanced Features
pip install imbalanced-learn shap lime fairlearn
```

---

## 🚀 Quick Start

```python
from core.responsible_ai import ResponsibleAI
from core.model_wrapper import ResponsibleModelWrapper
from core.mitigation_engine import MitigationEngine
from sklearn.ensemble import RandomForestClassifier

# 1. Khởi tạo framework
rai = ResponsibleAI(config_path='config.yaml')

# 2. Tự động xử lý bias
mitigation_engine = MitigationEngine(rai)
X_clean, y_clean, report = mitigation_engine.analyze_and_mitigate_bias(
    X_train, y_train, sensitive_features
)

# 3. Train với responsible wrapper
model = RandomForestClassifier()
responsible_model = ResponsibleModelWrapper(model, rai)
responsible_model.fit(X_clean, y_clean, sensitive_features=sensitive_features)

# 4. Predict và đánh giá
predictions = responsible_model.predict(X_test)
fairness_report = responsible_model.evaluate_fairness(
    X_test, y_test, predictions, sensitive_features
)

# 5. Generate report
print(responsible_model.generate_responsibility_report(X_test, y_test))
```

---

## 🏗️ Kiến Trúc

```
EAI-RAIDS/
├── core/                           # Core framework
│   ├── responsible_ai.py          # Main framework class
│   ├── model_wrapper.py           # Responsible model wrapper
│   ├── validator.py               # Compliance validator
│   ├── mitigation_engine.py       # 🆕 Auto bias/drift mitigation
│   ├── alert_manager.py           # 🆕 Intelligent alerting
│   └── adapters/                  # 🆕 Framework adapters
│       ├── sklearn_adapter.py
│       ├── pytorch_adapter.py
│       └── tensorflow_adapter.py
│
├── fairness/                      # Fairness & Bias
│   ├── metrics.py                 # Fairness metrics
│   └── bias_detector.py           # Bias detection
│
├── explainability/                # Explainability
│   ├── shap_explainer.py          # SHAP explanations
│   ├── lime_explainer.py          # LIME explanations
│   └── causal_explainer.py        # 🆕 Causal analysis
│
├── privacy/                       # Privacy Protection
│   ├── differential_privacy.py    # Differential privacy
│   ├── anonymization.py           # Data anonymization
│   └── dp_sgd.py                  # 🆕 DP-SGD for DL
│
├── robustness/                    # 🆕 Adversarial Robustness
│   ├── attack_generator.py        # FGSM, PGD, DeepFool
│   └── defense_trainer.py         # Adversarial training
│
├── audit/                         # Audit & Logging
│   ├── logger.py                  # Audit logger
│   ├── reporter.py                # Report generator
│   └── handlers/                  # 🆕 Multi-backend handlers
│       ├── file_handler.py
│       ├── postgres_handler.py
│       ├── elasticsearch_handler.py
│       └── cloud_handler.py
│
├── monitoring/                    # Monitoring
│   └── drift_detector.py          # Drift detection
│
└── examples/                      # Examples
    ├── demo.py                    # Basic demo
    └── advanced_demo.py           # 🆕 Advanced features
```

---

## 🎯 Tính Năng Chi Tiết

### 1. **Mitigation Engine** 🔧

Tự động phát hiện và xử lý các vấn đề:

```python
from core.mitigation_engine import MitigationEngine

engine = MitigationEngine(rai)

# Tự động xử lý class imbalance
X_balanced, y_balanced, report = engine.analyze_and_mitigate_bias(
    X, y, sensitive_features
)

# Tính sample weights
weights = engine.compute_sample_weights(y, sensitive_features)

# Post-process predictions cho fairness
predictions_fair = engine.postprocess_predictions(
    predictions, sensitive_features, 
    fairness_constraint='demographic_parity'
)
```

**Techniques:**
- Class Reweighting
- SMOTE Oversampling
- Representation Balancing
- Fairness-constrained Post-processing

---

### 2. **Adversarial Robustness** 🛡️

Test và defense against adversarial attacks:

```python
from robustness.attack_generator import AttackGenerator
from robustness.defense_trainer import DefenseTrainer

# Generate attacks
attack_gen = AttackGenerator(model)
X_adv, report = attack_gen.fgsm_attack(X, y, epsilon=0.3)
X_adv, report = attack_gen.pgd_attack(X, y, epsilon=0.3, num_iter=40)

# Comprehensive robustness evaluation
robustness = attack_gen.evaluate_robustness(X_test, y_test)

# Adversarial training
defense = DefenseTrainer(model, attack_gen)
defense.adversarial_training(
    X_train, y_train,
    attack_config={'attack_type': 'pgd', 'epsilon': 0.3},
    ratio=0.5
)
```

**Attacks Supported:**
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- DeepFool
- C&W (Carlini & Wagner)

---

### 3. **Multi-Backend Audit Logging** 📝

Enterprise-grade logging với multiple backends:

```python
from audit.logger import AuditLogger
from audit.handlers import PostgreSQLHandler, ElasticsearchHandler

# Multiple handlers
handlers = [
    FileLogHandler({'log_dir': 'logs'}),
    PostgreSQLHandler({
        'host': 'localhost',
        'database': 'audit_db',
        'user': 'admin'
    }),
    ElasticsearchHandler({
        'hosts': ['http://localhost:9200'],
        'index_name': 'ai-audit'
    })
]

logger = AuditLogger(handlers=handlers)

# Logs automatically written to all backends
logger.log_prediction(X, predictions, model_info={'version': '1.0'})
```

**Supported Backends:**
- File (JSONL)
- PostgreSQL
- Elasticsearch
- AWS CloudWatch
- Azure Monitor
- GCP Cloud Logging

---

### 4. **Alert Manager** 🚨

Intelligent alerting system:

```python
from core.alert_manager import AlertManager, AlertLevel

alert_manager = AlertManager({
    'channels': {
        'console': {'enabled': True},
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'recipient_emails': ['admin@company.com']
        },
        'slack': {
            'enabled': True,
            'webhook_url': 'https://hooks.slack.com/...'
        }
    }
})

# Auto alerts on issues
alert_manager.alert_on_drift(drift_results)
alert_manager.alert_on_fairness_violation(fairness_results)
alert_manager.alert_on_bias(bias_report)
alert_manager.alert_on_adversarial_attack(attack_results)
```

**Channels:**
- Console
- Email (SMTP)
- Slack
- Webhook
- SMS (Twilio)

---

### 5. **Framework Adapters** 🔌

Support cho multiple ML frameworks:

```python
from core.adapters import SklearnAdapter, PyTorchAdapter, TensorFlowAdapter

# Sklearn
sklearn_adapter = SklearnAdapter(RandomForestClassifier())

# PyTorch
pytorch_model = MyNeuralNetwork()
pytorch_adapter = PyTorchAdapter(
    pytorch_model, 
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(pytorch_model.parameters())
)

# TensorFlow
tf_model = tf.keras.Sequential([...])
tf_adapter = TensorFlowAdapter(tf_model)

# Use như nhau
responsible_model = ResponsibleModelWrapper(adapter.model, rai)
```

---

### 6. **Drift Detection** 📊

Comprehensive drift monitoring:

```python
from monitoring.drift_detector import DriftDetector

detector = DriftDetector(threshold=0.05)
detector.set_reference(X_train, y_train, predictions=train_preds)

# Detect all types of drift
drift_results = detector.detect_all_drifts(
    X_new, y_new, predictions_new, model
)

# PSI (Population Stability Index)
psi_scores = detector.monitor_features_psi(X_new)
```

**Drift Types:**
- Data Drift (Kolmogorov-Smirnov test)
- Prediction Drift
- Concept Drift
- PSI monitoring

---

## 📚 Advanced Usage

### Complete Pipeline Example

```python
# 1. Setup
rai = ResponsibleAI()
alert_manager = AlertManager(config)
audit_logger = AuditLogger(handlers=[...])
mitigation_engine = MitigationEngine(rai)

# 2. Data preparation & mitigation
X_clean, y_clean, _ = mitigation_engine.analyze_and_mitigate_bias(
    X_train, y_train, sensitive_features
)

# 3. Model training
model = RandomForestClassifier()
responsible_model = ResponsibleModelWrapper(model, rai)
responsible_model.fit(X_clean, y_clean)

# 4. Robustness testing
attack_gen = AttackGenerator(model)
robustness = attack_gen.evaluate_robustness(X_test, y_test)
alert_manager.alert_on_adversarial_attack(robustness)

# 5. Adversarial training if needed
if robustness['overall_robustness_score'] < 0.7:
    defense = DefenseTrainer(model, attack_gen)
    defense.adversarial_training(X_train, y_train)

# 6. Deployment monitoring
drift_detector = DriftDetector()
drift_detector.set_reference(X_train, predictions=model.predict(X_train))

# In production
drift = drift_detector.detect_all_drifts(X_production, ...)
if drift['overall_drift_detected']:
    alert_manager.alert_on_drift(drift)
    # Trigger retraining pipeline
```

---

## 🎬 Examples

### Basic Demo
```bash
python examples/demo.py
```

### Advanced Demo (All Features)
```bash
python examples/advanced_demo.py
```

---

## 📊 Research Papers & References

Framework này triển khai các kỹ thuật từ:

1. **Fairness:**
   - Hardt et al. "Equality of Opportunity in Supervised Learning" (2016)
   - Feldman et al. "Certifying and Removing Disparate Impact" (2015)

2. **Adversarial Robustness:**
   - Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (2015) - FGSM
   - Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018) - PGD

3. **Privacy:**
   - Dwork et al. "Differential Privacy" (2006)
   - Abadi et al. "Deep Learning with Differential Privacy" (2016) - DP-SGD

4. **Explainability:**
   - Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (2017) - SHAP
   - Ribeiro et al. "Why Should I Trust You?" (2016) - LIME

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Causal explainability module
- DP-SGD implementation
- More attack methods
- Additional fairness metrics

---

## 📄 License

MIT License - see LICENSE file

---

## 👥 Authors

**EAI-RAIDS Team**
- Advanced Responsible AI Research

---

## 🌟 Citation

If you use this framework in your research, please cite:

```bibtex
@software{eai_raids_2025,
  title={EAI-RAIDS: Enterprise Responsible AI Framework},
  author={EAI-RAIDS Team},
  year={2025},
  url={https://github.com/Hoangnhat2711/EAI-RAIDSs}
}
```

---

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/Hoangnhat2711/EAI-RAIDSs/issues)
- 📧 Email: support@eai-raids.com
- 📖 Docs: [Documentation](https://eai-raids.readthedocs.io)

---

**⭐ If you find this useful, please star the repository!**
