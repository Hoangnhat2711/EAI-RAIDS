# Framework AI Có Trách Nhiệm (Responsible AI Framework)

## Giới thiệu

Framework này cung cấp một bộ công cụ toàn diện để phát triển và triển khai các hệ thống AI có trách nhiệm, đảm bảo tính công bằng, minh bạch, bảo mật và có thể giải thích được.

## Các Nguyên Tắc Cốt Lõi

1. **Công bằng (Fairness)**: Đảm bảo AI không phân biệt đối xử
2. **Minh bạch (Transparency)**: Quy trình ra quyết định rõ ràng
3. **Bảo mật (Privacy)**: Bảo vệ dữ liệu cá nhân
4. **Trách nhiệm (Accountability)**: Theo dõi và ghi nhận mọi quyết định
5. **Giải thích được (Explainability)**: Hiểu được cách AI đưa ra quyết định
6. **Vững chắc (Robustness)**: Hoạt động ổn định trong nhiều tình huống

## Cấu trúc

```
EAI Raid Ver2/
├── core/                    # Core framework
│   ├── responsible_ai.py    # Lớp chính của framework
│   ├── model_wrapper.py     # Wrapper cho các models
│   └── validator.py         # Xác thực compliance
├── fairness/                # Công cụ đánh giá công bằng
│   ├── metrics.py          # Các metrics về fairness
│   └── bias_detector.py    # Phát hiện bias
├── explainability/         # Công cụ giải thích
│   ├── shap_explainer.py   # SHAP explanations
│   └── lime_explainer.py   # LIME explanations
├── privacy/                # Bảo vệ privacy
│   ├── differential_privacy.py
│   └── anonymization.py
├── audit/                  # Hệ thống audit
│   ├── logger.py          # Audit logging
│   └── reporter.py        # Report generation
├── monitoring/             # Giám sát model
│   └── drift_detector.py  # Phát hiện model drift
└── examples/              # Ví dụ sử dụng
    └── demo.py

```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng nhanh

```python
from core.responsible_ai import ResponsibleAI
from core.model_wrapper import ResponsibleModelWrapper
from sklearn.ensemble import RandomForestClassifier

# Khởi tạo framework
rai = ResponsibleAI(config_path='config.yaml')

# Wrap model của bạn
model = RandomForestClassifier()
responsible_model = ResponsibleModelWrapper(model, rai)

# Train với các kiểm tra trách nhiệm
responsible_model.fit(X_train, y_train, sensitive_features=sensitive_train)

# Predict với audit logging
predictions = responsible_model.predict(X_test)

# Tạo báo cáo đầy đủ
report = responsible_model.generate_responsibility_report(X_test, y_test)
```

## Tính năng chính

- ✅ Đánh giá và giám sát fairness tự động
- ✅ Giải thích predictions với SHAP và LIME
- ✅ Differential Privacy cho training data
- ✅ Audit logging toàn diện
- ✅ Phát hiện model drift
- ✅ Báo cáo trách nhiệm tự động
- ✅ Bias detection và mitigation

## Giấy phép

MIT License

