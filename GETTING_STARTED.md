# 🚀 Getting Started with EAI-RAIDS

**Quick guide to get you started in 5 minutes!**

---

## 📋 Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **pip** package manager
- **Git** (for cloning repository)

---

## ⚡ 5-Minute Quick Start

### Step 1: Clone Repository

```bash
git clone https://github.com/Hoangnhat2711/EAI-RAIDS
cd EAI-RAIDS
```

### Step 2: Install Dependencies

**Option A: Automatic (Recommended)**
```bash
chmod +x install.sh
./install.sh
# Choose option 2 (Standard) when prompted
```

**Option B: Manual**
```bash
pip install -r requirements.txt
```

### Step 3: Run Demo

```bash
# Basic demo (no external dependencies needed)
python3 basic_demo.py

# Full demo (requires dependencies)
python3 examples/demo.py
```

**🎉 That's it! You're ready to go!**

---

## 🎯 Your First Responsible AI Model

Create a file `my_first_model.py`:

```python
from core import ResponsibleAI, ResponsibleModelWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Create sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
sensitive = np.random.randint(0, 2, size=1000)  # Simulated sensitive feature
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sens_train, sens_test = sensitive[:800], sensitive[800:]

# 2. Initialize Responsible AI framework
rai = ResponsibleAI()

# 3. Create and wrap your model
model = ResponsibleModelWrapper(RandomForestClassifier(), rai)

# 4. Train with automatic fairness checks
model.fit(X_train, y_train, sensitive_features=sens_train)

# 5. Predict with monitoring
predictions = model.predict(X_test)

# 6. Get fairness evaluation
fairness = model.evaluate_fairness(X_test, y_test, predictions, sens_test)
print("Fairness Metrics:", fairness)

# 7. Generate full report
print(model.generate_responsibility_report(X_test, y_test, sens_test))
```

Run it:
```bash
python3 my_first_model.py
```

---

## 📚 Learning Path

### Beginner (Week 1)
1. ✅ Run `basic_demo.py` - Understand framework structure
2. ✅ Read `QUICK_REFERENCE.md` - Learn common APIs
3. ✅ Try your first model (above example)
4. ✅ Explore `examples/demo.py` - See all features

### Intermediate (Week 2-3)
1. 🔧 Learn **Fairness In-Processing**
   ```python
   from fairness import AdversarialDebiasing
   model = AdversarialDebiasing(sensitive_attribute_idx=0)
   model.fit(X_train, y_train)
   ```

2. 🛡️ Learn **Certified Robustness**
   ```python
   from robustness import RandomizedSmoothing
   smoother = RandomizedSmoothing(model, sigma=0.25)
   results = smoother.certify(X_test, y_test, epsilon=0.5)
   ```

3. 🔒 Learn **Privacy-Preserving Training**
   ```python
   from privacy import OpacusIntegration
   opacus = OpacusIntegration(epsilon=1.0, delta=1e-5)
   private_model = opacus.make_private(model, optimizer, loader)
   ```

### Advanced (Week 4+)
1. 🧠 **Causal Inference**
   ```python
   from explainability import DoWhyIntegration
   dowhy = DoWhyIntegration('treatment', 'outcome', confounders)
   causal_effect = dowhy.complete_analysis(data)
   ```

2. 📊 **Statistical Testing**
   ```python
   from utils import ModelComparison
   comparator = ModelComparison(check_assumptions=True)
   results = comparator.compare_two_models(baseline, improved)
   ```

3. 🔧 **MLOps Integration**
   ```python
   from core import ExperimentTracker
   tracker = ExperimentTracker("my-experiment")
   tracker.start_run()
   # ... train and log ...
   tracker.end_run()
   ```

---

## 🎓 Common Use Cases

### Use Case 1: Fair Hiring Model

```python
from fairness import AdversarialDebiasing

# Sensitive: gender (column 0)
model = AdversarialDebiasing(
    sensitive_attribute_idx=0,
    adversary_loss_weight=1.0
)

model.fit(applicant_features, hiring_decisions)
predictions = model.predict(new_applicants)

# Model is trained to be fair across genders!
```

### Use Case 2: Robust Medical Diagnosis

```python
from robustness import RandomizedSmoothing, AttackGenerator

# Test robustness
attack_gen = AttackGenerator(medical_model)
robustness = attack_gen.evaluate_robustness(X_test, y_test)

# Certify robustness
smoother = RandomizedSmoothing(medical_model, sigma=0.25)
cert_results = smoother.certify(X_test, y_test, epsilon=0.5)

print(f"Certified accuracy: {cert_results['certified_accuracy']:.2%}")
```

### Use Case 3: Private Customer Analytics

```python
from privacy import OpacusIntegration

# Train with differential privacy
opacus = OpacusIntegration(epsilon=1.0, delta=1e-5)
model, optimizer, engine = opacus.make_private(
    model, optimizer, data_loader, epochs=10
)

# Privacy guarantee: (ε=1.0, δ=10⁻⁵)-DP
```

### Use Case 4: Explainable Loan Decisions

```python
from explainability import CounterfactualExplainer

explainer = CounterfactualExplainer(loan_model)
cf = explainer.explain(rejected_application, desired_class=1)

print("To be approved, you need:")
for feature, change in cf['changes'].items():
    print(f"  - {feature}: {change}")
```

---

## 🆘 Troubleshooting

### Issue: Import errors

**Problem:** `ImportError: cannot import name 'X'`

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or use installation script
./install.sh
```

### Issue: Slow performance

**Problem:** Counterfactuals or gradients are slow

**Solution:**
```python
# Use analytical gradients (10-3000x faster!)
from core.adapters import GradientWrapper

grad_wrapper = GradientWrapper(model)
print(f"Using analytical: {grad_wrapper.is_analytical()}")

# For counterfactuals
explainer = CounterfactualExplainer(
    model,
    use_analytical_gradients=True  # Enable fast mode!
)
```

### Issue: Causal inference errors

**Problem:** `CausalInferenceError: causal-learn not installed`

**Solution:**
```bash
# Install causal inference libraries
pip install dowhy causalml causal-learn lingam
```

### Issue: Memory errors with DP-SGD

**Problem:** Out of memory during private training

**Solution:**
```python
# Reduce batch size
opacus = OpacusIntegration(epsilon=1.0, delta=1e-5)
model, optimizer, engine = opacus.make_private(
    model,
    optimizer,
    data_loader,
    epochs=10,
    max_physical_batch_size=32  # Reduce this!
)
```

---

## 📖 Next Steps

### Documentation
- `README.md` - Comprehensive user guide
- `QUICK_REFERENCE.md` - API quick reference
- `SOTA_FEATURES.md` - Deep dive into SOTA features
- `EXECUTIVE_SUMMARY.md` - High-level overview

### Examples
- `examples/demo.py` - Complete demo
- `examples/advanced_demo.py` - Advanced features
- `basic_demo.py` - System check

### Research
- `RESEARCH_RIGOR_FIXES.md` - Research integrity details
- `CITATION.bib` - How to cite
- `PUBLICATION_READY.md` - Publication status

---

## 💬 Get Help

### Community Support
- **GitHub Issues:** https://github.com/Hoangnhat2711/EAI-RAIDS/issues
- **Discussions:** https://github.com/Hoangnhat2711/EAI-RAIDS/discussions

### Documentation
- **Quick Reference:** `QUICK_REFERENCE.md`
- **Full Documentation:** All `*.md` files
- **Code Examples:** `examples/` directory

### Contact
- **Research:** research@eai-raids.com
- **Enterprise:** enterprise@eai-raids.com

---

## ✨ Pro Tips

1. **Start simple** - Begin with `ResponsibleModelWrapper`
2. **Check assumptions** - Use `ModelComparison` with auto-checks
3. **Track experiments** - Use `ExperimentTracker` from day 1
4. **Read examples** - `examples/` has working code
5. **Use analytical gradients** - 10-3000x faster!

---

## 🎯 What's Next?

Ready to dive deeper? Check out:

- 📚 **SOTA Features:** `SOTA_FEATURES.md`
- 🔬 **Research Guide:** `RESEARCH_RIGOR_FIXES.md`
- ⚡ **Quick Reference:** `QUICK_REFERENCE.md`
- 🏆 **Executive Summary:** `EXECUTIVE_SUMMARY.md`

**Happy coding! 🚀**

---

**Version:** 3.1.0 | **License:** MIT | **Repository:** https://github.com/Hoangnhat2711/EAI-RAIDS

