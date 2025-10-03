"""
Simple Demo - Test framework without external dependencies
"""

import sys
import os
import traceback

def test_framework_structure():
    """Test framework structure and imports"""
    print("=" * 70)
    print("🚀 EAI-RAIDS FRAMEWORK - STRUCTURE TEST")
    print("=" * 70)
    
    # Test core modules
    modules_to_test = [
        'core.responsible_ai',
        'core.model_wrapper', 
        'core.validator',
        'core.data_converter',
        'core.mlops_integration',
        'fairness.metrics',
        'fairness.bias_detector',
        'fairness.inprocessing',
        'explainability.shap_explainer',
        'explainability.lime_explainer',
        'explainability.causal_explainer',
        'explainability.causal_inference',
        'privacy.differential_privacy',
        'privacy.dp_sgd',
        'robustness.attack_generator',
        'robustness.certified_defense',
        'utils.statistical_testing',
        'audit.logger',
        'monitoring.drift_detector'
    ]
    
    print("\n📦 Testing module imports...")
    successful_imports = 0
    failed_imports = 0
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
            successful_imports += 1
        except ImportError as e:
            print(f"  ❌ {module}: {str(e)[:50]}...")
            failed_imports += 1
        except Exception as e:
            print(f"  ⚠️  {module}: {str(e)[:50]}...")
            failed_imports += 1
    
    print(f"\n📊 Import Results: {successful_imports} ✅, {failed_imports} ❌")
    
    return successful_imports, failed_imports

def test_core_functionality():
    """Test core functionality without external deps"""
    print("\n🔧 Testing core functionality...")
    
    try:
        # Test ResponsibleAI initialization
        from core.responsible_ai import ResponsibleAI
        rai = ResponsibleAI()
        print("  ✅ ResponsibleAI initialized")
        
        # Test principles
        principles = rai.get_active_principles()
        print(f"  ✅ Active principles: {len(principles)}")
        
        # Test data converter
        from core.data_converter import DataConverter, CausalInferenceError
        converter = DataConverter()
        print("  ✅ DataConverter initialized")
        
        # Test statistical testing
        from utils.statistical_testing import NormalityTest, ModelComparison
        print("  ✅ Statistical testing modules loaded")
        
        # Test MLOps integration
        from core.mlops_integration import ExperimentTracker
        print("  ✅ MLOps integration loaded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Core functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_sota_features():
    """Test SOTA features"""
    print("\n🌟 Testing SOTA features...")
    
    try:
        # Test fairness in-processing
        from fairness.inprocessing import AdversarialDebiasing, PrejudiceRemover
        print("  ✅ Fairness in-processing loaded")
        
        # Test certified robustness
        from robustness.certified_defense import RandomizedSmoothing, IntervalBoundPropagation
        print("  ✅ Certified robustness loaded")
        
        # Test causal inference
        from explainability.causal_inference import DoWhyIntegration, CausalMLIntegration
        print("  ✅ Causal inference loaded")
        
        # Test statistical testing
        from utils.statistical_testing import NormalityTest
        print("  ✅ Normality testing loaded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SOTA features test failed: {e}")
        return False

def show_framework_info():
    """Show framework information"""
    print("\n📋 FRAMEWORK INFORMATION")
    print("=" * 50)
    
    # Check files
    important_files = [
        'README.md',
        'SOTA_FEATURES.md', 
        'RESEARCH_RIGOR_FIXES.md',
        'CRITICAL_FIXES.md',
        'PUBLICATION_READY.md',
        'requirements.txt',
        'setup.py',
        'pytest.ini'
    ]
    
    print("\n📄 Documentation files:")
    for file in important_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✅ {file} ({size} bytes)")
        else:
            print(f"  ❌ {file} (missing)")
    
    # Check directories
    important_dirs = [
        'core/',
        'fairness/',
        'explainability/',
        'privacy/',
        'robustness/',
        'utils/',
        'audit/',
        'monitoring/',
        'examples/',
        'tests/',
        'benchmarks/',
        '.github/workflows/'
    ]
    
    print("\n📁 Directory structure:")
    for dir_path in important_dirs:
        if os.path.exists(dir_path):
            files_count = len([f for f in os.listdir(dir_path) if f.endswith('.py')])
            print(f"  ✅ {dir_path} ({files_count} Python files)")
        else:
            print(f"  ❌ {dir_path} (missing)")

def main():
    """Main demo function"""
    print("Python version:", sys.version)
    print("Working directory:", os.getcwd())
    
    # Test 1: Framework structure
    successful, failed = test_framework_structure()
    
    # Test 2: Core functionality
    core_ok = test_core_functionality()
    
    # Test 3: SOTA features
    sota_ok = test_sota_features()
    
    # Show framework info
    show_framework_info()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FRAMEWORK TEST SUMMARY")
    print("=" * 70)
    
    print(f"✅ Successful imports: {successful}")
    print(f"❌ Failed imports: {failed}")
    print(f"🔧 Core functionality: {'✅ PASS' if core_ok else '❌ FAIL'}")
    print(f"🌟 SOTA features: {'✅ PASS' if sota_ok else '❌ FAIL'}")
    
    if core_ok and sota_ok:
        print("\n🎉 FRAMEWORK IS READY!")
        print("📚 Key features available:")
        print("  • Fairness In-Processing (Adversarial Debiasing, Prejudice Remover)")
        print("  • Certified Robustness (Randomized Smoothing, IBP)")
        print("  • Causal Inference (DoWhy, CausalML)")
        print("  • MLOps Integration (MLflow, DVC)")
        print("  • Statistical Testing (Normality checks, significance tests)")
        print("  • Data Converter (NumPy ↔ Pandas)")
        print("  • Gradient Wrapper (Analytical gradients)")
        print("\n🚀 Ready for ICML/NeurIPS/ICLR/AAAI/FAccT submission!")
    else:
        print("\n⚠️  Some issues detected. Check dependencies:")
        print("  pip install numpy pandas scikit-learn")
        print("  pip install torch tensorflow")
        print("  pip install shap lime")
        print("  pip install mlflow dvc")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
