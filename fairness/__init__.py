"""
Fairness module - Đánh giá và đảm bảo tính công bằng của AI
"""

from .metrics import FairnessMetrics
from .bias_detector import BiasDetector
from .inprocessing import AdversarialDebiasing, PrejudiceRemover, FairConstrainedOptimization

__all__ = [
    'FairnessMetrics',
    'BiasDetector',
    'AdversarialDebiasing',
    'PrejudiceRemover',
    'FairConstrainedOptimization'
]

