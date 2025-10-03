"""
Utils module - Utilities and helpers
"""

from .statistical_testing import (
    SignificanceTest,
    MultipleComparisonCorrection,
    BootstrapCI,
    NormalityTest,
    ModelComparison
)

__all__ = [
    'SignificanceTest',
    'MultipleComparisonCorrection',
    'BootstrapCI',
    'NormalityTest',
    'ModelComparison'
]

