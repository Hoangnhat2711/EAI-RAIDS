"""
Scikit-learn Adapter
"""

import numpy as np
from typing import Any, Optional
import joblib
from .base_adapter import BaseModelAdapter


class SklearnAdapter(BaseModelAdapter):
    """
    Adapter cho scikit-learn models
    """
    
    def _detect_framework(self) -> str:
        """Detect sklearn"""
        return "scikit-learn"
    
    def fit(self, X, y, **kwargs):
        """Train sklearn model"""
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict với sklearn model"""
        return self.model.predict(X)
    
    def predict_proba(self, X) -> Optional[np.ndarray]:
        """Predict probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def get_params(self) -> dict:
        """Lấy parameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}
    
    def save_model(self, path: str):
        """Lưu sklearn model"""
        joblib.dump(self.model, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load sklearn model"""
        self.model = joblib.load(path)
        print(f"✓ Model loaded from {path}")
        return self

