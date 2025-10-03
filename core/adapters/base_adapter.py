"""
Base Model Adapter - Abstract interface cho tất cả ML frameworks
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np


class BaseModelAdapter(ABC):
    """
    Abstract base class cho model adapters
    
    Mọi adapter phải implement các phương thức này
    """
    
    def __init__(self, model: Any):
        """
        Khởi tạo adapter
        
        Args:
            model: Model instance
        """
        self.model = model
        self.framework = self._detect_framework()
    
    @abstractmethod
    def _detect_framework(self) -> str:
        """
        Detect ML framework của model
        
        Returns:
            Framework name
        """
        pass
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """
        Train model
        
        Args:
            X: Features
            y: Labels
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> Optional[np.ndarray]:
        """
        Predict probabilities (nếu hỗ trợ)
        
        Args:
            X: Features
        
        Returns:
            Probabilities hoặc None
        """
        pass
    
    @abstractmethod
    def get_params(self) -> dict:
        """
        Lấy model parameters
        
        Returns:
            Parameters dictionary
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str):
        """
        Lưu model
        
        Args:
            path: Path để lưu
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """
        Load model
        
        Args:
            path: Path của model
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(framework={self.framework})"

