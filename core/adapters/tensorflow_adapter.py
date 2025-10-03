"""
TensorFlow/Keras Adapter
"""

import numpy as np
from typing import Any, Optional
from .base_adapter import BaseModelAdapter


class TensorFlowAdapter(BaseModelAdapter):
    """
    Adapter cho TensorFlow/Keras models
    
    Yêu cầu: pip install tensorflow
    """
    
    def __init__(self, model: Any):
        """
        Khởi tạo TensorFlow adapter
        
        Args:
            model: TensorFlow/Keras model
        """
        super().__init__(model)
        
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Install: pip install tensorflow")
    
    def _detect_framework(self) -> str:
        """Detect TensorFlow"""
        return "tensorflow"
    
    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
        Train TensorFlow model
        
        Args:
            X: Features
            y: Labels
            epochs: Số epochs
            batch_size: Batch size
            **kwargs: Additional arguments (validation_split, callbacks, etc.)
        """
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict với TensorFlow model"""
        predictions = self.model.predict(X, verbose=0)
        
        # Nếu là classification với softmax output
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = (predictions > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities"""
        probabilities = self.model.predict(X, verbose=0)
        
        # Nếu binary classification với single output
        if len(probabilities.shape) == 1 or probabilities.shape[1] == 1:
            prob_positive = probabilities.flatten()
            probabilities = np.column_stack([1 - prob_positive, prob_positive])
        
        return probabilities
    
    def get_params(self) -> dict:
        """Lấy model parameters"""
        params = {}
        
        for layer in self.model.layers:
            weights = layer.get_weights()
            if weights:
                params[layer.name] = {
                    f'weight_{i}': w for i, w in enumerate(weights)
                }
        
        return params
    
    def save_model(self, path: str):
        """Lưu TensorFlow model"""
        self.model.save(path)
        print(f"✓ TensorFlow model saved to {path}")
    
    def load_model(self, path: str):
        """Load TensorFlow model"""
        self.model = self.tf.keras.models.load_model(path)
        print(f"✓ TensorFlow model loaded from {path}")
        return self
    
    def compute_gradients(self, X, y) -> np.ndarray:
        """
        Tính gradients (cho adversarial attacks)
        
        Args:
            X: Input
            y: Target labels
        
        Returns:
            Gradients w.r.t input
        """
        X_tensor = self.tf.convert_to_tensor(X, dtype=self.tf.float32)
        y_tensor = self.tf.convert_to_tensor(y)
        
        with self.tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor, training=False)
            
            # Loss
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class
                loss = self.tf.keras.losses.sparse_categorical_crossentropy(
                    y_tensor, predictions
                )
            else:
                # Binary
                loss = self.tf.keras.losses.binary_crossentropy(
                    y_tensor, predictions
                )
        
        gradients = tape.gradient(loss, X_tensor)
        
        return gradients.numpy()

