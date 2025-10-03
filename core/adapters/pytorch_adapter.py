"""
PyTorch Adapter
"""

import numpy as np
from typing import Any, Optional
from .base_adapter import BaseModelAdapter


class PyTorchAdapter(BaseModelAdapter):
    """
    Adapter cho PyTorch models
    
    Yêu cầu: pip install torch
    """
    
    def __init__(self, model: Any, loss_fn=None, optimizer=None, device='cpu'):
        """
        Khởi tạo PyTorch adapter
        
        Args:
            model: PyTorch model
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device ('cpu' hoặc 'cuda')
        """
        super().__init__(model)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        try:
            import torch
            self.torch = torch
            
            # Move model to device
            self.model = self.model.to(device)
        except ImportError:
            raise ImportError("PyTorch not installed. Install: pip install torch")
    
    def _detect_framework(self) -> str:
        """Detect PyTorch"""
        return "pytorch"
    
    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
        Train PyTorch model
        
        Args:
            X: Features (numpy array hoặc tensor)
            y: Labels (numpy array hoặc tensor)
            epochs: Số epochs
            batch_size: Batch size
            **kwargs: Additional arguments
        """
        # Convert to tensors
        if not isinstance(X, self.torch.Tensor):
            X = self.torch.tensor(X, dtype=self.torch.float32)
        if not isinstance(y, self.torch.Tensor):
            y = self.torch.tensor(y, dtype=self.torch.long)
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Create DataLoader
        dataset = self.torch.utils.data.TensorDataset(X, y)
        dataloader = self.torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict với PyTorch model"""
        self.model.eval()
        
        # Convert to tensor
        if not isinstance(X, self.torch.Tensor):
            X = self.torch.tensor(X, dtype=self.torch.float32)
        
        X = X.to(self.device)
        
        with self.torch.no_grad():
            outputs = self.model(X)
            predictions = self.torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities"""
        self.model.eval()
        
        if not isinstance(X, self.torch.Tensor):
            X = self.torch.tensor(X, dtype=self.torch.float32)
        
        X = X.to(self.device)
        
        with self.torch.no_grad():
            outputs = self.model(X)
            probabilities = self.torch.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_params(self) -> dict:
        """Lấy model parameters"""
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params
    
    def save_model(self, path: str):
        """Lưu PyTorch model"""
        self.torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }, path)
        print(f"✓ PyTorch model saved to {path}")
    
    def load_model(self, path: str):
        """Load PyTorch model"""
        checkpoint = self.torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ PyTorch model loaded from {path}")
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
        self.model.eval()
        
        # Convert to tensors với requires_grad
        if not isinstance(X, self.torch.Tensor):
            X = self.torch.tensor(X, dtype=self.torch.float32, requires_grad=True)
        else:
            X.requires_grad = True
        
        if not isinstance(y, self.torch.Tensor):
            y = self.torch.tensor(y, dtype=self.torch.long)
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        # Forward pass
        outputs = self.model(X)
        loss = self.loss_fn(outputs, y)
        
        # Backward pass để lấy gradients
        self.model.zero_grad()
        loss.backward()
        
        gradients = X.grad.detach().cpu().numpy()
        
        return gradients

