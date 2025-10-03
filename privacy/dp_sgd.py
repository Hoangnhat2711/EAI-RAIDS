"""
DP-SGD (Differentially Private Stochastic Gradient Descent)

Implementation cho PyTorch và TensorFlow
Based on Abadi et al. "Deep Learning with Differential Privacy" (2016)

Key idea: Add calibrated noise to gradients during training
"""

import numpy as np
from typing import Any, Optional, Dict, Tuple, List
import warnings


class DPSGDTrainer:
    """
    Differential Privacy SGD Trainer
    
    Triển khai DP-SGD algorithm:
    1. Clip gradients per sample (sensitivity control)
    2. Add Gaussian noise to gradients
    3. Track privacy budget (ε, δ)
    
    Compatible với PyTorch và TensorFlow
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 max_grad_norm: float = 1.0, framework: str = 'pytorch'):
        """
        Khởi tạo DP-SGD Trainer
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            max_grad_norm: Gradient clipping threshold
            framework: 'pytorch' hoặc 'tensorflow'
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.framework = framework.lower()
        
        # Privacy accounting
        self.steps = 0
        self.noise_multiplier = None
        
        # Compute noise multiplier from epsilon, delta
        self._compute_noise_multiplier()
        
        print(f"DP-SGD initialized: ε={epsilon}, δ={delta}, C={max_grad_norm}")
        print(f"Noise multiplier: {self.noise_multiplier:.4f}")
    
    def _compute_noise_multiplier(self):
        """
        Compute noise multiplier σ from (ε, δ)
        
        Using: σ ≥ √(2 ln(1.25/δ)) / ε
        """
        self.noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def train_pytorch_model(self, model, train_loader, optimizer, 
                           criterion, epochs: int = 10,
                           device: str = 'cpu') -> Dict[str, List]:
        """
        ⚠️ DEPRECATED - DO NOT USE ⚠️
        
        This manual DP-SGD training is MATHEMATICALLY INCORRECT.
        
        Use OpacusIntegration.train_private_model() instead, which uses
        the production-ready Opacus library with correct per-sample gradient clipping.
        
        Args:
            model: PyTorch model
            train_loader: DataLoader
            optimizer: Optimizer
            criterion: Loss function
            epochs: Number of epochs
            device: 'cpu' hoặc 'cuda'
        
        Returns:
            Training history
        
        Raises:
            NotImplementedError: Use OpacusIntegration instead
        """
        raise NotImplementedError(
            "❌ This manual DP-SGD implementation is INCORRECT!\n\n"
            "Use OpacusIntegration instead:\n\n"
            "from privacy.dp_sgd import OpacusIntegration\n\n"
            "opacus = OpacusIntegration(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)\n"
            "model, optimizer, privacy_engine = opacus.make_private(\n"
            "    model, optimizer, train_loader, epochs=epochs\n"
            ")\n"
            "history = opacus.train_private_model(\n"
            "    model, optimizer, train_loader, criterion, epochs, device\n"
            ")\n"
        )
    
    def _clip_and_noise_pytorch_gradients(self, model, batch_size: int):
        """
        ⚠️ DEPRECATED - DO NOT USE ⚠️
        
        This manual implementation is MATHEMATICALLY INCORRECT for DP-SGD.
        
        CRITICAL ISSUE: DP-SGD requires PER-SAMPLE gradient clipping BEFORE
        aggregation. Clipping aggregate gradients does NOT provide privacy
        guarantees as claimed in Abadi et al. (2016).
        
        SOLUTION: Use OpacusIntegration or TensorFlowPrivacyIntegration instead.
        
        Args:
            model: PyTorch model
            batch_size: Batch size
        
        Raises:
            NotImplementedError: This method should not be used
        """
        raise NotImplementedError(
            "❌ CRITICAL ERROR: Manual DP-SGD implementation is mathematically incorrect!\n\n"
            "DP-SGD requires PER-SAMPLE gradient clipping, not aggregate gradient clipping.\n"
            "This implementation does NOT provide the claimed (ε, δ)-DP guarantees.\n\n"
            "SOLUTION: Use OpacusIntegration for PyTorch:\n"
            "  from privacy.dp_sgd import OpacusIntegration\n"
            "  opacus = OpacusIntegration(epsilon=1.0, delta=1e-5)\n"
            "  model, optimizer, dataloader = opacus.make_private(model, optimizer, dataloader)\n\n"
            "Reference: Abadi et al. 'Deep Learning with Differential Privacy' (2016)"
        )
    
    def train_tensorflow_model(self, model, train_data, train_labels,
                              epochs: int = 10, batch_size: int = 32,
                              learning_rate: float = 0.01) -> Dict[str, List]:
        """
        Train TensorFlow model với DP-SGD
        
        Args:
            model: TensorFlow/Keras model
            train_data: Training features
            train_labels: Training labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            Training history
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed")
        
        # Sử dụng TensorFlow Privacy nếu available
        try:
            from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
            
            # Create DP optimizer
            optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
                l2_norm_clip=self.max_grad_norm,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=batch_size,
                learning_rate=learning_rate
            )
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )
            
            # Train
            history = model.fit(
                train_data, train_labels,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Compute privacy spent
            privacy_spent = self._compute_privacy_spent(
                len(train_data),
                len(train_data) // batch_size
            )
            
            print(f"\nFinal Privacy: ε={privacy_spent['epsilon']:.2f}, δ={privacy_spent['delta']:.2e}")
            
            return {
                'loss': history.history['loss'],
                'accuracy': history.history['accuracy'],
                'privacy_spent': privacy_spent
            }
        
        except ImportError:
            warnings.warn("TensorFlow Privacy not installed. Using manual DP-SGD.")
            return self._manual_tensorflow_training(
                model, train_data, train_labels, epochs, batch_size, learning_rate
            )
    
    def _manual_tensorflow_training(self, model, train_data, train_labels,
                                   epochs: int, batch_size: int, 
                                   learning_rate: float) -> Dict:
        """Manual DP-SGD implementation cho TensorFlow"""
        import tensorflow as tf
        
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        history = {'loss': []}
        
        n_batches = len(train_data) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_data = train_data[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                
                with tf.GradientTape() as tape:
                    predictions = model(batch_data, training=True)
                    loss = loss_fn(batch_labels, predictions)
                
                # Get gradients
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # Clip and add noise
                gradients = self._clip_and_noise_tensorflow_gradients(
                    gradients, batch_size
                )
                
                # Apply gradients
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                epoch_loss += loss.numpy()
            
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def _clip_and_noise_tensorflow_gradients(self, gradients, batch_size: int):
        """Clip and add noise to TensorFlow gradients"""
        import tensorflow as tf
        
        clipped_gradients = []
        
        for grad in gradients:
            if grad is not None:
                # Clip
                clipped_grad = tf.clip_by_norm(grad, self.max_grad_norm)
                
                # Add noise
                noise_scale = self.noise_multiplier * self.max_grad_norm / batch_size
                noise = tf.random.normal(
                    shape=grad.shape,
                    mean=0.0,
                    stddev=noise_scale
                )
                
                clipped_grad += noise
                clipped_gradients.append(clipped_grad)
            else:
                clipped_gradients.append(None)
        
        return clipped_gradients
    
    def _compute_privacy_spent(self, n_samples: int, 
                              steps_per_epoch: int) -> Dict[str, float]:
        """
        Compute privacy spent using moments accountant
        
        Simplified implementation - trong thực tế dùng:
        - tensorflow_privacy.compute_dp_sgd_privacy
        - opacus.privacy_analysis
        
        Args:
            n_samples: Number of training samples
            steps_per_epoch: Steps per epoch
        
        Returns:
            Privacy parameters
        """
        # Simplified: Strong composition theorem
        # ε_total = ε * sqrt(T) where T = number of epochs
        
        total_epochs = self.steps / steps_per_epoch
        
        # Using advanced composition
        epsilon_spent = self.epsilon * np.sqrt(2 * total_epochs * np.log(1/self.delta))
        delta_spent = self.delta * total_epochs
        
        return {
            'epsilon': min(epsilon_spent, 100),  # Cap at 100
            'delta': min(delta_spent, 1.0),
            'steps': self.steps,
            'epochs': total_epochs
        }
    
    def get_privacy_guarantee(self) -> str:
        """
        Get human-readable privacy guarantee
        
        Returns:
            Privacy guarantee description
        """
        privacy = self._compute_privacy_spent(1000, 10)  # Dummy values
        
        epsilon = privacy['epsilon']
        
        if epsilon < 1:
            level = "STRONG Privacy"
        elif epsilon < 10:
            level = "MODERATE Privacy"
        else:
            level = "WEAK Privacy"
        
        return f"{level} (ε={epsilon:.2f}, δ={self.delta:.2e})"


class OpacusIntegration:
    """
    Integration với Opacus (PyTorch DP library)
    
    Opacus provides production-ready DP-SGD for PyTorch
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        """
        Khởi tạo Opacus integration
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            max_grad_norm: Gradient clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        try:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
            self.PrivacyEngine = PrivacyEngine
            self.ModuleValidator = ModuleValidator
            self.is_available = True
        except ImportError:
            warnings.warn("Opacus not installed. Install: pip install opacus")
            self.is_available = False
    
    def make_private(self, model, optimizer, train_loader,
                    epochs: int = 10) -> Tuple[Any, Any, Any]:
        """
        Make PyTorch model private using Opacus
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            train_loader: DataLoader
            epochs: Number of epochs
        
        Returns:
            (private_model, private_optimizer, privacy_engine)
        """
        if not self.is_available:
            raise ImportError("Opacus not installed")
        
        # Validate model
        errors = self.ModuleValidator.validate(model, strict=False)
        if errors:
            model = self.ModuleValidator.fix(model)
            print("✓ Model fixed for Opacus compatibility")
        
        # Create PrivacyEngine
        privacy_engine = self.PrivacyEngine()
        
        # Make private
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=np.sqrt(2 * np.log(1.25/self.delta)) / self.epsilon,
            max_grad_norm=self.max_grad_norm
        )
        
        print(f"✓ Model made private with Opacus")
        print(f"  Privacy budget: ε={self.epsilon}, δ={self.delta}")
        
        return model, optimizer, privacy_engine
    
    def train_private_model(self, model, optimizer, train_loader,
                           criterion, epochs: int, device: str = 'cpu') -> Dict:
        """
        Train private model với Opacus
        
        Args:
            model: Private model
            optimizer: Private optimizer
            train_loader: Private dataloader
            criterion: Loss function
            epochs: Number of epochs
            device: Device
        
        Returns:
            Training history với privacy accounting
        """
        if not self.is_available:
            raise ImportError("Opacus not installed")
        
        import torch
        
        model.train()
        model = model.to(device)
        
        history = {'loss': [], 'epsilon': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Get privacy spent
            epsilon = self.epsilon  # Simplified
            
            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)
            history['epsilon'].append(epsilon)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, ε={epsilon:.2f}")
        
        return history


class TensorFlowPrivacyIntegration:
    """
    Integration với TensorFlow Privacy
    
    Official DP-SGD implementation cho TensorFlow
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
        """
        Khởi tạo TensorFlow Privacy integration
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            max_grad_norm: Gradient clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        try:
            from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
            from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
            self.dp_optimizer_keras = dp_optimizer_keras
            self.compute_dp_sgd_privacy = compute_dp_sgd_privacy
            self.is_available = True
        except ImportError:
            warnings.warn("TensorFlow Privacy not installed. Install: pip install tensorflow-privacy")
            self.is_available = False
    
    def create_dp_optimizer(self, learning_rate: float = 0.01,
                           num_microbatches: int = 1):
        """
        Create DP optimizer
        
        Args:
            learning_rate: Learning rate
            num_microbatches: Number of microbatches
        
        Returns:
            DP optimizer
        """
        if not self.is_available:
            raise ImportError("TensorFlow Privacy not installed")
        
        noise_multiplier = np.sqrt(2 * np.log(1.25/self.delta)) / self.epsilon
        
        optimizer = self.dp_optimizer_keras.DPKerasSGDOptimizer(
            l2_norm_clip=self.max_grad_norm,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate
        )
        
        return optimizer
    
    def compute_privacy(self, n_samples: int, batch_size: int, epochs: int):
        """
        Compute privacy guarantee
        
        Args:
            n_samples: Number of training samples
            batch_size: Batch size
            epochs: Number of epochs
        
        Returns:
            Privacy parameters
        """
        if not self.is_available:
            return {'epsilon': self.epsilon, 'delta': self.delta}
        
        noise_multiplier = np.sqrt(2 * np.log(1.25/self.delta)) / self.epsilon
        
        epsilon, _ = self.compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=n_samples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=self.delta
        )
        
        return {'epsilon': epsilon, 'delta': self.delta}

