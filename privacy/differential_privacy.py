"""
Differential Privacy - Bảo vệ privacy bằng cách thêm noise vào data
"""

import numpy as np
from typing import Any, Optional, Tuple


class DifferentialPrivacy:
    """
    Triển khai Differential Privacy để bảo vệ dữ liệu cá nhân
    
    Differential Privacy đảm bảo rằng việc thêm/xóa một data point
    không ảnh hưởng đáng kể đến kết quả, bảo vệ privacy của individuals
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Khởi tạo Differential Privacy
        
        Args:
            epsilon: Privacy budget (càng nhỏ càng private, nhưng càng ít accurate)
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta
        
        if epsilon <= 0:
            raise ValueError("Epsilon phải > 0")
        if delta < 0 or delta >= 1:
            raise ValueError("Delta phải trong khoảng [0, 1)")
    
    def add_laplace_noise(self, data: np.ndarray, 
                         sensitivity: float = 1.0) -> np.ndarray:
        """
        Thêm Laplace noise cho differential privacy
        
        Args:
            data: Dữ liệu gốc
            sensitivity: Sensitivity của function (max change từ 1 data point)
        
        Returns:
            Dữ liệu đã được thêm noise
        """
        data = np.array(data, dtype=float)
        
        # Tính scale cho Laplace distribution
        scale = sensitivity / self.epsilon
        
        # Tạo Laplace noise
        noise = np.random.laplace(0, scale, data.shape)
        
        return data + noise
    
    def add_gaussian_noise(self, data: np.ndarray, 
                          sensitivity: float = 1.0) -> np.ndarray:
        """
        Thêm Gaussian noise cho differential privacy
        
        Args:
            data: Dữ liệu gốc
            sensitivity: Sensitivity của function
        
        Returns:
            Dữ liệu đã được thêm noise
        """
        data = np.array(data, dtype=float)
        
        # Tính standard deviation cho Gaussian
        # Theo Gaussian mechanism: sigma >= sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # Tạo Gaussian noise
        noise = np.random.normal(0, sigma, data.shape)
        
        return data + noise
    
    def privatize_dataset(self, X: np.ndarray, method: str = 'laplace',
                         sensitivity: float = 1.0) -> np.ndarray:
        """
        Privatize toàn bộ dataset
        
        Args:
            X: Dataset
            method: 'laplace' hoặc 'gaussian'
            sensitivity: Sensitivity
        
        Returns:
            Privatized dataset
        """
        if method.lower() == 'laplace':
            return self.add_laplace_noise(X, sensitivity)
        elif method.lower() == 'gaussian':
            return self.add_gaussian_noise(X, sensitivity)
        else:
            raise ValueError(f"Method '{method}' không được hỗ trợ. Chọn 'laplace' hoặc 'gaussian'.")
    
    def private_mean(self, data: np.ndarray, 
                    data_range: Tuple[float, float]) -> float:
        """
        Tính mean với differential privacy
        
        Args:
            data: Dữ liệu
            data_range: (min, max) của dữ liệu
        
        Returns:
            Private mean
        """
        data = np.array(data)
        
        # Sensitivity của mean = (max - min) / n
        sensitivity = (data_range[1] - data_range[0]) / len(data)
        
        # Tính mean thực
        true_mean = np.mean(data)
        
        # Thêm Laplace noise
        noise = np.random.laplace(0, sensitivity / self.epsilon)
        
        return true_mean + noise
    
    def private_sum(self, data: np.ndarray,
                   data_range: Tuple[float, float]) -> float:
        """
        Tính sum với differential privacy
        
        Args:
            data: Dữ liệu
            data_range: (min, max) của dữ liệu
        
        Returns:
            Private sum
        """
        data = np.array(data)
        
        # Sensitivity của sum = max - min
        sensitivity = data_range[1] - data_range[0]
        
        # Tính sum thực
        true_sum = np.sum(data)
        
        # Thêm Laplace noise
        noise = np.random.laplace(0, sensitivity / self.epsilon)
        
        return true_sum + noise
    
    def private_count(self, data: np.ndarray, 
                     condition_fn=None) -> int:
        """
        Đếm với differential privacy
        
        Args:
            data: Dữ liệu
            condition_fn: Function để filter (optional)
        
        Returns:
            Private count
        """
        data = np.array(data)
        
        if condition_fn is not None:
            count = np.sum(condition_fn(data))
        else:
            count = len(data)
        
        # Sensitivity của count = 1
        sensitivity = 1
        
        # Thêm Laplace noise
        noise = np.random.laplace(0, sensitivity / self.epsilon)
        
        return int(max(0, count + noise))  # Count không thể âm
    
    def private_histogram(self, data: np.ndarray, 
                         bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo histogram với differential privacy
        
        Args:
            data: Dữ liệu
            bins: Số lượng bins
        
        Returns:
            (counts, bin_edges) với noise
        """
        data = np.array(data)
        
        # Tạo histogram thực
        counts, bin_edges = np.histogram(data, bins=bins)
        
        # Sensitivity = 1 (một data point chỉ ảnh hưởng 1 bin)
        sensitivity = 1
        scale = sensitivity / self.epsilon
        
        # Thêm Laplace noise vào mỗi bin
        noise = np.random.laplace(0, scale, len(counts))
        private_counts = np.maximum(0, counts + noise)  # Counts không thể âm
        
        return private_counts, bin_edges
    
    def get_privacy_budget_remaining(self, used_epsilon: float) -> float:
        """
        Tính privacy budget còn lại
        
        Args:
            used_epsilon: Epsilon đã sử dụng
        
        Returns:
            Epsilon còn lại
        """
        return max(0, self.epsilon - used_epsilon)
    
    def compose_privacy(self, epsilons: list, deltas: list) -> Tuple[float, float]:
        """
        Tính composed privacy parameters khi sử dụng nhiều mechanisms
        
        Args:
            epsilons: List các epsilon đã sử dụng
            deltas: List các delta đã sử dụng
        
        Returns:
            (total_epsilon, total_delta)
        """
        # Basic composition (conservative)
        total_epsilon = sum(epsilons)
        total_delta = sum(deltas)
        
        return total_epsilon, total_delta
    
    def advanced_composition(self, epsilon_per_query: float,
                           num_queries: int,
                           delta_prime: float = 1e-6) -> Tuple[float, float]:
        """
        Advanced composition theorem cho privacy
        
        Args:
            epsilon_per_query: Epsilon cho mỗi query
            num_queries: Số lượng queries
            delta_prime: Additional delta parameter
        
        Returns:
            (total_epsilon, total_delta)
        """
        # Advanced composition theorem
        total_epsilon = epsilon_per_query * np.sqrt(2 * num_queries * np.log(1 / delta_prime))
        total_delta = num_queries * self.delta + delta_prime
        
        return total_epsilon, total_delta
    
    def check_privacy_guarantee(self, epsilon_used: float,
                               delta_used: float) -> Dict[str, Any]:
        """
        Kiểm tra privacy guarantee
        
        Args:
            epsilon_used: Epsilon đã sử dụng
            delta_used: Delta đã sử dụng
        
        Returns:
            Dictionary với privacy status
        """
        privacy_status = {
            'epsilon_budget': self.epsilon,
            'epsilon_used': epsilon_used,
            'epsilon_remaining': self.get_privacy_budget_remaining(epsilon_used),
            'delta_budget': self.delta,
            'delta_used': delta_used,
            'is_within_budget': epsilon_used <= self.epsilon and delta_used <= self.delta,
            'privacy_level': self._get_privacy_level(epsilon_used)
        }
        
        return privacy_status
    
    def _get_privacy_level(self, epsilon: float) -> str:
        """Đánh giá mức độ privacy dựa trên epsilon"""
        if epsilon < 0.1:
            return 'Very High Privacy'
        elif epsilon < 1.0:
            return 'High Privacy'
        elif epsilon < 5.0:
            return 'Moderate Privacy'
        elif epsilon < 10.0:
            return 'Low Privacy'
        else:
            return 'Very Low Privacy'
    
    def __repr__(self) -> str:
        return f"DifferentialPrivacy(epsilon={self.epsilon}, delta={self.delta})"

