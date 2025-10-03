"""
Abstract Base Handler cho Audit Logging
Triển khai Strategy Pattern để linh hoạt thay đổi backend
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class AbstractLogHandler(ABC):
    """
    Abstract base class cho tất cả log handlers
    
    Mọi handler cụ thể phải implement các phương thức này
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo handler
        
        Args:
            config: Configuration dictionary cho handler
        """
        self.config = config or {}
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Kết nối đến backend storage
        
        Returns:
            True nếu kết nối thành công
        """
        pass
    
    @abstractmethod
    def write_event(self, event: Dict[str, Any]) -> bool:
        """
        Ghi một event vào backend
        
        Args:
            event: Event data
        
        Returns:
            True nếu ghi thành công
        """
        pass
    
    @abstractmethod
    def read_events(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Đọc events từ backend
        
        Args:
            filters: Điều kiện lọc (event_type, timestamp range, etc.)
            limit: Số lượng events tối đa
        
        Returns:
            List of events
        """
        pass
    
    @abstractmethod
    def close(self):
        """Đóng kết nối đến backend"""
        pass
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

