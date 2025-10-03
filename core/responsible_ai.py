"""
Lớp chính của Responsible AI Framework
Quản lý tất cả các thành phần và nguyên tắc AI có trách nhiệm
"""

import yaml
import logging
from typing import Dict, Any, Optional
from datetime import datetime


class ResponsibleAI:
    """
    Framework chính để đảm bảo AI có trách nhiệm
    
    Principles:
        - Fairness: Công bằng, không phân biệt đối xử
        - Transparency: Minh bạch trong quyết định
        - Privacy: Bảo vệ dữ liệu cá nhân
        - Accountability: Có trách nhiệm với quyết định
        - Explainability: Có thể giải thích được
        - Robustness: Vững chắc và đáng tin cậy
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Khởi tạo Responsible AI Framework
        
        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        self.config = self._load_config(config_path)
        self.principles = self.config.get('principles', {})
        self.audit_logs = []
        self.creation_time = datetime.now()
        
        # Thiết lập logging
        self._setup_logging()
        
        # Kiểm tra các nguyên tắc được kích hoạt
        self._validate_principles()
        
        self.logger.info("Responsible AI Framework đã được khởi tạo")
        self.logger.info(f"Các nguyên tắc được kích hoạt: {self.get_active_principles()}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Tải cấu hình từ file YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file config tại {config_path}, sử dụng config mặc định")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Trả về cấu hình mặc định"""
        return {
            'principles': {
                'fairness': True,
                'transparency': True,
                'privacy': True,
                'accountability': True,
                'explainability': True,
                'robustness': True
            },
            'fairness': {
                'metrics': ['demographic_parity', 'equalized_odds'],
                'threshold': 0.8
            },
            'privacy': {
                'differential_privacy': {
                    'enabled': True,
                    'epsilon': 1.0,
                    'delta': 1e-5
                }
            },
            'audit': {
                'enabled': True,
                'log_predictions': True
            }
        }
    
    def _setup_logging(self):
        """Thiết lập hệ thống logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ResponsibleAI')
    
    def _validate_principles(self):
        """Xác thực các nguyên tắc được cấu hình"""
        required_principles = [
            'fairness', 'transparency', 'privacy', 
            'accountability', 'explainability', 'robustness'
        ]
        
        for principle in required_principles:
            if principle not in self.principles:
                self.logger.warning(f"Nguyên tắc '{principle}' không được cấu hình, mặc định là True")
                self.principles[principle] = True
    
    def get_active_principles(self) -> list:
        """Lấy danh sách các nguyên tắc đang được kích hoạt"""
        return [key for key, value in self.principles.items() if value]
    
    def is_principle_active(self, principle: str) -> bool:
        """Kiểm tra xem một nguyên tắc có được kích hoạt không"""
        return self.principles.get(principle, False)
    
    def log_decision(self, decision_type: str, details: Dict[str, Any]):
        """
        Ghi lại một quyết định của AI để audit
        
        Args:
            decision_type: Loại quyết định (prediction, recommendation, etc.)
            details: Chi tiết về quyết định
        """
        if not self.is_principle_active('accountability'):
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'decision_type': decision_type,
            'details': details
        }
        
        self.audit_logs.append(log_entry)
        self.logger.info(f"Đã ghi nhận quyết định: {decision_type}")
    
    def get_audit_logs(self, limit: Optional[int] = None) -> list:
        """
        Lấy audit logs
        
        Args:
            limit: Số lượng logs tối đa cần lấy
        
        Returns:
            Danh sách audit logs
        """
        if limit:
            return self.audit_logs[-limit:]
        return self.audit_logs
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy cấu hình
        
        Args:
            section: Section cụ thể cần lấy, None để lấy toàn bộ
        
        Returns:
            Cấu hình được yêu cầu
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def validate_compliance(self) -> Dict[str, bool]:
        """
        Kiểm tra tuân thủ các nguyên tắc
        
        Returns:
            Dictionary với kết quả kiểm tra cho từng nguyên tắc
        """
        compliance_status = {}
        
        for principle, is_active in self.principles.items():
            if is_active:
                compliance_status[principle] = self._check_principle_compliance(principle)
        
        return compliance_status
    
    def _check_principle_compliance(self, principle: str) -> bool:
        """Kiểm tra tuân thủ cho một nguyên tắc cụ thể"""
        # Đây là một implementation đơn giản, có thể mở rộng
        if principle == 'accountability':
            return len(self.audit_logs) > 0 or self.config.get('audit', {}).get('enabled', False)
        
        # Mặc định trả về True nếu nguyên tắc được kích hoạt
        return True
    
    def generate_summary_report(self) -> str:
        """
        Tạo báo cáo tổng kết về framework
        
        Returns:
            Báo cáo dạng string
        """
        report = []
        report.append("=" * 60)
        report.append("BÁO CÁO RESPONSIBLE AI FRAMEWORK")
        report.append("=" * 60)
        report.append(f"\nThời gian khởi tạo: {self.creation_time}")
        report.append(f"Số lượng quyết định đã ghi nhận: {len(self.audit_logs)}")
        
        report.append("\nCác nguyên tắc được kích hoạt:")
        for principle in self.get_active_principles():
            report.append(f"  ✓ {principle.capitalize()}")
        
        report.append("\nTrạng thái tuân thủ:")
        compliance = self.validate_compliance()
        for principle, status in compliance.items():
            icon = "✓" if status else "✗"
            report.append(f"  {icon} {principle.capitalize()}: {'Đạt' if status else 'Không đạt'}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def __repr__(self) -> str:
        return f"ResponsibleAI(principles={self.get_active_principles()})"

