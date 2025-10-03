"""
Data Anonymization - Ẩn danh hóa dữ liệu cá nhân
"""

import numpy as np
import re
from typing import Any, List, Dict, Optional


class DataAnonymizer:
    """
    Anonymize data để bảo vệ personally identifiable information (PII)
    """
    
    def __init__(self):
        """Khởi tạo Data Anonymizer"""
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        self.anonymization_map = {}
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Phát hiện PII trong text
        
        Args:
            text: Text cần kiểm tra
        
        Returns:
            Dictionary với các loại PII tìm thấy
        """
        detected = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, str(text))
            if matches:
                detected[pii_type] = matches
        
        return detected
    
    def redact_pii(self, text: str, replacement: str = '[REDACTED]') -> str:
        """
        Xóa PII khỏi text
        
        Args:
            text: Text gốc
            replacement: Text thay thế
        
        Returns:
            Text đã được redact
        """
        text = str(text)
        
        for pattern in self.pii_patterns.values():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def mask_data(self, data: str, mask_char: str = '*', 
                 keep_first: int = 0, keep_last: int = 0) -> str:
        """
        Mask data với ký tự đặc biệt
        
        Args:
            data: Data cần mask
            mask_char: Ký tự dùng để mask
            keep_first: Số ký tự giữ lại ở đầu
            keep_last: Số ký tự giữ lại ở cuối
        
        Returns:
            Masked data
        """
        data = str(data)
        length = len(data)
        
        if length <= keep_first + keep_last:
            return mask_char * length
        
        masked_length = length - keep_first - keep_last
        
        return (data[:keep_first] + 
                mask_char * masked_length + 
                data[-keep_last:] if keep_last > 0 else '')
    
    def pseudonymize(self, data: str, salt: Optional[str] = None) -> str:
        """
        Pseudonymize data (thay thế bằng pseudonym nhất quán)
        
        Args:
            data: Data cần pseudonymize
            salt: Salt cho hashing
        
        Returns:
            Pseudonymized data
        """
        import hashlib
        
        data = str(data)
        
        # Kiểm tra xem đã có pseudonym chưa
        if data in self.anonymization_map:
            return self.anonymization_map[data]
        
        # Tạo pseudonym mới
        if salt:
            hash_input = (data + salt).encode()
        else:
            hash_input = data.encode()
        
        pseudonym = hashlib.sha256(hash_input).hexdigest()[:16]
        
        # Lưu mapping
        self.anonymization_map[data] = pseudonym
        
        return pseudonym
    
    def generalize_numeric(self, values: np.ndarray, 
                          bins: int = 5) -> np.ndarray:
        """
        Generalize numeric values thành ranges
        
        Args:
            values: Numeric values
            bins: Số lượng bins
        
        Returns:
            Generalized values (bin labels)
        """
        values = np.array(values)
        
        # Tạo bins
        bin_edges = np.percentile(values, np.linspace(0, 100, bins + 1))
        
        # Assign values to bins
        generalized = np.digitize(values, bin_edges[1:-1])
        
        return generalized
    
    def generalize_categorical(self, values: List[str],
                              hierarchy: Optional[Dict] = None) -> List[str]:
        """
        Generalize categorical values theo hierarchy
        
        Args:
            values: Categorical values
            hierarchy: Mapping từ specific -> general
        
        Returns:
            Generalized values
        """
        if hierarchy is None:
            # Mặc định: trả về 'OTHER' cho tất cả
            return ['OTHER'] * len(values)
        
        generalized = []
        for value in values:
            generalized.append(hierarchy.get(value, 'OTHER'))
        
        return generalized
    
    def k_anonymize(self, data: np.ndarray, sensitive_indices: List[int],
                   k: int = 5) -> np.ndarray:
        """
        Thực hiện k-anonymity
        
        K-anonymity đảm bảo mỗi individual không thể phân biệt được
        từ ít nhất k-1 individuals khác
        
        Args:
            data: Dataset
            sensitive_indices: Indices của sensitive attributes
            k: K value cho k-anonymity
        
        Returns:
            K-anonymized data
        """
        data = np.array(data)
        anonymized_data = data.copy()
        
        # Đơn giản hóa: generalize sensitive attributes
        for idx in sensitive_indices:
            if idx < data.shape[1]:
                column = data[:, idx]
                
                # Nếu numeric, generalize
                if np.issubdtype(column.dtype, np.number):
                    anonymized_data[:, idx] = self.generalize_numeric(column, bins=k)
        
        return anonymized_data
    
    def l_diversity(self, data: np.ndarray, quasi_identifiers: List[int],
                   sensitive_attr: int, l: int = 3) -> bool:
        """
        Kiểm tra l-diversity
        
        L-diversity đảm bảo mỗi equivalence class có ít nhất l giá trị
        khác nhau cho sensitive attribute
        
        Args:
            data: Dataset
            quasi_identifiers: Indices của quasi-identifier attributes
            sensitive_attr: Index của sensitive attribute
            l: L value
        
        Returns:
            True nếu đạt l-diversity
        """
        data = np.array(data)
        
        # Tạo equivalence classes dựa trên quasi-identifiers
        equivalence_classes = {}
        
        for row in data:
            qi_key = tuple(row[qi] for qi in quasi_identifiers)
            
            if qi_key not in equivalence_classes:
                equivalence_classes[qi_key] = []
            
            equivalence_classes[qi_key].append(row[sensitive_attr])
        
        # Kiểm tra mỗi equivalence class
        for sensitive_values in equivalence_classes.values():
            unique_values = len(set(sensitive_values))
            if unique_values < l:
                return False
        
        return True
    
    def anonymize_dataset(self, data: np.ndarray,
                         pii_columns: List[int],
                         method: str = 'mask') -> np.ndarray:
        """
        Anonymize toàn bộ dataset
        
        Args:
            data: Dataset
            pii_columns: Indices của PII columns
            method: 'mask', 'redact', hoặc 'pseudonymize'
        
        Returns:
            Anonymized dataset
        """
        data = np.array(data, dtype=object)
        anonymized = data.copy()
        
        for col_idx in pii_columns:
            if col_idx < data.shape[1]:
                column = data[:, col_idx]
                
                if method == 'mask':
                    anonymized[:, col_idx] = [
                        self.mask_data(val, keep_first=1, keep_last=1)
                        for val in column
                    ]
                elif method == 'redact':
                    anonymized[:, col_idx] = [
                        self.redact_pii(val) 
                        for val in column
                    ]
                elif method == 'pseudonymize':
                    anonymized[:, col_idx] = [
                        self.pseudonymize(val)
                        for val in column
                    ]
        
        return anonymized
    
    def generate_anonymization_report(self, original_data: np.ndarray,
                                     anonymized_data: np.ndarray) -> str:
        """
        Tạo báo cáo về anonymization
        
        Args:
            original_data: Dữ liệu gốc
            anonymized_data: Dữ liệu đã anonymize
        
        Returns:
            Báo cáo dạng string
        """
        report = []
        report.append("=" * 60)
        report.append("BÁO CÁO ANONYMIZATION")
        report.append("=" * 60)
        
        report.append(f"\nSố lượng records: {len(original_data)}")
        report.append(f"Số lượng attributes: {original_data.shape[1]}")
        report.append(f"Số lượng pseudonyms đã tạo: {len(self.anonymization_map)}")
        
        # Tính information loss
        if original_data.dtype in [int, float]:
            original_variance = np.var(original_data)
            anonymized_variance = np.var(anonymized_data.astype(float))
            info_loss = abs(original_variance - anonymized_variance) / original_variance * 100
            report.append(f"Information loss (variance): {info_loss:.2f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def clear_mappings(self):
        """Xóa tất cả anonymization mappings"""
        self.anonymization_map.clear()
    
    def __repr__(self) -> str:
        return f"DataAnonymizer(mappings={len(self.anonymization_map)})"

