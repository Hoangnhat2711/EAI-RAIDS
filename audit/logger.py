"""
Audit Logger - Ghi nhận tất cả các quyết định và hành động của AI
Hỗ trợ multiple backends thông qua Strategy Pattern
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging


class AuditLogger:
    """
    Logger để ghi nhận audit trail cho AI system
    
    Hỗ trợ multiple handlers:
    - FileLogHandler (default)
    - PostgreSQLHandler
    - ElasticsearchHandler
    - CloudLoggerHandler
    """
    
    def __init__(self, log_dir: str = 'audit_logs', 
                 responsible_ai_framework: Optional[Any] = None,
                 handlers: Optional[List[Any]] = None):
        """
        Khởi tạo Audit Logger
        
        Args:
            log_dir: Thư mục lưu audit logs (cho FileHandler)
            responsible_ai_framework: Instance của ResponsibleAI
            handlers: List of AbstractLogHandler instances (None = dùng FileHandler)
        """
        self.log_dir = log_dir
        self.rai = responsible_ai_framework
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup handlers
        if handlers is None:
            # Default: FileLogHandler
            from audit.handlers import FileLogHandler
            
            os.makedirs(log_dir, exist_ok=True)
            self.handlers = [FileLogHandler({
                'log_dir': log_dir,
                'log_file': f'audit_log_{self.session_id}.jsonl'
            })]
        else:
            self.handlers = handlers
        
        # Connect all handlers
        for handler in self.handlers:
            handler.connect()
        
        # File paths (for backward compatibility)
        self.current_log_file = os.path.join(
            log_dir, 
            f'audit_log_{self.session_id}.jsonl'
        )
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Audit Logger initialized. Session: {self.session_id}")
        self.logger.info(f"Active handlers: {[type(h).__name__ for h in self.handlers]}")
    
    def _setup_logging(self):
        """Setup Python logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AuditLogger')
    
    def log_event(self, event_type: str, details: Dict[str, Any],
                 user_id: Optional[str] = None,
                 metadata: Optional[Dict] = None):
        """
        Ghi một audit event
        
        Args:
            event_type: Loại event (prediction, training, decision, etc.)
            details: Chi tiết về event
            user_id: ID của user (nếu có)
            metadata: Metadata bổ sung
        """
        event = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'user_id': user_id,
            'metadata': metadata or {}
        }
        
        # Ghi vào file
        self._write_to_file(event)
        
        # Log ra console
        self.logger.info(f"Audit Event: {event_type}")
    
    def log_prediction(self, input_data: Any, prediction: Any,
                      model_info: Optional[Dict] = None,
                      confidence: Optional[float] = None):
        """
        Ghi prediction event
        
        Args:
            input_data: Input data
            prediction: Prediction result
            model_info: Thông tin về model
            confidence: Confidence score
        """
        details = {
            'input_shape': str(getattr(input_data, 'shape', len(input_data))),
            'prediction': str(prediction),
            'model_info': model_info or {},
            'confidence': confidence
        }
        
        self.log_event('prediction', details)
    
    def log_training(self, model_type: str, dataset_info: Dict,
                    hyperparameters: Dict, metrics: Dict):
        """
        Ghi training event
        
        Args:
            model_type: Loại model
            dataset_info: Thông tin về dataset
            hyperparameters: Hyperparameters
            metrics: Training metrics
        """
        details = {
            'model_type': model_type,
            'dataset_info': dataset_info,
            'hyperparameters': hyperparameters,
            'metrics': metrics
        }
        
        self.log_event('training', details)
    
    def log_decision(self, decision_type: str, rationale: str,
                    alternatives: Optional[List] = None,
                    confidence: Optional[float] = None):
        """
        Ghi decision event với rationale
        
        Args:
            decision_type: Loại decision
            rationale: Lý do cho decision
            alternatives: Các alternatives đã xem xét
            confidence: Confidence level
        """
        details = {
            'decision_type': decision_type,
            'rationale': rationale,
            'alternatives': alternatives or [],
            'confidence': confidence
        }
        
        self.log_event('decision', details)
    
    def log_fairness_check(self, metrics: Dict, passed: bool,
                          threshold: float):
        """
        Ghi fairness check event
        
        Args:
            metrics: Fairness metrics
            passed: Có pass không
            threshold: Ngưỡng fairness
        """
        details = {
            'metrics': metrics,
            'passed': passed,
            'threshold': threshold
        }
        
        self.log_event('fairness_check', details)
    
    def log_privacy_operation(self, operation: str, method: str,
                             parameters: Dict):
        """
        Ghi privacy operation event
        
        Args:
            operation: Loại operation
            method: Phương pháp privacy
            parameters: Parameters (epsilon, delta, etc.)
        """
        details = {
            'operation': operation,
            'method': method,
            'parameters': parameters
        }
        
        self.log_event('privacy_operation', details)
    
    def log_bias_detection(self, bias_type: str, detected: bool,
                          severity: str, details: Dict):
        """
        Ghi bias detection event
        
        Args:
            bias_type: Loại bias
            detected: Có phát hiện bias không
            severity: Mức độ nghiêm trọng
            details: Chi tiết
        """
        event_details = {
            'bias_type': bias_type,
            'detected': detected,
            'severity': severity,
            'details': details
        }
        
        self.log_event('bias_detection', event_details)
    
    def log_error(self, error_type: str, error_message: str,
                 stack_trace: Optional[str] = None):
        """
        Ghi error event
        
        Args:
            error_type: Loại error
            error_message: Error message
            stack_trace: Stack trace (nếu có)
        """
        details = {
            'error_type': error_type,
            'error_message': error_message,
            'stack_trace': stack_trace
        }
        
        self.log_event('error', details)
        self.logger.error(f"Error logged: {error_type} - {error_message}")
    
    def _write_to_file(self, event: Dict):
        """Ghi event vào tất cả handlers"""
        for handler in self.handlers:
            try:
                success = handler.write_event(event)
                if not success:
                    self.logger.warning(f"Failed to write to {type(handler).__name__}")
            except Exception as e:
                self.logger.error(f"Error in {type(handler).__name__}: {e}")
    
    def read_logs(self, event_type: Optional[str] = None,
                 limit: Optional[int] = None) -> List[Dict]:
        """
        Đọc audit logs
        
        Args:
            event_type: Filter theo event type
            limit: Số lượng logs tối đa
        
        Returns:
            List of log events
        """
        logs = []
        
        try:
            with open(self.current_log_file, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    
                    if event_type is None or event['event_type'] == event_type:
                        logs.append(event)
        except FileNotFoundError:
            self.logger.warning(f"Log file not found: {self.current_log_file}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading logs: {e}")
            return []
        
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def get_event_summary(self) -> Dict[str, int]:
        """
        Lấy summary về các events
        
        Returns:
            Dictionary với count của mỗi event type
        """
        summary = {}
        logs = self.read_logs()
        
        for log in logs:
            event_type = log['event_type']
            summary[event_type] = summary.get(event_type, 0) + 1
        
        return summary
    
    def generate_summary_report(self) -> str:
        """
        Tạo summary report
        
        Returns:
            Report string
        """
        summary = self.get_event_summary()
        logs = self.read_logs()
        
        report = []
        report.append("=" * 60)
        report.append("AUDIT LOG SUMMARY")
        report.append("=" * 60)
        report.append(f"\nSession ID: {self.session_id}")
        report.append(f"Total Events: {len(logs)}")
        
        if logs:
            first_event = logs[0]['timestamp']
            last_event = logs[-1]['timestamp']
            report.append(f"Time Range: {first_event} to {last_event}")
        
        report.append("\nEvent Types:")
        for event_type, count in sorted(summary.items()):
            report.append(f"  - {event_type}: {count}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def export_logs(self, output_file: str, format: str = 'json'):
        """
        Export logs sang file khác
        
        Args:
            output_file: Output file path
            format: 'json' hoặc 'csv'
        """
        logs = self.read_logs()
        
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(logs, f, indent=2)
        elif format == 'csv':
            import csv
            
            if not logs:
                return
            
            keys = logs[0].keys()
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(logs)
        else:
            raise ValueError(f"Format '{format}' không được hỗ trợ")
        
        self.logger.info(f"Logs exported to {output_file}")
    
    def add_handler(self, handler: Any):
        """
        Thêm handler mới
        
        Args:
            handler: AbstractLogHandler instance
        """
        if handler.connect():
            self.handlers.append(handler)
            self.logger.info(f"Added handler: {type(handler).__name__}")
        else:
            self.logger.error(f"Failed to add handler: {type(handler).__name__}")
    
    def close_all_handlers(self):
        """Đóng tất cả handlers"""
        for handler in self.handlers:
            try:
                handler.close()
            except Exception as e:
                self.logger.error(f"Error closing {type(handler).__name__}: {e}")
    
    def __del__(self):
        """Cleanup khi object bị destroy"""
        self.close_all_handlers()
    
    def __repr__(self) -> str:
        return f"AuditLogger(session={self.session_id}, handlers={len(self.handlers)})"

