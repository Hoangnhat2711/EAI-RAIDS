"""
Audit Reporter - Tạo các báo cáo audit chi tiết
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json


class AuditReporter:
    """
    Tạo các báo cáo audit toàn diện
    """
    
    def __init__(self, audit_logger: Any):
        """
        Khởi tạo Audit Reporter
        
        Args:
            audit_logger: Instance của AuditLogger
        """
        self.audit_logger = audit_logger
    
    def generate_compliance_report(self) -> str:
        """
        Tạo compliance report
        
        Returns:
            Report string
        """
        logs = self.audit_logger.read_logs()
        
        # Phân tích compliance
        fairness_checks = [
            log for log in logs 
            if log['event_type'] == 'fairness_check'
        ]
        
        bias_detections = [
            log for log in logs
            if log['event_type'] == 'bias_detection'
        ]
        
        privacy_operations = [
            log for log in logs
            if log['event_type'] == 'privacy_operation'
        ]
        
        report = []
        report.append("=" * 70)
        report.append("BÁO CÁO TUÂN THỦ (COMPLIANCE REPORT)")
        report.append("=" * 70)
        report.append(f"\nThời gian tạo: {datetime.now().isoformat()}")
        report.append(f"Session ID: {self.audit_logger.session_id}")
        
        # Fairness Compliance
        report.append("\n--- FAIRNESS COMPLIANCE ---")
        if fairness_checks:
            passed = sum(1 for log in fairness_checks if log['details'].get('passed', False))
            total = len(fairness_checks)
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            report.append(f"Số lần kiểm tra: {total}")
            report.append(f"Đạt chuẩn: {passed} ({pass_rate:.1f}%)")
            
            if pass_rate < 80:
                report.append("⚠ CẢNH BÁO: Tỷ lệ đạt fairness thấp!")
        else:
            report.append("Chưa có fairness checks")
        
        # Bias Detection
        report.append("\n--- BIAS DETECTION ---")
        if bias_detections:
            detected = sum(1 for log in bias_detections if log['details'].get('detected', False))
            total = len(bias_detections)
            
            report.append(f"Số lần quét: {total}")
            report.append(f"Phát hiện bias: {detected}")
            
            if detected > 0:
                report.append("⚠ Có bias được phát hiện, xem chi tiết bên dưới")
                
                for log in bias_detections:
                    if log['details'].get('detected', False):
                        bias_type = log['details'].get('bias_type', 'Unknown')
                        severity = log['details'].get('severity', 'Unknown')
                        report.append(f"  • {bias_type} (Severity: {severity})")
        else:
            report.append("Chưa có bias detection checks")
        
        # Privacy Compliance
        report.append("\n--- PRIVACY COMPLIANCE ---")
        if privacy_operations:
            report.append(f"Số operations: {len(privacy_operations)}")
            
            methods_used = set()
            for log in privacy_operations:
                method = log['details'].get('method', 'Unknown')
                methods_used.add(method)
            
            report.append(f"Phương pháp: {', '.join(methods_used)}")
        else:
            report.append("Chưa có privacy operations")
        
        # Overall Status
        report.append("\n--- TRẠNG THÁI TỔNG THỂ ---")
        
        issues = []
        if fairness_checks:
            passed = sum(1 for log in fairness_checks if log['details'].get('passed', False))
            if passed / len(fairness_checks) < 0.8:
                issues.append("Fairness compliance thấp")
        
        if bias_detections:
            detected = sum(1 for log in bias_detections if log['details'].get('detected', False))
            if detected > 0:
                issues.append(f"{detected} bias detected")
        
        if not privacy_operations:
            issues.append("Không có privacy protection")
        
        if issues:
            report.append("❌ CÓ VẤN ĐỀ:")
            for issue in issues:
                report.append(f"  • {issue}")
        else:
            report.append("✅ TẤT CẢ ĐỀU ĐẠT CHUẨN")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def generate_activity_report(self, time_range: Optional[int] = None) -> str:
        """
        Tạo activity report
        
        Args:
            time_range: Số giờ để xem (None = tất cả)
        
        Returns:
            Report string
        """
        logs = self.audit_logger.read_logs()
        
        # Filter by time range nếu cần
        if time_range:
            cutoff_time = datetime.now() - timedelta(hours=time_range)
            logs = [
                log for log in logs
                if datetime.fromisoformat(log['timestamp']) > cutoff_time
            ]
        
        report = []
        report.append("=" * 70)
        report.append("BÁO CÁO HOẠT ĐỘNG (ACTIVITY REPORT)")
        report.append("=" * 70)
        
        if time_range:
            report.append(f"\nTime Range: {time_range} giờ gần nhất")
        else:
            report.append("\nTime Range: Tất cả")
        
        report.append(f"Tổng số events: {len(logs)}")
        
        # Event breakdown
        report.append("\n--- PHÂN LOẠI EVENTS ---")
        event_counts = {}
        for log in logs:
            event_type = log['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {event_type}: {count}")
        
        # Recent activities
        report.append("\n--- HOẠT ĐỘNG GẦN ĐÂY ---")
        recent_logs = logs[-10:] if len(logs) > 10 else logs
        
        for log in reversed(recent_logs):
            timestamp = log['timestamp']
            event_type = log['event_type']
            report.append(f"  [{timestamp}] {event_type}")
        
        # Error summary
        errors = [log for log in logs if log['event_type'] == 'error']
        if errors:
            report.append("\n--- ERRORS ---")
            report.append(f"⚠ Tổng số errors: {len(errors)}")
            
            for error_log in errors[-5:]:
                error_msg = error_log['details'].get('error_message', 'Unknown')
                timestamp = error_log['timestamp']
                report.append(f"  [{timestamp}] {error_msg}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def generate_model_report(self) -> str:
        """
        Tạo report về model activities
        
        Returns:
            Report string
        """
        logs = self.audit_logger.read_logs()
        
        training_logs = [log for log in logs if log['event_type'] == 'training']
        prediction_logs = [log for log in logs if log['event_type'] == 'prediction']
        
        report = []
        report.append("=" * 70)
        report.append("BÁO CÁO MODEL")
        report.append("=" * 70)
        
        # Training summary
        report.append("\n--- TRAINING ACTIVITIES ---")
        if training_logs:
            report.append(f"Số lần training: {len(training_logs)}")
            
            for i, log in enumerate(training_logs, 1):
                details = log['details']
                model_type = details.get('model_type', 'Unknown')
                timestamp = log['timestamp']
                
                report.append(f"\n  Training #{i}:")
                report.append(f"    Thời gian: {timestamp}")
                report.append(f"    Model: {model_type}")
                
                metrics = details.get('metrics', {})
                if metrics:
                    report.append(f"    Metrics:")
                    for metric, value in metrics.items():
                        report.append(f"      • {metric}: {value}")
        else:
            report.append("Chưa có training activities")
        
        # Prediction summary
        report.append("\n--- PREDICTION ACTIVITIES ---")
        if prediction_logs:
            report.append(f"Tổng số predictions: {len(prediction_logs)}")
            
            # Average confidence
            confidences = [
                log['details'].get('confidence', 0)
                for log in prediction_logs
                if log['details'].get('confidence') is not None
            ]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                report.append(f"Average confidence: {avg_confidence:.4f}")
        else:
            report.append("Chưa có prediction activities")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def generate_full_report(self) -> str:
        """
        Tạo full comprehensive report
        
        Returns:
            Full report string
        """
        reports = []
        
        reports.append("=" * 70)
        reports.append("BÁO CÁO TOÀN DIỆN RESPONSIBLE AI")
        reports.append("=" * 70)
        reports.append(f"\nThời gian: {datetime.now().isoformat()}")
        reports.append(f"Session: {self.audit_logger.session_id}\n")
        
        # Thêm các sub-reports
        reports.append(self.generate_compliance_report())
        reports.append("\n\n")
        reports.append(self.generate_activity_report())
        reports.append("\n\n")
        reports.append(self.generate_model_report())
        
        return "\n".join(reports)
    
    def export_report(self, report_type: str = 'full', 
                     output_file: Optional[str] = None):
        """
        Export report ra file
        
        Args:
            report_type: 'full', 'compliance', 'activity', hoặc 'model'
            output_file: Output file path
        """
        if report_type == 'full':
            report = self.generate_full_report()
        elif report_type == 'compliance':
            report = self.generate_compliance_report()
        elif report_type == 'activity':
            report = self.generate_activity_report()
        elif report_type == 'model':
            report = self.generate_model_report()
        else:
            raise ValueError(f"Report type '{report_type}' không được hỗ trợ")
        
        if output_file is None:
            output_file = f"audit_report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report đã được export: {output_file}")
    
    def __repr__(self) -> str:
        return f"AuditReporter(logger_session={self.audit_logger.session_id})"

