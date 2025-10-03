"""
Alert Manager - Tự động gửi cảnh báo khi phát hiện vấn đề nghiêm trọng
"""

import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum


class AlertLevel(Enum):
    """Mức độ nghiêm trọng của alert"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Kênh gửi alert"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"


class AlertManager:
    """
    Quản lý và gửi cảnh báo tự động
    
    Hỗ trợ:
    - Email notifications
    - Slack notifications
    - Webhook calls
    - SMS (Twilio)
    - Console logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo Alert Manager
        
        Args:
            config: Configuration cho các alert channels
        """
        self.config = config or {}
        self.alert_history = []
        self.enabled_channels = set()
        
        # Setup channels
        self._setup_channels()
        
        # Alert thresholds
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        self.fairness_threshold = self.config.get('fairness_threshold', 0.8)
        self.bias_severity_threshold = self.config.get('bias_severity', 'high')
    
    def _setup_channels(self):
        """Setup các alert channels được config"""
        channels_config = self.config.get('channels', {})
        
        if channels_config.get('console', {}).get('enabled', True):
            self.enabled_channels.add(AlertChannel.CONSOLE)
        
        if channels_config.get('email', {}).get('enabled', False):
            self.enabled_channels.add(AlertChannel.EMAIL)
            self.email_config = channels_config['email']
        
        if channels_config.get('slack', {}).get('enabled', False):
            self.enabled_channels.add(AlertChannel.SLACK)
            self.slack_config = channels_config['slack']
        
        if channels_config.get('webhook', {}).get('enabled', False):
            self.enabled_channels.add(AlertChannel.WEBHOOK)
            self.webhook_config = channels_config['webhook']
        
        print(f"Alert Manager initialized with channels: {[c.value for c in self.enabled_channels]}")
    
    def send_alert(self, level: AlertLevel, title: str, 
                   message: str, details: Optional[Dict] = None):
        """
        Gửi alert qua tất cả channels đã enable
        
        Args:
            level: Mức độ alert
            title: Tiêu đề alert
            message: Nội dung alert
            details: Chi tiết bổ sung
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'title': title,
            'message': message,
            'details': details or {}
        }
        
        # Lưu vào history
        self.alert_history.append(alert)
        
        # Gửi qua các channels
        for channel in self.enabled_channels:
            try:
                if channel == AlertChannel.CONSOLE:
                    self._send_console(alert)
                elif channel == AlertChannel.EMAIL:
                    self._send_email(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_slack(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_webhook(alert)
            except Exception as e:
                print(f"Error sending alert via {channel.value}: {e}")
    
    def _send_console(self, alert: Dict):
        """Gửi alert ra console"""
        level_icons = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        
        icon = level_icons.get(alert['level'], '📢')
        print(f"\n{icon} ALERT [{alert['level'].upper()}]")
        print(f"Time: {alert['timestamp']}")
        print(f"Title: {alert['title']}")
        print(f"Message: {alert['message']}")
        
        if alert['details']:
            print("Details:")
            for key, value in alert['details'].items():
                print(f"  - {key}: {value}")
        print()
    
    def _send_email(self, alert: Dict):
        """Gửi alert qua email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender_email = self.email_config.get('sender_email')
            sender_password = self.email_config.get('sender_password')
            recipient_emails = self.email_config.get('recipient_emails', [])
            
            if not all([smtp_server, sender_email, sender_password, recipient_emails]):
                print("Email config incomplete, skipping email notification")
                return
            
            # Tạo message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipient_emails)
            msg['Subject'] = f"[{alert['level'].upper()}] {alert['title']}"
            
            body = f"""
Responsible AI Alert

Level: {alert['level'].upper()}
Time: {alert['timestamp']}
Title: {alert['title']}

Message:
{alert['message']}

Details:
{json.dumps(alert['details'], indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Gửi email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            print(f"✓ Email alert sent to {len(recipient_emails)} recipients")
        
        except ImportError:
            print("smtplib not available")
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    def _send_slack(self, alert: Dict):
        """Gửi alert qua Slack"""
        try:
            import requests
            
            webhook_url = self.slack_config.get('webhook_url')
            
            if not webhook_url:
                print("Slack webhook URL not configured")
                return
            
            # Format cho Slack
            level_colors = {
                'info': '#36a64f',
                'warning': '#ff9900',
                'error': '#ff0000',
                'critical': '#990000'
            }
            
            slack_message = {
                "attachments": [
                    {
                        "color": level_colors.get(alert['level'], '#808080'),
                        "title": alert['title'],
                        "text": alert['message'],
                        "fields": [
                            {
                                "title": "Level",
                                "value": alert['level'].upper(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert['timestamp'],
                                "short": True
                            }
                        ],
                        "footer": "Responsible AI Framework"
                    }
                ]
            }
            
            # Thêm details
            if alert['details']:
                for key, value in alert['details'].items():
                    slack_message['attachments'][0]['fields'].append({
                        "title": key,
                        "value": str(value),
                        "short": True
                    })
            
            response = requests.post(webhook_url, json=slack_message)
            
            if response.status_code == 200:
                print("✓ Slack alert sent")
            else:
                print(f"Failed to send Slack alert: {response.status_code}")
        
        except ImportError:
            print("requests library not available")
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
    
    def _send_webhook(self, alert: Dict):
        """Gửi alert qua webhook"""
        try:
            import requests
            
            webhook_url = self.webhook_config.get('url')
            headers = self.webhook_config.get('headers', {})
            
            if not webhook_url:
                print("Webhook URL not configured")
                return
            
            response = requests.post(
                webhook_url,
                json=alert,
                headers=headers
            )
            
            if response.status_code in [200, 201, 204]:
                print("✓ Webhook alert sent")
            else:
                print(f"Failed to send webhook alert: {response.status_code}")
        
        except ImportError:
            print("requests library not available")
        except Exception as e:
            print(f"Failed to send webhook: {e}")
    
    def alert_on_drift(self, drift_result: Dict):
        """
        Gửi alert khi phát hiện drift
        
        Args:
            drift_result: Kết quả từ DriftDetector
        """
        if not drift_result.get('overall_drift_detected', False):
            return
        
        # Determine level
        level = AlertLevel.WARNING
        
        # Check data drift
        if drift_result.get('data_drift'):
            n_features_drifted = len(drift_result['data_drift'].get('features_with_drift', []))
            if n_features_drifted > 5:
                level = AlertLevel.ERROR
        
        # Check prediction drift
        if drift_result.get('prediction_drift'):
            if drift_result['prediction_drift'].get('mean_shift', 0) > self.drift_threshold:
                level = AlertLevel.ERROR
        
        self.send_alert(
            level=level,
            title="Model Drift Detected",
            message=f"Drift detected in model. Recommendation: Retrain model with recent data.",
            details={
                'data_drift': drift_result.get('data_drift', {}).get('drift_detected', False),
                'prediction_drift': drift_result.get('prediction_drift', {}).get('drift_detected', False),
                'concept_drift': drift_result.get('concept_drift', {}).get('drift_detected', False)
            }
        )
    
    def alert_on_fairness_violation(self, fairness_results: Dict):
        """
        Gửi alert khi vi phạm fairness
        
        Args:
            fairness_results: Kết quả từ FairnessMetrics
        """
        violations = []
        
        for metric, value in fairness_results.items():
            if metric == 'overall_fairness_score':
                continue
            
            if value < self.fairness_threshold:
                violations.append(f"{metric}: {value:.3f}")
        
        if not violations:
            return
        
        # Determine level
        level = AlertLevel.WARNING if len(violations) <= 2 else AlertLevel.ERROR
        
        self.send_alert(
            level=level,
            title="Fairness Violation Detected",
            message=f"Model failing {len(violations)} fairness metrics. Review and retrain recommended.",
            details={
                'violations': violations,
                'threshold': self.fairness_threshold
            }
        )
    
    def alert_on_bias(self, bias_report: Dict):
        """
        Gửi alert khi phát hiện bias
        
        Args:
            bias_report: Kết quả từ BiasDetector
        """
        if not bias_report.get('bias_detected', False):
            return
        
        # Determine level based on severity
        max_severity = 'low'
        for bias in bias_report.get('biases', []):
            if bias['severity'] == 'critical':
                max_severity = 'critical'
                break
            elif bias['severity'] == 'high' and max_severity != 'critical':
                max_severity = 'high'
            elif bias['severity'] == 'medium' and max_severity == 'low':
                max_severity = 'medium'
        
        level_map = {
            'low': AlertLevel.INFO,
            'medium': AlertLevel.WARNING,
            'high': AlertLevel.ERROR,
            'critical': AlertLevel.CRITICAL
        }
        
        level = level_map.get(max_severity, AlertLevel.WARNING)
        
        bias_types = [bias['type'] for bias in bias_report.get('biases', [])]
        
        self.send_alert(
            level=level,
            title="Bias Detected in Data/Model",
            message=f"Detected {len(bias_types)} types of bias: {', '.join(bias_types)}",
            details={
                'severity': max_severity,
                'bias_types': bias_types,
                'recommendation': 'Apply mitigation techniques or review data'
            }
        )
    
    def alert_on_adversarial_attack(self, attack_result: Dict):
        """
        Gửi alert khi model bị attack
        
        Args:
            attack_result: Kết quả từ adversarial attack test
        """
        success_rate = attack_result.get('success_rate', 0)
        
        if success_rate < 0.1:  # Model robust
            return
        
        level = AlertLevel.WARNING if success_rate < 0.5 else AlertLevel.CRITICAL
        
        self.send_alert(
            level=level,
            title="Adversarial Attack Vulnerability",
            message=f"Model vulnerable to adversarial attacks (success rate: {success_rate:.1%})",
            details={
                'success_rate': f"{success_rate:.1%}",
                'attack_type': attack_result.get('attack_type', 'unknown'),
                'recommendation': 'Apply adversarial training or defensive distillation'
            }
        )
    
    def get_alert_summary(self) -> str:
        """
        Lấy summary về alerts
        
        Returns:
            Summary string
        """
        if not self.alert_history:
            return "No alerts generated yet"
        
        report = []
        report.append("=" * 60)
        report.append("ALERT SUMMARY")
        report.append("=" * 60)
        
        report.append(f"\nTotal alerts: {len(self.alert_history)}")
        
        # Count by level
        from collections import Counter
        levels = [a['level'] for a in self.alert_history]
        level_counts = Counter(levels)
        
        report.append("\nBy level:")
        for level in ['info', 'warning', 'error', 'critical']:
            count = level_counts.get(level, 0)
            if count > 0:
                report.append(f"  • {level.upper()}: {count}")
        
        # Recent alerts
        report.append("\nRecent alerts (last 5):")
        for alert in self.alert_history[-5:]:
            report.append(f"  [{alert['level'].upper()}] {alert['title']} - {alert['timestamp']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def clear_history(self):
        """Xóa alert history"""
        self.alert_history.clear()
    
    def __repr__(self) -> str:
        return f"AlertManager(alerts={len(self.alert_history)}, channels={len(self.enabled_channels)})"

