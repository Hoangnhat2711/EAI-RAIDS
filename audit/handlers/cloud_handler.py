"""
Cloud Logger Handler cho AWS CloudWatch, Azure Monitor, GCP Cloud Logging
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_handler import AbstractLogHandler


class CloudLoggerHandler(AbstractLogHandler):
    """
    Handler để ghi logs vào các cloud logging services
    
    Hỗ trợ:
    - AWS CloudWatch
    - Azure Monitor
    - GCP Cloud Logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        
        # Cloud provider config
        self.provider = self.config.get('provider', 'aws')  # 'aws', 'azure', 'gcp'
        self.log_group = self.config.get('log_group', 'responsible-ai-audit')
        self.log_stream = self.config.get('log_stream', 'default-stream')
    
    def connect(self) -> bool:
        """Kết nối đến cloud logging service"""
        if self.provider == 'aws':
            return self._connect_aws()
        elif self.provider == 'azure':
            return self._connect_azure()
        elif self.provider == 'gcp':
            return self._connect_gcp()
        else:
            print(f"✗ Unsupported provider: {self.provider}")
            return False
    
    def _connect_aws(self) -> bool:
        """Kết nối AWS CloudWatch"""
        try:
            import boto3
            
            self.client = boto3.client('logs')
            
            # Tạo log group và stream nếu chưa có
            try:
                self.client.create_log_group(logGroupName=self.log_group)
            except self.client.exceptions.ResourceAlreadyExistsException:
                pass
            
            try:
                self.client.create_log_stream(
                    logGroupName=self.log_group,
                    logStreamName=self.log_stream
                )
            except self.client.exceptions.ResourceAlreadyExistsException:
                pass
            
            self.is_connected = True
            print(f"✓ Connected to AWS CloudWatch: {self.log_group}/{self.log_stream}")
            return True
        
        except ImportError:
            print("⚠ boto3 not installed. Install: pip install boto3")
            return False
        except Exception as e:
            print(f"✗ Failed to connect to AWS CloudWatch: {e}")
            return False
    
    def _connect_azure(self) -> bool:
        """Kết nối Azure Monitor"""
        try:
            from azure.monitor.ingestion import LogsIngestionClient
            from azure.identity import DefaultAzureCredential
            
            endpoint = self.config.get('endpoint')
            credential = DefaultAzureCredential()
            
            self.client = LogsIngestionClient(
                endpoint=endpoint,
                credential=credential
            )
            
            self.is_connected = True
            print(f"✓ Connected to Azure Monitor")
            return True
        
        except ImportError:
            print("⚠ azure-monitor-ingestion not installed. Install: pip install azure-monitor-ingestion azure-identity")
            return False
        except Exception as e:
            print(f"✗ Failed to connect to Azure Monitor: {e}")
            return False
    
    def _connect_gcp(self) -> bool:
        """Kết nối GCP Cloud Logging"""
        try:
            from google.cloud import logging
            
            self.client = logging.Client()
            self.logger = self.client.logger(self.log_group)
            
            self.is_connected = True
            print(f"✓ Connected to GCP Cloud Logging: {self.log_group}")
            return True
        
        except ImportError:
            print("⚠ google-cloud-logging not installed. Install: pip install google-cloud-logging")
            return False
        except Exception as e:
            print(f"✗ Failed to connect to GCP Cloud Logging: {e}")
            return False
    
    def write_event(self, event: Dict[str, Any]) -> bool:
        """Ghi event vào cloud logging"""
        if not self.is_connected:
            return False
        
        if self.provider == 'aws':
            return self._write_aws(event)
        elif self.provider == 'azure':
            return self._write_azure(event)
        elif self.provider == 'gcp':
            return self._write_gcp(event)
        
        return False
    
    def _write_aws(self, event: Dict[str, Any]) -> bool:
        """Ghi vào AWS CloudWatch"""
        try:
            import json
            
            self.client.put_log_events(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
                logEvents=[
                    {
                        'timestamp': int(datetime.now().timestamp() * 1000),
                        'message': json.dumps(event)
                    }
                ]
            )
            return True
        except Exception as e:
            print(f"Error writing to AWS CloudWatch: {e}")
            return False
    
    def _write_azure(self, event: Dict[str, Any]) -> bool:
        """Ghi vào Azure Monitor"""
        try:
            rule_id = self.config.get('rule_id')
            stream_name = self.config.get('stream_name')
            
            self.client.upload(
                rule_id=rule_id,
                stream_name=stream_name,
                logs=[event]
            )
            return True
        except Exception as e:
            print(f"Error writing to Azure Monitor: {e}")
            return False
    
    def _write_gcp(self, event: Dict[str, Any]) -> bool:
        """Ghi vào GCP Cloud Logging"""
        try:
            self.logger.log_struct(event)
            return True
        except Exception as e:
            print(f"Error writing to GCP Cloud Logging: {e}")
            return False
    
    def read_events(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Đọc events từ cloud logging
        
        Note: Implementation phụ thuộc vào từng provider
        """
        if not self.is_connected:
            return []
        
        # Simplified - mỗi provider có API riêng để query logs
        print(f"Read operation for {self.provider} - implement based on specific API")
        return []
    
    def close(self):
        """Đóng kết nối cloud logging"""
        self.client = None
        self.is_connected = False
        print(f"✓ {self.provider.upper()} logging connection closed")

