"""
File-based Log Handler (implementation hiện tại)
"""

import json
import os
from typing import Dict, Any, List, Optional
from .base_handler import AbstractLogHandler


class FileLogHandler(AbstractLogHandler):
    """
    Handler ghi logs vào file
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.log_file = self.config.get('log_file', 'audit_log.jsonl')
        self.log_dir = self.config.get('log_dir', 'audit_logs')
        
        # Tạo thư mục nếu chưa có
        os.makedirs(self.log_dir, exist_ok=True)
        self.full_path = os.path.join(self.log_dir, self.log_file)
    
    def connect(self) -> bool:
        """File handler không cần connect"""
        self.is_connected = True
        return True
    
    def write_event(self, event: Dict[str, Any]) -> bool:
        """Ghi event vào file"""
        try:
            with open(self.full_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
            return True
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False
    
    def read_events(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Đọc events từ file"""
        events = []
        
        try:
            with open(self.full_path, 'r') as f:
                for line in f:
                    event = json.loads(line)
                    
                    # Apply filters
                    if filters:
                        if 'event_type' in filters and event.get('event_type') != filters['event_type']:
                            continue
                    
                    events.append(event)
            
            if limit:
                events = events[-limit:]
            
            return events
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading from file: {e}")
            return []
    
    def close(self):
        """File handler không cần close"""
        self.is_connected = False

