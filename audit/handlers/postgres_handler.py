"""
PostgreSQL Handler cho enterprise-grade audit logging
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_handler import AbstractLogHandler


class PostgreSQLHandler(AbstractLogHandler):
    """
    Handler lưu audit logs vào PostgreSQL database
    
    Yêu cầu: pip install psycopg2-binary
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.connection = None
        self.cursor = None
        
        # Database config
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 5432)
        self.database = self.config.get('database', 'responsible_ai')
        self.user = self.config.get('user', 'postgres')
        self.password = self.config.get('password', '')
        self.table_name = self.config.get('table_name', 'audit_logs')
    
    def connect(self) -> bool:
        """Kết nối đến PostgreSQL"""
        try:
            import psycopg2
            
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor()
            self.is_connected = True
            
            # Tạo table nếu chưa có
            self._create_table_if_not_exists()
            
            print(f"✓ Connected to PostgreSQL: {self.database}")
            return True
        
        except ImportError:
            print("⚠ psycopg2 not installed. Install: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"✗ Failed to connect to PostgreSQL: {e}")
            return False
    
    def _create_table_if_not_exists(self):
        """Tạo audit logs table"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255),
            timestamp TIMESTAMP,
            event_type VARCHAR(100),
            details JSONB,
            user_id VARCHAR(255),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_event_type ON {self.table_name}(event_type);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON {self.table_name}(timestamp);
        CREATE INDEX IF NOT EXISTS idx_session_id ON {self.table_name}(session_id);
        """
        
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
        except Exception as e:
            print(f"Warning: Table creation issue: {e}")
    
    def write_event(self, event: Dict[str, Any]) -> bool:
        """Ghi event vào PostgreSQL"""
        if not self.is_connected:
            return False
        
        try:
            insert_query = f"""
            INSERT INTO {self.table_name} 
            (session_id, timestamp, event_type, details, user_id, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            self.cursor.execute(insert_query, (
                event.get('session_id'),
                datetime.fromisoformat(event.get('timestamp')) if isinstance(event.get('timestamp'), str) else event.get('timestamp'),
                event.get('event_type'),
                json.dumps(event.get('details', {})),
                event.get('user_id'),
                json.dumps(event.get('metadata', {}))
            ))
            
            self.connection.commit()
            return True
        
        except Exception as e:
            print(f"Error writing to PostgreSQL: {e}")
            self.connection.rollback()
            return False
    
    def read_events(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Đọc events từ PostgreSQL"""
        if not self.is_connected:
            return []
        
        try:
            query = f"SELECT * FROM {self.table_name}"
            conditions = []
            params = []
            
            if filters:
                if 'event_type' in filters:
                    conditions.append("event_type = %s")
                    params.append(filters['event_type'])
                
                if 'session_id' in filters:
                    conditions.append("session_id = %s")
                    params.append(filters['session_id'])
                
                if 'start_time' in filters:
                    conditions.append("timestamp >= %s")
                    params.append(filters['start_time'])
                
                if 'end_time' in filters:
                    conditions.append("timestamp <= %s")
                    params.append(filters['end_time'])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            # Convert to dict
            events = []
            for row in rows:
                events.append({
                    'id': row[0],
                    'session_id': row[1],
                    'timestamp': row[2].isoformat() if row[2] else None,
                    'event_type': row[3],
                    'details': row[4],
                    'user_id': row[5],
                    'metadata': row[6]
                })
            
            return events
        
        except Exception as e:
            print(f"Error reading from PostgreSQL: {e}")
            return []
    
    def close(self):
        """Đóng kết nối PostgreSQL"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.is_connected = False
        print("✓ PostgreSQL connection closed")

