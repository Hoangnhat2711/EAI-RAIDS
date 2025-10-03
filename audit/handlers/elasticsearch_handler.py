"""
Elasticsearch Handler cho real-time analytics và visualization
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_handler import AbstractLogHandler


class ElasticsearchHandler(AbstractLogHandler):
    """
    Handler lưu audit logs vào Elasticsearch
    
    Ưu điểm:
    - Real-time search và analytics
    - Dễ dàng tích hợp với Kibana cho visualization
    - Scalable cho lượng logs lớn
    
    Yêu cầu: pip install elasticsearch
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        
        # Elasticsearch config
        self.hosts = self.config.get('hosts', ['http://localhost:9200'])
        self.index_name = self.config.get('index_name', 'responsible-ai-audit')
        self.api_key = self.config.get('api_key')
        self.cloud_id = self.config.get('cloud_id')
    
    def connect(self) -> bool:
        """Kết nối đến Elasticsearch"""
        try:
            from elasticsearch import Elasticsearch
            
            # Kết nối với credentials
            if self.cloud_id and self.api_key:
                self.client = Elasticsearch(
                    cloud_id=self.cloud_id,
                    api_key=self.api_key
                )
            else:
                self.client = Elasticsearch(self.hosts)
            
            # Kiểm tra kết nối
            if self.client.ping():
                self.is_connected = True
                print(f"✓ Connected to Elasticsearch: {self.index_name}")
                
                # Tạo index mapping nếu chưa có
                self._create_index_if_not_exists()
                return True
            else:
                print("✗ Cannot ping Elasticsearch")
                return False
        
        except ImportError:
            print("⚠ elasticsearch not installed. Install: pip install elasticsearch")
            return False
        except Exception as e:
            print(f"✗ Failed to connect to Elasticsearch: {e}")
            return False
    
    def _create_index_if_not_exists(self):
        """Tạo index với mapping phù hợp"""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "session_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "event_type": {"type": "keyword"},
                        "details": {"type": "object", "enabled": True},
                        "user_id": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": True}
                    }
                }
            }
            
            try:
                self.client.indices.create(index=self.index_name, body=mapping)
                print(f"✓ Created Elasticsearch index: {self.index_name}")
            except Exception as e:
                print(f"Warning: Index creation issue: {e}")
    
    def write_event(self, event: Dict[str, Any]) -> bool:
        """Ghi event vào Elasticsearch"""
        if not self.is_connected:
            return False
        
        try:
            # Index document
            response = self.client.index(
                index=self.index_name,
                document=event
            )
            
            return response['result'] in ['created', 'updated']
        
        except Exception as e:
            print(f"Error writing to Elasticsearch: {e}")
            return False
    
    def read_events(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Đọc events từ Elasticsearch"""
        if not self.is_connected:
            return []
        
        try:
            # Build query
            query = {"match_all": {}}
            
            if filters:
                must_clauses = []
                
                if 'event_type' in filters:
                    must_clauses.append({
                        "term": {"event_type": filters['event_type']}
                    })
                
                if 'session_id' in filters:
                    must_clauses.append({
                        "term": {"session_id": filters['session_id']}
                    })
                
                if 'start_time' in filters or 'end_time' in filters:
                    range_query = {"range": {"timestamp": {}}}
                    if 'start_time' in filters:
                        range_query['range']['timestamp']['gte'] = filters['start_time']
                    if 'end_time' in filters:
                        range_query['range']['timestamp']['lte'] = filters['end_time']
                    must_clauses.append(range_query)
                
                if must_clauses:
                    query = {"bool": {"must": must_clauses}}
            
            # Search
            response = self.client.search(
                index=self.index_name,
                query=query,
                size=limit or 100,
                sort=[{"timestamp": {"order": "desc"}}]
            )
            
            # Extract events
            events = []
            for hit in response['hits']['hits']:
                event = hit['_source']
                event['_id'] = hit['_id']
                events.append(event)
            
            return events
        
        except Exception as e:
            print(f"Error reading from Elasticsearch: {e}")
            return []
    
    def search_with_aggregation(self, agg_field: str = 'event_type') -> Dict[str, int]:
        """
        Thực hiện aggregation query
        
        Args:
            agg_field: Field để aggregate
        
        Returns:
            Dictionary với aggregation results
        """
        if not self.is_connected:
            return {}
        
        try:
            response = self.client.search(
                index=self.index_name,
                size=0,
                aggs={
                    f"{agg_field}_count": {
                        "terms": {"field": agg_field}
                    }
                }
            )
            
            results = {}
            for bucket in response['aggregations'][f'{agg_field}_count']['buckets']:
                results[bucket['key']] = bucket['doc_count']
            
            return results
        
        except Exception as e:
            print(f"Error in aggregation: {e}")
            return {}
    
    def close(self):
        """Đóng kết nối Elasticsearch"""
        if self.client:
            self.client.close()
        self.is_connected = False
        print("✓ Elasticsearch connection closed")

