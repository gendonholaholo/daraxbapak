from typing import Dict, List, Optional
import json
import zlib
import logging
from datetime import datetime

class ContextManager:
    def __init__(self, max_size: int = 1000):
        self.context: Dict[str, List[Dict]] = {}
        self.max_size = max_size
        self.compression_threshold = 500  # Threshold for compression in bytes

    async def store_context(self, key: str, data: Dict) -> None:
        """Store context data with compression if needed."""
        try:
            if key not in self.context:
                self.context[key] = []
            
            # Add timestamp
            data['timestamp'] = datetime.now().isoformat()
            
            # Check if compression is needed
            serialized = json.dumps(data)
            if len(serialized) > self.compression_threshold:
                compressed = zlib.compress(serialized.encode())
                data = {
                    'compressed': True,
                    'data': compressed.decode('latin1'),
                    'timestamp': data['timestamp']
                }
            
            self.context[key].append(data)
            
            # Maintain max size
            if len(self.context[key]) > self.max_size:
                self.context[key] = self.context[key][-self.max_size:]
                
        except Exception as e:
            logging.error(f"Failed to store context: {str(e)}")
            raise

    async def retrieve_context(self, key: str, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve and decompress context data if needed."""
        try:
            if key not in self.context:
                return []
            
            context_data = self.context[key]
            if limit:
                context_data = context_data[-limit:]
            
            result = []
            for item in context_data:
                if item.get('compressed', False):
                    decompressed = zlib.decompress(item['data'].encode('latin1'))
                    data = json.loads(decompressed)
                    data['timestamp'] = item['timestamp']
                    result.append(data)
                else:
                    result.append(item)
            
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve context: {str(e)}")
            raise

    async def update_context(self, key: str, data: Dict) -> None:
        """Update the most recent context entry."""
        try:
            if key not in self.context or not self.context[key]:
                await self.store_context(key, data)
                return
            
            # Update the most recent entry
            self.context[key][-1].update(data)
        except Exception as e:
            logging.error(f"Failed to update context: {str(e)}")
            raise

    async def clear_context(self, key: Optional[str] = None) -> None:
        """Clear context for a specific key or all keys."""
        try:
            if key:
                if key in self.context:
                    del self.context[key]
            else:
                self.context.clear()
        except Exception as e:
            logging.error(f"Failed to clear context: {str(e)}")
            raise 