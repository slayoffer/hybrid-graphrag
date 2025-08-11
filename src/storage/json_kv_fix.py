"""Fixed JSON KV storage implementation with proper async locks.

This module provides a corrected version of the JSON KV storage
that properly handles async locks to avoid AttributeError: __aenter__ issues.
"""

import json
import os
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class JsonKVStorageFixed:
    """Fixed JSON key-value storage with proper async lock handling."""
    
    def __init__(self, storage_file: str):
        """Initialize JSON KV storage with proper async lock.
        
        Args:
            storage_file: Path to JSON storage file
        """
        self.storage_file = Path(storage_file)
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use asyncio.Lock() for proper async context manager
        self._storage_lock = asyncio.Lock()
        
        # Initialize data
        self.data = {}
        self._load_data()
        
        logger.debug(f"JsonKVStorageFixed initialized: {storage_file}")
    
    def _load_data(self):
        """Load data from JSON file."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    self.data = json.load(f)
                logger.debug(f"Loaded {len(self.data)} entries from {self.storage_file}")
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                self.data = {}
        else:
            self.data = {}
    
    def _save_data(self):
        """Save data to JSON file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.debug(f"Saved {len(self.data)} entries to {self.storage_file}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    async def get_by_id(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value by key with proper async lock.
        
        Args:
            key: The key to retrieve
            
        Returns:
            Value associated with the key, or None if not found
        """
        async with self._storage_lock:
            return self.data.get(key)
    
    async def get_by_ids(self, keys: list[str]) -> Dict[str, Any]:
        """Get multiple values by keys.
        
        Args:
            keys: List of keys to retrieve
            
        Returns:
            Dictionary of key-value pairs
        """
        async with self._storage_lock:
            return {k: self.data.get(k) for k in keys if k in self.data}
    
    async def filter_keys(self, prefix: str = "") -> list[str]:
        """Filter keys by prefix.
        
        Args:
            prefix: Prefix to filter by
            
        Returns:
            List of matching keys
        """
        async with self._storage_lock:
            if prefix:
                return [k for k in self.data.keys() if k.startswith(prefix)]
            return list(self.data.keys())
    
    async def upsert(self, data: Dict[str, Any]) -> None:
        """Insert or update data.
        
        Args:
            data: Dictionary of key-value pairs to upsert
        """
        async with self._storage_lock:
            self.data.update(data)
            self._save_data()
    
    async def delete_by_id(self, key: str) -> bool:
        """Delete entry by key.
        
        Args:
            key: Key to delete
            
        Returns:
            True if deleted, False if not found
        """
        async with self._storage_lock:
            if key in self.data:
                del self.data[key]
                self._save_data()
                return True
            return False
    
    async def delete_by_ids(self, keys: list[str]) -> int:
        """Delete multiple entries by keys.
        
        Args:
            keys: List of keys to delete
            
        Returns:
            Number of entries deleted
        """
        async with self._storage_lock:
            deleted = 0
            for key in keys:
                if key in self.data:
                    del self.data[key]
                    deleted += 1
            if deleted > 0:
                self._save_data()
            return deleted
    
    async def drop(self) -> None:
        """Clear all data."""
        async with self._storage_lock:
            self.data = {}
            self._save_data()
    
    async def index_done_callback(self) -> None:
        """Callback after indexing is complete."""
        async with self._storage_lock:
            self._save_data()
            logger.debug("Index done callback - data persisted")


def patch_lightrag_json_kv():
    """Patch LightRAG's JSON KV implementation with the fixed version.
    
    This function can be called to replace the problematic JSON KV storage
    with our fixed implementation at runtime.
    """
    try:
        import lightrag.kg.json_kv_impl as json_kv_module
        
        # Replace the problematic class with our fixed version
        original_class = json_kv_module.JsonKVStorage
        
        class PatchedJsonKVStorage(JsonKVStorageFixed):
            """Patched version that matches LightRAG's interface."""
            
            def __init__(self, namespace: str, global_config: Dict[str, Any], 
                        embedding_func=None, meta_fields=None):
                """Initialize with LightRAG's expected interface."""
                working_dir = global_config.get("working_dir", "./rag_storage")
                workspace = global_config.get("workspace", "")
                
                if workspace:
                    storage_file = os.path.join(working_dir, workspace, f"kv_{namespace}.json")
                else:
                    storage_file = os.path.join(working_dir, f"kv_{namespace}.json")
                
                super().__init__(storage_file)
                
                self.namespace = namespace
                self.global_config = global_config
                self.embedding_func = embedding_func
                self.meta_fields = meta_fields or []
        
        # Replace the class
        json_kv_module.JsonKVStorage = PatchedJsonKVStorage
        logger.info("Successfully patched JsonKVStorage with fixed async lock implementation")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch JsonKVStorage: {e}")
        return False


async def test_fixed_storage():
    """Test the fixed JSON KV storage."""
    storage = JsonKVStorageFixed("./test_storage.json")
    
    # Test upsert
    await storage.upsert({
        "key1": {"data": "value1"},
        "key2": {"data": "value2"}
    })
    
    # Test get
    value = await storage.get_by_id("key1")
    assert value == {"data": "value1"}
    
    # Test filter
    keys = await storage.filter_keys("key")
    assert len(keys) == 2
    
    # Test delete
    deleted = await storage.delete_by_id("key1")
    assert deleted == True
    
    # Clean up
    await storage.drop()
    
    logger.info("All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_fixed_storage())