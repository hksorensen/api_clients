"""
Modern file-based caching system for API data.

Replaces MySQL with local file storage for single-user scenarios.
Much simpler, faster, and more portable than database storage.

Features:
- Automatic directory management
- Efficient pickle storage with compression
- Metadata tracking (fetch time, query params)
- Easy inspection and debugging
- Optional cache expiration
"""

from pathlib import Path
import pickle
import gzip
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import hashlib
import logging

logger = logging.getLogger(__name__)


class LocalCache:
    """
    Simple file-based cache for API responses.
    
    Each cached item is stored as a compressed pickle file with metadata.
    Much faster and simpler than MySQL for single-user scenarios.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = "~/.cache/scopus_data",
        compression: bool = True,
        max_age_days: Optional[int] = None,
    ):
        """
        Initialize local cache.
        
        Args:
            cache_dir: Directory for cache files
            compression: Use gzip compression (recommended)
            max_age_days: Expire cache entries after this many days (None = never expire)
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.max_age_days = max_age_days
        
        # Metadata file tracks all cached queries
        self.metadata_file = self.cache_dir / "_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata about cached items."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_cache_key(self, query: str) -> str:
        """
        Generate a safe filename from query string.
        
        Uses hash to handle long/special characters.
        """
        # Create a hash for the filename
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        
        # Also create a human-readable prefix
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
        
        return f"{safe_query}_{query_hash}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for a cache file."""
        ext = ".pkl.gz" if self.compression else ".pkl"
        return self.cache_dir / f"{cache_key}{ext}"
    
    def has(self, query: str) -> bool:
        """Check if query is in cache and not expired."""
        cache_key = self._get_cache_key(query)
        
        if cache_key not in self.metadata:
            return False
        
        # Check expiration
        if self.max_age_days is not None:
            cached_time = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
            age = datetime.now() - cached_time
            if age > timedelta(days=self.max_age_days):
                logger.info(f"Cache expired for query: {query[:50]}")
                return False
        
        # Check if file exists
        cache_path = self._get_cache_path(cache_key)
        return cache_path.exists()
    
    def get(self, query: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data for a query.
        
        Returns:
            DataFrame if found, None if not in cache
        """
        if not self.has(query):
            return None
        
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            if self.compression:
                with gzip.open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.info(f"Cache hit for query: {query[:50]}")
            return data
        
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def store(self, query: str, data: pd.DataFrame, **meta_kwargs):
        """
        Store data in cache.
        
        Args:
            query: Query string (used as key)
            data: DataFrame to cache
            **meta_kwargs: Additional metadata to store
        """
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Save data
            if self.compression:
                with gzip.open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'num_rows': len(data),
                'cache_key': cache_key,
                **meta_kwargs
            }
            self._save_metadata()
            
            logger.info(f"Cached {len(data)} rows for query: {query[:50]}")
        
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def delete(self, query: str):
        """Remove a query from cache."""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            cache_path.unlink()
        
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
    
    def list_queries(self) -> List[Dict[str, Any]]:
        """List all cached queries with metadata."""
        return list(self.metadata.values())
    
    def clear_expired(self):
        """Remove expired cache entries."""
        if self.max_age_days is None:
            return
        
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        expired = []
        
        for cache_key, meta in self.metadata.items():
            cached_time = datetime.fromisoformat(meta['timestamp'])
            if cached_time < cutoff:
                expired.append(meta['query'])
        
        for query in expired:
            self.delete(query)
        
        logger.info(f"Cleared {len(expired)} expired cache entries")
    
    def clear_all(self):
        """Remove all cached data."""
        for cache_key in list(self.metadata.keys()):
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
        
        self.metadata = {}
        self._save_metadata()
        logger.info("Cleared all cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            self._get_cache_path(key).stat().st_size 
            for key in self.metadata.keys()
            if self._get_cache_path(key).exists()
        )
        
        return {
            'num_entries': len(self.metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
        }


class MultiQueryCache:
    """
    Cache that handles multiple queries and aggregates results.
    
    Useful when you want to fetch abstracts for many EIDs,
    or run multiple related searches.
    """
    
    def __init__(self, cache_dir: Union[str, Path] = "~/.cache/scopus_data"):
        self.cache = LocalCache(cache_dir)
        self.batch_dir = self.cache.cache_dir / "batches"
        self.batch_dir.mkdir(exist_ok=True)
    
    def store_batch(self, batch_name: str, queries: List[str], results: Dict[str, pd.DataFrame]):
        """
        Store results from multiple queries as a batch.
        
        Args:
            batch_name: Name for this batch of queries
            queries: List of query strings
            results: Dict mapping query to results DataFrame
        """
        batch_file = self.batch_dir / f"{batch_name}.pkl.gz"
        
        batch_data = {
            'batch_name': batch_name,
            'timestamp': datetime.now().isoformat(),
            'queries': queries,
            'results': results,
        }
        
        with gzip.open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        
        logger.info(f"Stored batch '{batch_name}' with {len(queries)} queries")
    
    def get_batch(self, batch_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored batch."""
        batch_file = self.batch_dir / f"{batch_name}.pkl.gz"
        
        if not batch_file.exists():
            return None
        
        with gzip.open(batch_file, 'rb') as f:
            return pickle.load(f)
    
    def list_batches(self) -> List[str]:
        """List all stored batches."""
        return [f.stem for f in self.batch_dir.glob("*.pkl.gz")]


# Example usage and comparison with MySQL
if __name__ == "__main__":
    import time
    
    # Create some test data
    test_query = "TITLE-ABS-KEY(test)"
    test_data = pd.DataFrame({
        'ID': ['test1', 'test2', 'test3'],
        'data': [{'key': 'value1'}, {'key': 'value2'}, {'key': 'value3'}]
    })
    
    # Test local cache
    cache = LocalCache(cache_dir="/tmp/test_cache")
    
    # Store
    start = time.time()
    cache.store(test_query, test_data)
    store_time = time.time() - start
    
    # Retrieve
    start = time.time()
    result = cache.get(test_query)
    retrieve_time = time.time() - start
    
    print(f"Store: {store_time*1000:.2f}ms")
    print(f"Retrieve: {retrieve_time*1000:.2f}ms")
    print(f"Match: {result.equals(test_data)}")
    
    # Show stats
    stats = cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Entries: {stats['num_entries']}")
    print(f"  Size: {stats['total_size_mb']:.2f} MB")
    
    # List queries
    print(f"\nCached queries:")
    for item in cache.list_queries():
        print(f"  - {item['query'][:50]} ({item['num_rows']} rows)")
    
    # Clean up
    cache.clear_all()
