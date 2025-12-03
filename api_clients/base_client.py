"""
Base API client with rate limiting, caching, and error handling.

This base class provides common functionality for API clients:
- Proactive rate limiting with token bucket algorithm
- Local file-based caching
- Exponential backoff for retries
- Comprehensive error handling
- Cursor-based pagination support
"""

from pathlib import Path
import requests
import pandas as pd
import time
import datetime
from urllib.parse import quote_plus
from typing import Optional, Dict, List, Any, Iterator
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .local_cache import LocalCache

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Base configuration for API clients."""
    # Rate limiting
    requests_per_second: float = 2.0
    burst_size: int = 5
    
    # Retry configuration
    max_retries: int = 3
    initial_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    retry_backoff_factor: float = 2.0
    
    # Request configuration
    timeout: int = 30
    max_results_per_query: int = 5000
    
    # Default parameters (API-specific)
    default_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.default_params is None:
            self.default_params = {}


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens. Returns True if successful."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on time passed
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Calculate time to wait before tokens are available."""
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.rate


class RateLimiter:
    """Rate limiter that respects both local token bucket and API rate limit headers."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.bucket = TokenBucket(config.requests_per_second, config.burst_size)
        self.api_rate_limit: Optional[int] = None
        self.api_remaining: Optional[int] = None
        self.api_reset_time: Optional[datetime.datetime] = None
    
    def wait_if_needed(self):
        """Wait if rate limiting requires it."""
        # First check API rate limits if we have them
        if self.api_remaining is not None and self.api_remaining < 10:
            if self.api_reset_time:
                wait_time = (self.api_reset_time - datetime.datetime.now()).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Approaching API rate limit. Waiting {wait_time:.1f}s until reset.")
                    time.sleep(wait_time + 1)
                    return
        
        # Then check local token bucket
        wait_time = self.bucket.wait_time()
        if wait_time > 0:
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        # Consume token
        while not self.bucket.consume():
            time.sleep(0.1)
    
    def update_from_headers(self, headers: Dict[str, str]):
        """Update rate limit state from API response headers (API-specific)."""
        # Default implementation - can be overridden by subclasses
        pass


class BaseAPIClient(ABC):
    """
    Base class for API clients with rate limiting, caching, and error handling.
    
    Subclasses must implement:
    - _build_search_url(): Construct search URL
    - _parse_page_response(): Parse a page of results
    - _get_next_page_url(): Get URL for next page (if pagination supported)
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.session = requests.Session()
        self._setup_session()
    
    @abstractmethod
    def _setup_session(self):
        """Setup session headers and authentication. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _build_search_url(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build URL for search query. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _parse_page_response(self, response_data: Dict[str, Any], page: int) -> Dict[str, Any]:
        """
        Parse API response into standardized format.
        
        Must return dict with keys:
            - page: page number
            - total_results: total number of results
            - results: list of result entries
            - cursor: optional cursor for this page
            - error: optional error code
        """
        pass
    
    @abstractmethod
    def _get_next_page_url(self, response_data: Dict[str, Any], current_url: str) -> Optional[str]:
        """Get URL for next page from response, or None if no more pages."""
        pass
    
    def _make_request(self, url: str, retry_count: int = 0) -> Optional[requests.Response]:
        """
        Make a request with retry logic.
        
        Returns None if request fails after all retries.
        """
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            self.rate_limiter.update_from_headers(response.headers)
            
            if response.ok:
                return response
            
            # Handle different error codes
            if response.status_code == 429:  # Too Many Requests
                logger.warning("Rate limit exceeded (429). Waiting before retry...")
                time.sleep(self.config.max_retry_delay)
                return self._retry_request(url, retry_count)
            
            elif response.status_code == 500:  # Server Error
                logger.warning(f"Server error (500). Retry {retry_count + 1}/{self.config.max_retries}")
                return self._retry_request(url, retry_count)
            
            elif response.status_code == 400:  # Bad Request
                logger.error(f"Bad request (400): {response.text[:200]}")
                return None
            
            elif response.status_code == 401:  # Unauthorized
                logger.error("Authentication failed (401). Check credentials.")
                raise RuntimeError("Invalid credentials or unauthorized access")
            
            else:
                logger.error(f"Unexpected status code {response.status_code}")
                return self._retry_request(url, retry_count)
        
        except requests.Timeout:
            logger.warning(f"Request timeout. Retry {retry_count + 1}/{self.config.max_retries}")
            return self._retry_request(url, retry_count)
        
        except requests.RequestException as e:
            logger.error(f"Request exception: {e}")
            return self._retry_request(url, retry_count)
    
    def _retry_request(self, url: str, retry_count: int) -> Optional[requests.Response]:
        """Handle retry logic with exponential backoff."""
        if retry_count >= self.config.max_retries:
            logger.error(f"Max retries ({self.config.max_retries}) reached")
            return None
        
        delay = min(
            self.config.initial_retry_delay * (self.config.retry_backoff_factor ** retry_count),
            self.config.max_retry_delay
        )
        logger.info(f"Retrying in {delay:.1f}s...")
        time.sleep(delay)
        
        return self._make_request(url, retry_count + 1)
    
    def search_iter(self, query: str, params: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """
        Execute a search query and yield pages of results.
        
        Args:
            query: Search query string
            params: Optional parameters to override defaults
        
        Yields:
            Dict containing page data (format defined by _parse_page_response)
        """
        # Determine how many rows are requested (for checking if we should enforce limit)
        requested_rows = None
        if params:
            requested_rows = params.get('rows') or params.get('count')
        
        url = self._build_search_url(query, params)
        page = 0
        
        while url:
            page += 1
            logger.info(f"Fetching page {page} for query: {query[:50]}...")
            
            response = self._make_request(url)
            if response is None:
                logger.error(f"Failed to fetch page {page}")
                break
            
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Invalid JSON response: {e}")
                break
            
            # Parse using API-specific parser
            page_data = self._parse_page_response(data, page)
            
            # Check for errors
            if page_data.get('error'):
                logger.error(f"API error: {page_data.get('error')}")
                yield page_data
                break
            
            # Check if query exceeds max results
            # Skip this check if we're only requesting a small number of rows (e.g., <= 10)
            # This is useful for citation resolution where we only need the top result
            total_results = page_data.get('total_results', 0)
            should_check_limit = requested_rows is None or requested_rows > 10
            
            if should_check_limit and total_results > self.config.max_results_per_query:
                logger.warning(
                    f"Query returned {total_results} results, exceeding max_results_per_query "
                    f"({self.config.max_results_per_query}). Consider refining your query."
                )
                page_data['error'] = 'too_many_results'
                yield page_data
                break
            
            # Stop iterating if there are no items/results on this page
            items = page_data.get('results')
            if items is None:
                items = page_data.get('items')
            if isinstance(items, list) and len(items) == 0:
                logger.info(f"No items on page {page}; stopping pagination.")
                break
            
            yield page_data
            
            # Get next page URL
            url = self._get_next_page_url(data, url)
    
    def search(self, query: str, params: Optional[Dict[str, Any]] = None, ignore_total_limit: bool = False) -> List[Dict[str, Any]]:
        """
        Execute a search query and return all results.
        
        Args:
            query: Search query string
            params: Optional parameters to override defaults
            ignore_total_limit: If True, don't raise error for too many total results.
                               Useful when you only need a few top results (e.g., citation resolution).
                               Default: False
        
        Returns:
            List of all result entries
        """
        all_results = []
        total_results = None
        
        for page_data in self.search_iter(query, params):
            if page_data.get('error') == 'too_many_results':
                if ignore_total_limit:
                    # Just log a warning but continue with the results we have
                    logger.warning(
                        f"Query returned {page_data['total_results']} total results, "
                        f"but ignoring limit since ignore_total_limit=True"
                    )
                    # Still yield the page data if it has results
                    if page_data.get('results'):
                        all_results.extend(page_data['results'])
                    break
                else:
                    raise ValueError(
                        f"Query returned {page_data['total_results']} results, "
                        f"exceeding limit of {self.config.max_results_per_query}"
                    )
            
            if total_results is None:
                total_results = page_data['total_results']
                logger.info(f"Query will fetch {total_results} total results")
            
            if page_data['results']:
                all_results.extend(page_data['results'])
        
        logger.info(f"Fetched {len(all_results)} results")
        return all_results


class BaseSearchFetcher:
    """
    Base class for search fetchers with caching.
    
    Wraps an API client with caching functionality.
    """
    
    def __init__(self, client: BaseAPIClient, cache: LocalCache):
        self.client = client
        self.cache = cache
    
    def fetch(self, query: str, force_refresh: bool = False, show_progress: bool = True, **params) -> Optional[pd.DataFrame]:
        """
        Fetch results for a query, using cache if available.
        
        Args:
            query: Query string
            force_refresh: If True, bypass cache
            show_progress: If True, show progress bar (default: True)
            **params: Additional API-specific parameters to pass to search
        
        Returns:
            DataFrame with columns: ID, page, num_hits, data
        """
        # Build cache key that includes params
        cache_key = query
        if params:
            # Sort params for consistent cache keys
            param_str = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
            cache_key = f"{query}|{param_str}"
        
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for query: {cache_key[:50]}")
                return cached
        
        logger.info(f"Fetching query: {cache_key[:50]}...")
        
        # Try to import tqdm for progress bar
        pbar = None
        if show_progress:
            try:
                from tqdm.auto import tqdm
                tqdm_available = True
            except ImportError:
                tqdm_available = False
                logger.debug("tqdm not available - install with: pip install tqdm")
        else:
            tqdm_available = False
        
        # Fetch from API
        pages = []
        total_results = None
        
        try:
            for page_data in self.client.search_iter(query, params if params else None):
                # Initialize progress bar on first page (when we know total)
                if pbar is None and tqdm_available and page_data.get('total_results'):
                    total_results = page_data['total_results']
                    # Estimate number of pages based on typical page size
                    estimated_pages = min(
                        (total_results // 100) + 1,  # Rough estimate
                        (total_results // 25) + 1    # More conservative
                    )
                    pbar = tqdm(
                        total=total_results,
                        desc=f"Fetching {cache_key[:30]}",
                        unit="results",
                        leave=True
                    )
                
                pages.append({
                    'ID': cache_key,
                    'page': page_data['page'],
                    'num_hits': page_data['total_results'],
                    'data': page_data.get('results'),
                    'error': page_data.get('error')
                })
                
                # Update progress bar
                if pbar is not None and page_data.get('results'):
                    pbar.update(len(page_data['results']))
                
                # Handle errors
                if page_data.get('error'):
                    if page_data.get('error') == 'too_many_results':
                        logger.warning(
                            f"Query returned {page_data['total_results']} results, "
                            f"exceeding max_hits ({self.client.config.max_results_per_query})"
                        )
                        pages[-1]['data'] = None
                    break
        
        except Exception as e:
            logger.error(f"Error fetching query: {e}")
            return None
        
        finally:
            # Close progress bar
            if pbar is not None:
                pbar.close()
        
        if not pages:
            logger.warning(f"No results for query: {cache_key[:50]}")
            return None
        
        # Store in cache
        df = pd.DataFrame(pages)
        self.cache.store(
            cache_key,
            df,
            total_results=pages[0]['num_hits'],
            num_pages=len(pages)
        )
        
        return df
    
    def provide(self, queries: List[str], force_refresh: bool = False, show_progress: bool = True) -> pd.DataFrame:
        """
        Fetch multiple queries, using cache when possible.
        
        Args:
            queries: List of query strings
            force_refresh: If True, bypass cache for all queries
            show_progress: If True, show progress bar (default: True)
        
        Returns:
            DataFrame with all results concatenated
        """
        # Try to import tqdm for progress bar
        if show_progress:
            try:
                from tqdm.auto import tqdm
                tqdm_available = True
            except ImportError:
                tqdm_available = False
                logger.debug("tqdm not available - install with: pip install tqdm")
        else:
            tqdm_available = False
        
        results = []
        
        # Determine which queries need fetching
        if force_refresh:
            queries_to_fetch = queries
        else:
            cached_queries = set(self.get_ID_list())
            queries_to_fetch = [q for q in queries if q not in cached_queries]
        
        # Fetch missing queries with progress bar
        if queries_to_fetch:
            logger.info(f"Fetching {len(queries_to_fetch)} queries...")
            
            query_iter = tqdm(queries_to_fetch, desc="Fetching queries", unit="query") if tqdm_available else queries_to_fetch
            
            for query in query_iter:
                # Disable individual progress bars when doing batch
                result = self.fetch(query, force_refresh=force_refresh, show_progress=False)
                if result is not None:
                    results.append(result)
        
        # Get all requested queries from cache
        for query in queries:
            cached = self.cache.get(query)
            if cached is not None and not any((cached is r).all().all() for r in results if not results):
                results.append(cached)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(results, ignore_index=True)
    
    def get(self, query: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get cached data for a query (or all cached data)."""
        if query is not None:
            return self.cache.get(query)
        
        # Return all cached data
        all_data = []
        for item in self.cache.list_queries():
            data = self.cache.get(item['query'])
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            return None
        
        return pd.concat(all_data, ignore_index=True)
    
    def get_ID_list(self) -> List[str]:
        """Get list of all cached query IDs."""
        return [item['query'] for item in self.cache.list_queries()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear_all()

    def expand(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Expand paged results into single DataFrame of items.
        
        Takes a DataFrame returned by fetch() (with 'data' column containing
        lists of items per page) and returns a DataFrame with one row per item.
        
        Args:
            results: DataFrame from fetch() with columns ['ID', 'page', 'num_hits', 'data', 'error']
            
        Returns:
            DataFrame with one row per item from all pages
            
        Example:
            >>> results = fetcher.fetch("machine learning")
            >>> results.shape
            (10, 5)  # 10 pages
            >>> 
            >>> items = fetcher.expand(results)
            >>> items.shape
            (1000, 50)  # 1000 items with ~50 columns
            >>> 
            >>> # Access individual items
            >>> print(items['DOI'].tolist())
            >>> print(items[['title', 'author']].head())
            
        Note:
            - Automatically filters out None and empty data
            - Preserves order from pagination
            - Returns empty DataFrame if no items found
            - Works with any API (Crossref, Scopus, etc.)
        """
        all_items = []
        
        # Extract all items from all pages
        for idx, row in results.iterrows():
            data = row.get('data')
            
            # Skip if no data or empty
            if data is None:
                continue
            
            if isinstance(data, list):
                if len(data) > 0:
                    all_items.extend(data)
            else:
                # Single item (edge case)
                all_items.append(data)
        
        # Return empty DataFrame if no items
        if not all_items:
            return pd.DataFrame()
        
        # Convert to DataFrame
        return pd.DataFrame(all_items)
