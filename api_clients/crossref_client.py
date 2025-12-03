"""
Crossref API client - specialization of base API client.

Handles Crossref-specific:
- Polite pool access (via mailto)
- Query formatting
- Response parsing  
- Cursor-based pagination
- Filter support
"""

from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from urllib.parse import quote_plus, urlencode
import logging
import pandas as pd
import yaml

from .base_client import BaseAPIClient, APIConfig, RateLimiter, BaseSearchFetcher
from .local_cache import LocalCache

logger = logging.getLogger(__name__)


@dataclass
class CrossrefConfig(APIConfig):
    """Crossref-specific configuration."""
    mailto: str = ""  # Email for polite pool access
    base_url: str = "https://api.crossref.org/works"
    
    # Crossref polite pool: ~50 requests/second (but be conservative)
    # Public pool: much lower
    requests_per_second: float = 10.0  # Conservative for polite pool
    burst_size: int = 20
    max_results_per_query: int = 10000
    
    # Crossref-specific defaults
    rows_per_page: int = 100  # Crossref uses 'rows' not 'count'
    
    def __post_init__(self):
        super().__post_init__()
        if not self.default_params:
            self.default_params = {'rows': self.rows_per_page}


class CrossrefRateLimiter(RateLimiter):
    """Crossref-specific rate limiter."""
    
    def update_from_headers(self, headers: Dict[str, str]):
        """Update rate limit state from Crossref API headers."""
        try:
            # Crossref uses X-Rate-Limit-Limit and X-Rate-Limit-Interval
            if 'X-Rate-Limit-Limit' in headers:
                self.api_rate_limit = int(headers['X-Rate-Limit-Limit'])
            if 'X-Rate-Limit-Interval' in headers:
                interval_seconds = int(headers['X-Rate-Limit-Interval'].rstrip('s'))
                # Calculate requests per second
                if interval_seconds > 0:
                    rate = self.api_rate_limit / interval_seconds
                    logger.debug(f"Crossref rate limit: {self.api_rate_limit} requests per {interval_seconds}s ({rate:.1f}/s)")
        except (ValueError, KeyError) as e:
            logger.warning(f"Error parsing Crossref rate limit headers: {e}")


class CrossrefSearchClient(BaseAPIClient):
    """Crossref search API client."""
    
    def __init__(self, config: CrossrefConfig):
        self.config = config
        self.rate_limiter = CrossrefRateLimiter(config)
        super().__init__(config)
    
    def _setup_session(self):
        """Setup Crossref session with polite pool headers."""
        user_agent = "api_clients/1.0"
        if self.config.mailto:
            user_agent += f" (mailto:{self.config.mailto})"
        
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json',
        })
        
        if self.config.mailto:
            logger.info(f"Crossref client using polite pool (mailto: {self.config.mailto})")
        else:
            logger.warning("No mailto provided - using public pool (slower rate limits)")
    
    def _build_search_url(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Build Crossref search URL.
        
        Query can be:
        - Simple text: "machine learning"
        - Field query: "query.title=machine learning"
        - Filter query: using filters parameter
        """
        url_params = {**self.config.default_params, **(params or {})}
        
        # Add query
        url_params['query'] = query
        
        # Add mailto if provided and not already in params
        if self.config.mailto and 'mailto' not in url_params:
            url_params['mailto'] = self.config.mailto
        
        # Add cursor=* for initial request to enable cursor-based pagination
        # The API requires cursor=* in the initial request to return next-cursor
        if 'cursor' not in url_params:
            url_params['cursor'] = '*'
        
        # Build URL
        param_str = urlencode(url_params)
        return f"{self.config.base_url}?{param_str}"
    
    def _parse_page_response(self, response_data: Dict[str, Any], page: int) -> Dict[str, Any]:
        """Parse Crossref API response."""
        # Check for errors
        if 'status' in response_data and response_data['status'] != 'ok':
            return {
                'page': page,
                'total_results': 0,
                'results': None,
                'error': response_data.get('message-type', 'unknown_error')
            }
        
        # Check for message
        if 'message' not in response_data:
            return {
                'page': page,
                'total_results': 0,
                'results': None,
                'error': 'no_message'
            }
        
        message = response_data['message']
        total_results = message.get('total-results', 0)
        items = message.get('items', [])
        
        # Get cursor for next page
        cursor = message.get('next-cursor', None)
        
        return {
            'page': page,
            'total_results': total_results,
            'results': items,
            'cursor': cursor,
        }
    
    def _get_next_page_url(self, response_data: Dict[str, Any], current_url: str) -> Optional[str]:
        """Get next page URL from Crossref response using cursor."""
        if 'message' not in response_data:
            return None
        
        message = response_data['message']
        next_cursor = message.get('next-cursor')
        items = message.get('items', [])
        
        if not items or len(items) == 0:
            return None
        
        if next_cursor:
            # Remove old cursor if present
            if 'cursor=' in current_url:
                # Split URL into base and params properly
                base, params = current_url.split('?', 1)
                # Remove cursor param
                param_parts = params.split('&')
                param_parts = [p for p in param_parts if not p.startswith('cursor=')]
                current_url = f"{base}?{'&'.join(param_parts)}"
            
            # Add new cursor
            return f"{current_url}&cursor={next_cursor}"
        
        return None


class CrossrefSearchFetcher(BaseSearchFetcher):
    """
    Crossref search fetcher with caching.
    
    Main interface for Crossref searches. Use this as the default.
    
    Usage:
        # Simple (auto-loads email from config)
        crossref = CrossrefSearchFetcher()
        
        # With explicit email
        crossref = CrossrefSearchFetcher(mailto="your@email.com")
    """
    
    def __init__(
        self,
        mailto: Optional[str] = None,
        cache_dir: str = "~/.cache/crossref/search",
        api_key_dir: str = "~/Documents/dh4pmp/api_keys",
        **kwargs
    ):
        """
        Initialize Crossref search fetcher.
        
        Args:
            mailto: Email address for polite pool access (if None, tries to load from yaml)
            cache_dir: Directory for cache files
            api_key_dir: Directory containing API config files (default: ~/Documents/dh4pmp/api_keys)
            **kwargs: Additional configuration:
                - requests_per_second: Rate limit (default: 10.0 for polite, 1.0 for public)
                - max_results_per_query: Max results (default: 10000)
                - max_retries: Retry attempts (default: 3)
                - cache_max_age_days: Cache expiration (default: None/never)
                - rows_per_page: Results per page (default: 100)
        """
        # Load email if not provided
        if mailto is None:
            mailto = self._load_email(api_key_dir)
        
        # Determine rate limit based on polite pool access
        if mailto:
            default_rate = 10.0  # Conservative for polite pool
        else:
            default_rate = 1.0  # Very conservative for public pool
            logger.warning("No email provided - using public pool (slower). Consider adding email to crossref.yaml")
        
        # Initialize configuration
        config = CrossrefConfig(
            mailto=mailto or "",
            requests_per_second=kwargs.get('requests_per_second', default_rate),
            max_results_per_query=kwargs.get('max_hits', kwargs.get('max_results_per_query', 10000)),
            max_retries=kwargs.get('max_retries', 3),
            rows_per_page=kwargs.get('rows_per_page', 100),
        )
        
        # Initialize client and cache
        client = CrossrefSearchClient(config)
        cache = LocalCache(
            cache_dir=cache_dir,
            compression=True,
            max_age_days=kwargs.get('cache_max_age_days', None)
        )
        
        super().__init__(client, cache)
        
        logger.info(f"Initialized Crossref fetcher with cache at {cache.cache_dir}")
    
    def _load_email(self, api_key_dir: str) -> Optional[str]:
        """
        Load email from YAML config file for polite pool access.
        
        Email is optional - Crossref works fine without it (just slower).
        File: crossref.yaml with optional 'mailto' or 'email' field
        """
        import yaml
        
        api_key_dir = Path(api_key_dir).expanduser()
        key_paths = [
            Path('.') / 'crossref.yaml',
            api_key_dir / 'crossref.yaml',
        ]
        
        for path in key_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        key_data = yaml.safe_load(f)
                        
                        # Handle empty file (None)
                        if key_data is None:
                            logger.info(f"Found {path} but it's empty - using public pool")
                            return None
                        
                        # Check for email fields
                        email = key_data.get('mailto') or key_data.get('email')
                        
                        if email and isinstance(email, str) and '@' in email:
                            logger.info(f"Loaded Crossref email from {path}")
                            return email
                        elif email is None or email == '':
                            # Explicitly set to None/empty - user wants public pool
                            logger.info(f"Found {path} with empty email - using public pool")
                            return None
                        
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
        
        # No config found - this is fine for Crossref!
        logger.info("No crossref.yaml found - using public pool (works fine, just slower)")
        return None
    
    def search_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata for a specific DOI.
        
        Args:
            doi: DOI string (e.g., "10.1371/journal.pone.0033693")
        
        Returns:
            Metadata dict or None if not found
        """
        import requests
        
        url = f"https://api.crossref.org/works/{doi}"
        
        # Add mailto if available
        if self.client.config.mailto:
            url += f"?mailto={self.client.config.mailto}"
        
        try:
            response = self.client._make_request(url)
            if response and response.ok:
                data = response.json()
                if 'message' in data:
                    return data['message']
        except Exception as e:
            logger.error(f"Error fetching DOI {doi}: {e}")
        
        return None
    
    def search_with_filters(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Search with Crossref filters.
        
        Args:
            query: Search query
            filters: Dict of filter names and values
                Examples:
                - {'has-abstract': True}
                - {'from-pub-date': '2020-01-01', 'until-pub-date': '2024-12-31'}
                - {'type': 'journal-article'}
            force_refresh: If True, bypass cache
            **kwargs: Additional search parameters
        
        Returns:
            DataFrame with results
        """
        import pandas as pd
        
        params = kwargs.copy()
        
        # Add filters
        if filters:
            # Crossref uses filter=key:value,key:value format
            filter_strs = []
            for key, value in filters.items():
                if isinstance(value, bool):
                    value = str(value).lower()
                filter_strs.append(f"{key}:{value}")
            
            if filter_strs:
                params['filter'] = ','.join(filter_strs)
        
        # Use inherited fetch method with params
        return self.fetch(query, force_refresh=force_refresh, **params)
    
    def fetch(self, query: str, force_refresh: bool = False, **params) -> Optional[pd.DataFrame]:
        """
        Fetch Crossref search results with optional parameters.
        
        Args:
            query: Search query (text or field query)
            force_refresh: If True, bypass cache
            **params: Additional Crossref API parameters
        
        Query Fields (use as 'query.field=value'):
            query.title           - Search in title
            query.author          - Search in author names
            query.affiliation     - Search in affiliations
            query.container-title - Search in journal/book name
            query.publisher-name  - Search in publisher name
            query.bibliographic   - Search all bibliographic fields
            query.editor          - Search in editor names
            
        Common Filters (pass as filter='key:value' or use search_with_filters):
            has-orcid            - Only works with ORCID
            has-abstract         - Only works with abstracts
            has-full-text        - Only works with full text
            has-references       - Only works with references
            has-funder           - Only works with funder info
            type                 - Document type (journal-article, book-chapter, etc.)
            from-pub-date        - Published on or after (YYYY-MM-DD)
            until-pub-date       - Published on or before (YYYY-MM-DD)
            from-online-pub-date - Online published on or after
            until-online-pub-date- Online published on or before
            issn                 - Journal ISSN
            isbn                 - Book ISBN
            publisher            - Publisher name
            funder               - Funder name
            
        Common Parameters:
            rows: int            - Results per page (default: 100, max: 1000)
            offset: int          - Starting index (for pagination)
            sort: str            - Sort field (score, published, updated, relevance)
            order: str           - Sort order (asc, desc)
            mailto: str          - Email for polite pool (set in config)
            filter: str          - Filter string (key:value,key:value)
            cursor: str          - Cursor for deep pagination (automatic)
            
        Document Types (for type filter):
            journal-article      - Journal articles
            book-chapter         - Book chapters
            monograph           - Monographs/books
            proceedings-article  - Conference papers
            report              - Reports
            dataset             - Datasets
            posted-content      - Preprints
            
        Examples:
            # Simple search
            results = fetcher.fetch("machine learning")
            
            # Field search
            results = fetcher.fetch("query.title=neural networks")
            results = fetcher.fetch("query.author=Smith")
            
            # With filters (use helper method)
            results = fetcher.search_with_filters(
                "deep learning",
                filters={'has-abstract': True, 'type': 'journal-article'}
            )
            
            # With parameters
            results = fetcher.fetch(
                "artificial intelligence",
                rows=200,
                sort="published",
                order="desc"
            )
            
            # Raw filter parameter
            results = fetcher.fetch(
                "climate change",
                filter="has-abstract:true,from-pub-date:2024-01-01",
                rows=100
            )
            
            # Multiple field search
            results = fetcher.fetch(
                "query.title=climate query.author=Smith"
            )
        
        Returns:
            DataFrame with columns: ID, page, num_hits, data, error
            
        Reference:
            https://api.crossref.org/swagger-ui/index.html
            https://github.com/CrossRef/rest-api-doc
        """
        # Use base fetch with params
        return super().fetch(query, force_refresh=force_refresh, **params)


# Convenience functions for common Crossref queries

def search_by_title(title: str, mailto: str, **kwargs) -> Optional[pd.DataFrame]:
    """Search Crossref by title."""
    fetcher = CrossrefSearchFetcher(mailto=mailto, **kwargs)
    return fetcher.fetch(f"query.title={title}")


def search_by_author(author: str, mailto: str, **kwargs) -> Optional[pd.DataFrame]:
    """Search Crossref by author name."""
    fetcher = CrossrefSearchFetcher(mailto=mailto, **kwargs)
    return fetcher.fetch(f"query.author={author}")


def search_journal_articles(query: str, mailto: str, **kwargs) -> Optional[pd.DataFrame]:
    """Search for journal articles only."""
    fetcher = CrossrefSearchFetcher(mailto=mailto, **kwargs)
    return fetcher.search_with_filters(query, filters={'type': 'journal-article'}, **kwargs)
