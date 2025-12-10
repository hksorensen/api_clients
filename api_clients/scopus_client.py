"""
Scopus API client - specialization of base API client.

Handles Scopus-specific:
- API key authentication
- Query string formatting
- Response parsing
- Cursor-based pagination
"""

from pathlib import Path
import yaml
import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from urllib.parse import quote_plus
import logging
import pandas as pd

from .base_client import BaseAPIClient, APIConfig, RateLimiter, BaseSearchFetcher
from .local_cache import LocalCache

logger = logging.getLogger(__name__)


@dataclass
class ScopusConfig(APIConfig):
    """Scopus-specific configuration."""
    api_key: str = ""
    base_url: str = "https://api.elsevier.com/content/search/scopus"
    view: str = "COMPLETE"
    
    # Scopus typically allows 2-3 requests per second for institutional access
    requests_per_second: float = 2.0
    burst_size: int = 5
    max_results_per_query: int = 5000
    
    def __post_init__(self):
        super().__post_init__()
        if not self.default_params:
            self.default_params = {'count': 25}


class ScopusRateLimiter(RateLimiter):
    """Scopus-specific rate limiter that reads X-RateLimit headers."""
    
    def update_from_headers(self, headers: Dict[str, str]):
        """Update rate limit state from Scopus API headers."""
        try:
            if 'X-RateLimit-Limit' in headers:
                self.api_rate_limit = int(headers['X-RateLimit-Limit'])
            if 'X-RateLimit-Remaining' in headers:
                self.api_remaining = int(headers['X-RateLimit-Remaining'])
                if self.api_remaining % 100 == 0:
                    logger.info(f"Scopus API rate limit: {self.api_remaining}/{self.api_rate_limit} remaining")
            if 'X-RateLimit-Reset' in headers:
                import datetime
                self.api_reset_time = datetime.datetime.fromtimestamp(int(headers['X-RateLimit-Reset']))
        except (ValueError, KeyError) as e:
            logger.warning(f"Error parsing Scopus rate limit headers: {e}")


class ScopusSearchClient(BaseAPIClient):
    """Scopus search API client."""
    
    def __init__(self, config: ScopusConfig):
        self.config = config
        self.rate_limiter = ScopusRateLimiter(config)  # Use Scopus-specific rate limiter
        super().__init__(config)
    
    def _setup_session(self):
        """Setup Scopus session with API key."""
        if not self.config.api_key:
            raise ValueError(
                "Scopus API key is required. Set it in the config or load from scopus.yaml file. "
                "Get your API key from: https://dev.elsevier.com/"
            )
        self.session.headers.update({
            'Accept': 'application/json',
            'X-ELS-APIKey': self.config.api_key,
        })
        print (self.session.headers)
    
    def _build_search_url(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build Scopus search URL."""
        url_params = {**self.config.default_params, **(params or {})}
        param_str = '&'.join(f'{k}={v}' for k, v in url_params.items())
        return f"{self.config.base_url}?query={quote_plus(query)}&view={self.config.view}&{param_str}"
    
    def _parse_page_response(self, response_data: Dict[str, Any], page: int) -> Dict[str, Any]:
        """Parse Scopus API response."""
        # Check for API errors
        if 'error' in response_data:
            return {
                'page': page,
                'total_results': 0,
                'results': None,
                'error': response_data.get('error')
            }
        
        # Check for search results
        if 'search-results' not in response_data:
            return {
                'page': page,
                'total_results': 0,
                'results': None,
                'error': 'no_search_results'
            }
        
        search_results = response_data['search-results']
        total_results = int(search_results.get('opensearch:totalResults', 0))
        entries = search_results.get('entry', [])
        
        return {
            'page': page,
            'total_results': total_results,
            'results': entries,
            'cursor': None,  # Scopus uses links for pagination, not cursor
        }
    
    def _get_next_page_url(self, response_data: Dict[str, Any], current_url: str) -> Optional[str]:
        """Get next page URL from Scopus response."""
        if 'search-results' not in response_data:
            return None
        
        links = response_data['search-results'].get('link', [])
        next_links = [link for link in links if link.get('@ref') == 'next']
        
        if next_links:
            return next_links[0]['@href']
        
        return None


class ScopusSearchFetcher(BaseSearchFetcher):
    """
    Scopus search fetcher with caching.
    
    Main interface for Scopus searches.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "~/.cache/scopus/search",
        api_key_dir: str = "~/Documents/dh4pmp/api_keys",
        **kwargs
    ):
        """
        Initialize Scopus search fetcher.
        
        Args:
            api_key: Scopus API key (if None, tries to load from yaml)
            cache_dir: Directory for cache files
            api_key_dir: Directory containing API config files (default: ~/Documents/dh4pmp/api_keys)
            **kwargs: Additional configuration:
                - requests_per_second: Rate limit (default: 2.0)
                - max_results_per_query: Max results (default: 5000)
                - max_retries: Retry attempts (default: 3)
                - cache_max_age_days: Cache expiration (default: None/never)
        """
        # Load API key
        if api_key is None:
            api_key = self._load_api_key(api_key_dir)
        
        # Initialize configuration
        config = ScopusConfig(
            api_key=api_key,
            requests_per_second=kwargs.get('requests_per_second', 2.0),
            max_results_per_query=kwargs.get('max_hits', kwargs.get('max_results_per_query', 5000)),
            max_retries=kwargs.get('max_retries', 3),
        )
        
        # Initialize client and cache
        client = ScopusSearchClient(config)
        cache = LocalCache(
            cache_dir=cache_dir,
            compression=True,
            max_age_days=kwargs.get('cache_max_age_days', None)
        )
        
        super().__init__(client, cache)
        
        logger.info(f"Initialized Scopus fetcher with cache at {cache.cache_dir}")
    
    def _load_api_key(self, api_key_dir: str) -> str:
        """
        Load API key from YAML config file.
        
        File: scopus.yaml with required 'X-ELS-APIKey' field
        """
        import yaml
        
        api_key_dir = Path(api_key_dir).expanduser()
        key_paths = [
            Path('.') / 'scopus.yaml',
            api_key_dir / 'scopus.yaml',
        ]
        
        for path in key_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        key_data = yaml.safe_load(f)
                        
                        if key_data and 'X-ELS-APIKey' in key_data:
                            api_key = key_data['X-ELS-APIKey']
                            if api_key:  # Make sure it's not empty
                                logger.info(f"Loaded Scopus API key from {path}")
                                return api_key
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
        
        raise FileNotFoundError(
            f"Scopus API key not found. Create one of:\n"
            f"  - ./scopus.yaml\n"
            f"  - {api_key_dir}/scopus.yaml\n\n"
            f"With content:\n"
            f"  X-ELS-APIKey: your_api_key_here\n\n"
            f"Get your API key from: https://dev.elsevier.com/"
        )
    
    def fetch(self, query: str, force_refresh: bool = False, **params) -> Optional[pd.DataFrame]:
        """
        Fetch Scopus search results with optional parameters.
        
        Args:
            query: Scopus search query string
            force_refresh: If True, bypass cache
            **params: Additional Scopus API parameters
        
        Common Query Fields:
            TITLE(text)           - Search in title
            ABS(text)             - Search in abstract  
            KEY(text)             - Search in keywords
            TITLE-ABS-KEY(text)   - Search in title, abstract, or keywords
            AUTH(name)            - Author name
            AUTHFIRST(name)       - First author
            AUTHLASTNAME(name)    - Author last name
            AU-ID(id)             - Author ID
            AFFIL(name)           - Affiliation name
            AF-ID(id)             - Affiliation ID
            PUBYEAR(year)         - Publication year (e.g., PUBYEAR = 2024)
            PUBYEAR IS 2020       - Exact year
            PUBYEAR > 2020        - After year
            PUBYEAR < 2020        - Before year
            ISSN(issn)            - Journal ISSN
            ISBN(isbn)            - Book ISBN
            DOI(doi)              - Document DOI
            LANGUAGE(lang)        - Language (e.g., english)
            DOCTYPE(type)         - Document type (ar, re, cp, bk, ch)
            SUBJAREA(area)        - Subject area (COMP, MEDI, etc.)
            
        Common Parameters:
            count: int            - Results per page (default: 25, max: 200)
            start: int            - Starting index (for pagination)
            sort: str             - Sort field (e.g., '+coverDate', '-citedby-count')
            view: str             - Response view (STANDARD, COMPLETE)
            field: str            - Specific fields to return
            date: str             - Date range (YYYY or YYYY-YYYY)
            
        Examples:
            # Simple search
            results = fetcher.fetch("TITLE-ABS-KEY(machine learning)")
            
            # Author search
            results = fetcher.fetch("AUTH(Smith) AND PUBYEAR = 2024")
            
            # With parameters
            results = fetcher.fetch(
                "TITLE(neural networks)",
                count=50,
                sort="+coverDate"
            )
            
            # Complex query
            results = fetcher.fetch(
                "TITLE-ABS-KEY(AI) AND PUBYEAR > 2020 AND SUBJAREA(COMP)",
                count=100
            )
        
        Query Operators:
            AND, OR, AND NOT      - Boolean operators
            W/n                   - Within n words (e.g., neural W/3 network)
            PRE/n                 - Precedes within n words
            {phrase}              - Exact phrase (use quotes in query string)
            *                     - Wildcard (e.g., climat*)
            
        Returns:
            DataFrame with columns: ID, page, num_hits, data, error
            
        Reference:
            https://dev.elsevier.com/sc_search_tips.html
            https://dev.elsevier.com/guides/ScopusSearchGuide.pdf
        """
        # Use base fetch with params
        return super().fetch(query, force_refresh=force_refresh, **params)
    
    def clean_max_hits(self):
        """Remove cached queries that exceeded max_hits."""
        removed = 0
        for item in self.cache.list_queries():
            data = self.cache.get(item['query'])
            if data is not None:
                # Check if any row has data=None and num_hits > max_hits
                problematic = data[
                    (data['data'].isna()) & 
                    (data['num_hits'] > self.client.config.max_results_per_query)
                ]
                if len(problematic) > 0:
                    self.cache.delete(item['query'])
                    removed += 1
        
        if removed > 0:
            logger.info(f"Removed {removed} queries that exceeded max_hits")


# Keep the old class names for backward compatibility
class ScopusAbstractFetcher:
    """
    Fetcher for individual Scopus abstracts by EID.
    
    Note: This is a simplified version. For production use,
    consider extending BaseAPIClient similarly to ScopusSearchClient.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "~/.cache/scopus/abstracts",
        **kwargs
    ):
        """Initialize abstract fetcher."""
        import requests
        
        # Load API key
        if api_key is None:
            api_key = self._load_api_key()
        
        self.base_url = 'https://api.elsevier.com/content/abstract/eid'
        self.api_key = api_key
        self.cache = LocalCache(
            cache_dir=cache_dir,
            compression=True,
            max_age_days=kwargs.get('cache_max_age_days', None)
        )
        
        # Rate limiting
        self.requests_per_second = kwargs.get('requests_per_second', 2.0)
        self.sleep_time = 1.0 / self.requests_per_second
    
    def _load_api_key(self) -> str:
        """Load API key from yaml file."""
        key_paths = [
            Path('.') / 'scopus.yaml',
            Path('~/Documents/dh4pmp/api_keys').expanduser() / 'scopus.yaml',
        ]
        
        for path in key_paths:
            if path.exists():
                with open(path, 'r') as f:
                    key_data = yaml.safe_load(f)
                    if key_data and 'X-ELS-APIKey' in key_data:
                        return key_data['X-ELS-APIKey']
        
        raise FileNotFoundError("Scopus API key file not found")
    
    def fetch(self, eid: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch abstract data for an EID."""
        import requests
        import time
        import pandas as pd
        
        # Check cache
        if not force_refresh:
            cached = self.cache.get(eid)
            if cached is not None and len(cached) > 0:
                return cached.iloc[0]['data']
        
        # Fetch from API
        url = f'{self.base_url}/{eid}?view=META_ABS'
        headers = {
            'Accept': 'application/json',
            'X-ELS-APIKey': self.api_key,
        }
        
        try:
            r = requests.get(url, headers=headers, timeout=30)
            time.sleep(self.sleep_time)
            
            if r.ok:
                data = r.json()
                
                # Cache it
                df = pd.DataFrame([{
                    'ID': eid,
                    'data': data
                }])
                self.cache.store(eid, df)
                
                return data
            else:
                logger.error(f"Error {r.status_code} fetching {eid}")
                return None
        
        except Exception as e:
            logger.error(f"Exception fetching {eid}: {e}")
            return None
    
    def provide(self, eids: List[str], force_refresh: bool = False) -> pd.DataFrame:
        """Fetch multiple EIDs."""
        from tqdm import tqdm
        import pandas as pd
        
        results = []
        
        for eid in tqdm(eids, desc="Fetching abstracts"):
            data = self.fetch(eid, force_refresh=force_refresh)
            if data is not None:
                results.append({'ID': eid, 'data': data})
        
        return pd.DataFrame(results)
