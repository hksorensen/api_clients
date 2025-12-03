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


class CrossrefBibliographicClient(CrossrefSearchClient):
    """
    Crossref client specialized for bibliographic citation resolution.
    
    This client is designed for resolving bibliographic citations to DOIs,
    where queries may return millions of total results but only the top
    match is needed. It bypasses max_results_per_query validation and
    always requests only the top result.
    
    Usage:
        config = CrossrefConfig(mailto="your@email.com")
        client = CrossrefBibliographicClient(config)
        result = client.resolve_bibliographic("Proc. Natl. Acad. Sci. U.S.A. 117, 15322 (2020)")
    """
    
    def resolve_bibliographic(
        self, 
        citation: str, 
        max_results: int = 5,
        min_score: Optional[float] = None,
        validate_doi: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a bibliographic citation to a DOI and return the full Crossref record.
        
        This method:
        - Uses query.bibliographic parameter (optimized for citation matching)
        - Can request multiple results and pick the best valid one
        - Bypasses max_results_per_query validation
        - Returns the full Crossref metadata record including relevance score
        
        Args:
            citation: Bibliographic citation string
                     (e.g., "Proc. Natl. Acad. Sci. U.S.A. 117, 15322 (2020)")
            max_results: Maximum number of results to consider (default: 5)
                        Higher values may find better matches but are slower
            min_score: Minimum score threshold (default: None, accepts any score)
                      Results below this score will be skipped
            validate_doi: If True, only return results with valid DOI format (default: True)
        
        Returns:
            Full Crossref record dict if found, None otherwise.
            The record contains all Crossref metadata fields including:
            - 'DOI': The DOI string
            - 'score': Relevance score (0-100+) indicating match quality
            - 'title': Article title
            - 'author': Author list
            - All other Crossref metadata fields
            
            **About the score:**
            The 'score' field is Crossref's relevance score for how well the
            bibliographic citation matches the returned record. Higher scores
            indicate better matches.
            
            **Important note for bibliographic queries:**
            Bibliographic queries (query.bibliographic) may have different score
            distributions than general text queries. Scores for bibliographic
            queries are often lower even when the match is correct, because:
            - Bibliographic citations are often incomplete or inconsistently formatted
            - Partial matches (e.g., missing authors or pages) reduce the score
            - The query.bibliographic parameter is designed to find the best match
              even with imperfect citations
            
            **Score interpretation:**
            - 80+: Excellent match (very high confidence)
            - 60-79: Good match (high confidence)  
            - 40-59: Moderate match (medium confidence - verify manually)
            - Below 40: Weak match (low confidence - likely incorrect, verify carefully)
            
            **Recommendation:**
            If you're getting correct DOIs with scores <40, consider:
            1. Verifying the matches are actually correct (low scores often indicate wrong matches)
            2. Lowering your confidence threshold to 30-40 for bibliographic queries
            3. Always manually verify matches with scores <50
            
            The score is calculated based on how many fields from the citation
            match the record (title, authors, journal, year, volume, pages, etc.)
            and how closely they match.
        
        Example:
            >>> client = CrossrefBibliographicClient(config)
            >>> result = client.resolve_bibliographic("Nature 123, 456 (2020)")
            >>> if result:
            ...     print(f"DOI: {result['DOI']}")
            ...     print(f"Score: {result.get('score', 0)}")
            ...     print(f"Title: {result.get('title', [None])[0]}")
        """
        # Import DOI validation function
        try:
            from sciec.citation_resolver import is_doi
        except ImportError:
            # Fallback if not available
            def is_doi(s):
                if not isinstance(s, str) or not s:
                    return False
                import re
                return bool(re.match(r"^10\.[0-9]{4,}(?:\.[0-9]+)*/.+", s.strip()))
        
        # Build bibliographic query
        bibliographic_query = f"query.bibliographic={citation}"
        
        # Request multiple results to find the best valid one
        params = {"rows": max_results}
        
        # Use search_iter which automatically bypasses max_results check for rows <= 10
        for page_data in self.search_iter(bibliographic_query, params):
            # Check for errors
            if page_data.get('error'):
                if page_data.get('error') == 'too_many_results':
                    # Even if too many results, we can still return the top match
                    logger.debug(
                        f"Query returned {page_data.get('total_results', 0)} total results, "
                        f"but returning top match (rows={max_results})"
                    )
                else:
                    logger.warning(f"Error resolving citation: {page_data.get('error')}")
                    return None
            
            # Get results from this page
            results = page_data.get('results')
            if results and len(results) > 0:
                # Filter and find the best result
                valid_results = []
                for record in results:
                    doi = record.get('DOI')
                    score = record.get('score', 0)
                    
                    # Skip if below minimum score
                    if min_score is not None and score < min_score:
                        continue
                    
                    # Validate DOI format if requested
                    if validate_doi and doi:
                        if not is_doi(doi):
                            logger.debug(f"Skipping result with invalid DOI format: {doi}")
                            continue
                    
                    # Ensure score is included
                    if 'score' not in record:
                        record['score'] = score
                    
                    valid_results.append(record)
                
                # Return the highest-scoring valid result
                if valid_results:
                    # Sort by score (descending) and return the best one
                    valid_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    best = valid_results[0]
                    
                    if len(valid_results) > 1:
                        logger.debug(
                            f"Found {len(valid_results)} valid results, "
                            f"returning highest score ({best.get('score', 0):.1f})"
                        )
                    
                    return best
                else:
                    logger.warning(
                        f"No valid results found (checked {len(results)} results, "
                        f"min_score={min_score}, validate_doi={validate_doi})"
                    )
                    # If no valid results but we have results, return the first one anyway
                    # (user can disable validation if needed)
                    if not validate_doi and min_score is None:
                        record = results[0]
                        if 'score' not in record:
                            record['score'] = record.get('score', 0)
                        return record
            
            # If we got here, no results on this page
            break
        
        # No results found
        return None
    
    def resolve_bibliographic_with_fallback(
        self,
        citation: str,
        max_results: int = 5,
        min_score: Optional[float] = None,
        validate_doi: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a bibliographic citation with fallback strategies.
        
        Tries multiple query strategies in order:
        1. query.bibliographic (optimized for citations)
        2. General query (broader search)
        
        Args:
            citation: Bibliographic citation string
            max_results: Maximum number of results to consider per strategy
            min_score: Minimum score threshold
            validate_doi: If True, only return results with valid DOI format
        
        Returns:
            Best matching Crossref record, or None if no valid match found
        """
        # Strategy 1: Bibliographic query (preferred)
        result = self.resolve_bibliographic(
            citation,
            max_results=max_results,
            min_score=min_score,
            validate_doi=validate_doi
        )
        
        if result:
            return result
        
        # Strategy 2: General query (fallback)
        logger.info(f"Bibliographic query failed, trying general query for: {citation[:60]}...")
        params = {"rows": max_results}
        
        for page_data in self.search_iter(citation, params):
            if page_data.get('error'):
                break
            
            results = page_data.get('results')
            if results and len(results) > 0:
                # Import DOI validation
                try:
                    from sciec.citation_resolver import is_doi
                except ImportError:
                    import re
                    def is_doi(s):
                        if not isinstance(s, str) or not s:
                            return False
                        return bool(re.match(r"^10\.[0-9]{4,}(?:\.[0-9]+)*/.+", s.strip()))
                
                # Filter and find best result
                valid_results = []
                for record in results:
                    doi = record.get('DOI')
                    score = record.get('score', 0)
                    
                    if min_score is not None and score < min_score:
                        continue
                    
                    if validate_doi and doi and not is_doi(doi):
                        continue
                    
                    valid_results.append(record)
                
                if valid_results:
                    valid_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    return valid_results[0]
            
            break
        
        return None


class CrossrefBibliographicFetcher:
    """
    Crossref bibliographic citation resolver with caching.
    
    Wraps CrossrefBibliographicClient with caching functionality for
    citation resolution. Designed for resolving bibliographic citations
    to DOIs with full metadata caching.
    
    Usage:
        # Simple (auto-loads email from config)
        fetcher = CrossrefBibliographicFetcher()
        
        # With explicit email
        fetcher = CrossrefBibliographicFetcher(mailto="your@email.com")
        
        # Resolve a citation
        result = fetcher.resolve("Proc. Natl. Acad. Sci. U.S.A. 117, 15322 (2020)")
    """
    
    def __init__(
        self,
        mailto: Optional[str] = None,
        cache_dir: str = "~/.cache/crossref/bibliographic",
        api_key_dir: str = "~/Documents/dh4pmp/api_keys",
        **kwargs
    ):
        """
        Initialize Crossref bibliographic fetcher.
        
        Args:
            mailto: Email address for polite pool access (if None, tries to load from yaml)
            cache_dir: Directory for cache files
            api_key_dir: Directory containing API config files
            **kwargs: Additional configuration:
                - requests_per_second: Rate limit (default: 10.0 for polite, 1.0 for public)
                - max_retries: Retry attempts (default: 3)
                - cache_max_age_days: Cache expiration (default: None/never)
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
            max_retries=kwargs.get('max_retries', 3),
        )
        
        # Initialize client and cache
        self.client = CrossrefBibliographicClient(config)
        self.cache = LocalCache(
            cache_dir=cache_dir,
            compression=True,
            max_age_days=kwargs.get('cache_max_age_days', None)
        )
        
        logger.info(f"Initialized Crossref bibliographic fetcher with cache at {self.cache.cache_dir}")
    
    def _load_email(self, api_key_dir: str) -> Optional[str]:
        """Load email from YAML config file (same as CrossrefSearchFetcher)."""
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
                        
                        if key_data is None:
                            logger.info(f"Found {path} but it's empty - using public pool")
                            return None
                        
                        email = key_data.get('mailto') or key_data.get('email')
                        
                        if email and isinstance(email, str) and '@' in email:
                            logger.info(f"Loaded Crossref email from {path}")
                            return email
                        elif email is None or email == '':
                            logger.info(f"Found {path} with empty email - using public pool")
                            return None
                        
                except Exception as e:
                    logger.warning(f"Error loading {path}: {e}")
        
        logger.info("No crossref.yaml found - using public pool (works fine, just slower)")
        return None
    
    def resolve(
        self, 
        citation: str, 
        force_refresh: bool = False,
        max_results: int = 5,
        min_score: Optional[float] = None,
        validate_doi: bool = True
    ) -> pd.DataFrame:
        """
        Resolve a bibliographic citation to a DOI with full Crossref metadata.
        
        Uses caching to avoid redundant API calls. The cache key is based on
        the citation string.
        
        Args:
            citation: Bibliographic citation string
            force_refresh: If True, bypass cache and fetch fresh data
            max_results: Maximum number of results to consider (default: 5)
                        Higher values may find better matches but are slower
            min_score: Minimum score threshold (default: None, accepts any score)
                      Results below this score will be skipped
            validate_doi: If True, only return results with valid DOI format (default: True)
        
        Returns:
            DataFrame with columns: ID, page, num_hits, data, error
            - ID: The citation string
            - page: Always 1 (single result)
            - num_hits: 1 if found, 0 if not found
            - data: List containing the full Crossref record (or empty list if not found)
            - error: None if successful, error message otherwise
            
            To access the DOI from the result:
                df = fetcher.resolve("citation")
                if len(df) > 0 and df.iloc[0]['data']:
                    record = df.iloc[0]['data'][0]  # Get first (and only) record
                    doi = record['DOI']  # Get the DOI
                    score = record.get('score', 0)  # Get the score
            
            **About the score:**
            The 'score' field (0-100+) indicates how well the citation matches
            the returned record. Higher = better match. 
            
            **Note:** Bibliographic queries often have lower scores (30-50) even
            for correct matches due to incomplete citation formatting. Scores <40
            should be verified manually. See CrossrefBibliographicClient documentation
            for detailed score interpretation.
            
            **Improving resolution:**
            If you're getting invalid or incorrect DOIs, try:
            1. Increase max_results (e.g., 10) to consider more candidates
            2. Set min_score (e.g., 30) to filter out very low-confidence matches
            3. Keep validate_doi=True to filter out malformed DOIs
        """
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(citation)
            if cached is not None:
                logger.debug(f"Cache hit for citation: {citation[:50]}...")
                # Cache stores DataFrames in the expected format
                if isinstance(cached, pd.DataFrame):
                    return cached
        
        # Fetch from API
        logger.info(f"Resolving citation: {citation[:60]}...")
        result = self.client.resolve_bibliographic(
            citation,
            max_results=max_results,
            min_score=min_score,
            validate_doi=validate_doi
        )
        
        # Format as DataFrame (consistent with CrossrefSearchFetcher)
        if result is None:
            # No result found
            df = pd.DataFrame([{
                'ID': citation,
                'page': 1,
                'num_hits': 0,
                'data': None,  # No data
                'error': None
            }])
        else:
            # Result found - store in DataFrame format
            df = pd.DataFrame([{
                'ID': citation,
                'page': 1,
                'num_hits': 1,
                'data': [result],  # Store as list to match CrossrefSearchFetcher format
                'error': None
            }])
        
        # Store in cache
        self.cache.store(citation, df, total_results=df.iloc[0]['num_hits'], num_pages=1)
        
        return df
    
    def resolve_doi(
        self,
        citation: str,
        force_refresh: bool = False,
        max_results: int = 5,
        min_score: Optional[float] = None,
        validate_doi: bool = True
    ) -> Optional[str]:
        """
        Resolve a bibliographic citation and return just the DOI string.
        
        Convenience method that calls resolve() and extracts the DOI.
        
        Args:
            citation: Bibliographic citation string
            force_refresh: If True, bypass cache and fetch fresh data
            max_results: Maximum number of results to consider
            min_score: Minimum score threshold
            validate_doi: If True, only return results with valid DOI format
        
        Returns:
            DOI string if found, None otherwise
        
        Example:
            >>> fetcher = CrossrefBibliographicFetcher()
            >>> doi = fetcher.resolve_doi("Nature 123, 456 (2020)")
            >>> print(doi)
            '10.1038/nature12345'
        """
        df = self.resolve(
            citation,
            force_refresh=force_refresh,
            max_results=max_results,
            min_score=min_score,
            validate_doi=validate_doi
        )
        
        # Extract DOI from DataFrame
        if len(df) > 0 and df.iloc[0]['data']:
            data = df.iloc[0]['data']
            if isinstance(data, list) and len(data) > 0:
                record = data[0]
                return record.get('DOI')
        
        return None
    
    def resolve_candidates(
        self,
        citation: str,
        force_refresh: bool = False,
        max_results: int = 10,
        min_score: Optional[float] = None,
        validate_doi: bool = True
    ) -> pd.DataFrame:
        """
        Resolve a bibliographic citation and return multiple candidate matches.
        
        This method returns ALL valid candidates (not just the top one), allowing
        you to review and pick the best match. Useful when the top result is wrong.
        
        Args:
            citation: Bibliographic citation string
            force_refresh: If True, bypass cache and fetch fresh data
            max_results: Maximum number of candidates to return (default: 10)
            min_score: Minimum score threshold (default: None, accepts any score)
                      Results below this score will be skipped
            validate_doi: If True, only return results with valid DOI format (default: True)
        
        Returns:
            DataFrame with columns: ID, page, num_hits, data, error
            - ID: The citation string
            - page: Always 1
            - num_hits: Number of valid candidates found
            - data: List containing ALL valid Crossref records, sorted by score (highest first)
            - error: None if successful, error message otherwise
            
            To access candidates:
                df = fetcher.resolve_candidates("citation")
                if len(df) > 0 and df.iloc[0]['data']:
                    candidates = df.iloc[0]['data']  # List of all candidates
                    for i, record in enumerate(candidates):
                        print(f"Candidate {i+1}: {record['DOI']} (score: {record.get('score', 0)})")
        
        Example:
            >>> fetcher = CrossrefBibliographicFetcher()
            >>> df = fetcher.resolve_candidates("Nature 123, 456 (2020)", max_results=5)
            >>> candidates = df.iloc[0]['data']
            >>> # Review candidates and pick the best one
            >>> best_match = candidates[0]  # Highest score
            >>> alternative = candidates[1]  # Second best
        """
        # Import DOI validation function
        try:
            from sciec.citation_resolver import is_doi
        except ImportError:
            import re
            def is_doi(s):
                if not isinstance(s, str) or not s:
                    return False
                return bool(re.match(r"^10\.[0-9]{4,}(?:\.[0-9]+)*/.+", s.strip()))
        
        # Build bibliographic query
        bibliographic_query = f"query.bibliographic={citation}"
        params = {"rows": max_results}
        
        # Fetch from API
        logger.info(f"Fetching {max_results} candidates for: {citation[:60]}...")
        
        all_candidates = []
        
        for page_data in self.client.search_iter(bibliographic_query, params):
            if page_data.get('error'):
                if page_data.get('error') == 'too_many_results':
                    logger.debug(
                        f"Query returned {page_data.get('total_results', 0)} total results, "
                        f"but returning top {max_results} candidates"
                    )
                else:
                    logger.warning(f"Error resolving citation: {page_data.get('error')}")
                    break
            
            results = page_data.get('results')
            if results:
                # Filter and collect all valid candidates
                for record in results:
                    doi = record.get('DOI')
                    score = record.get('score', 0)
                    
                    # Skip if below minimum score
                    if min_score is not None and score < min_score:
                        continue
                    
                    # Validate DOI format if requested
                    if validate_doi and doi:
                        if not is_doi(doi):
                            logger.debug(f"Skipping result with invalid DOI format: {doi}")
                            continue
                    
                    # Ensure score is included
                    if 'score' not in record:
                        record['score'] = score
                    
                    all_candidates.append(record)
            
            # Only need first page for candidates
            break
        
        # Sort by score (highest first)
        all_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Format as DataFrame
        df = pd.DataFrame([{
            'ID': citation,
            'page': 1,
            'num_hits': len(all_candidates),
            'data': all_candidates,  # All candidates, sorted by score
            'error': None
        }])
        
        logger.info(f"Found {len(all_candidates)} valid candidates (sorted by score)")
        
        return df
    
    def resolve_candidates_with_fallback(
        self,
        citation: str,
        force_refresh: bool = False,
        max_results: int = 10,
        min_score: Optional[float] = None,
        validate_doi: bool = True,
        try_general_query: bool = True
    ) -> pd.DataFrame:
        """
        Resolve a bibliographic citation using multiple query strategies.
        
        Tries different query approaches in order:
        1. query.bibliographic (optimized for citations)
        2. General query (broader search, may find matches bibliographic misses)
        
        This is useful when bibliographic queries don't find the correct DOI.
        
        Args:
            citation: Bibliographic citation string
            force_refresh: If True, bypass cache and fetch fresh data
            max_results: Maximum number of candidates per strategy (default: 10)
            min_score: Minimum score threshold (default: None, accepts any score)
            validate_doi: If True, only return results with valid DOI format (default: True)
            try_general_query: If True, also try general query if bibliographic fails (default: True)
        
        Returns:
            DataFrame with columns: ID, page, num_hits, data, error
            - data: List containing ALL valid Crossref records from all strategies,
                    sorted by score (highest first)
            - Candidates are deduplicated by DOI
        
        Example:
            >>> fetcher = CrossrefBibliographicFetcher()
            >>> df = fetcher.resolve_candidates_with_fallback("citation", max_results=10)
            >>> candidates = df.iloc[0]['data']
            >>> # Review all candidates from both query strategies
        """
        # Import DOI validation function
        try:
            from sciec.citation_resolver import is_doi
        except ImportError:
            import re
            def is_doi(s):
                if not isinstance(s, str) or not s:
                    return False
                return bool(re.match(r"^10\.[0-9]{4,}(?:\.[0-9]+)*/.+", s.strip()))
        
        all_candidates = []
        seen_dois = set()  # For deduplication
        
        def add_candidate(record):
            """Add candidate if valid and not duplicate."""
            doi = record.get('DOI')
            score = record.get('score', 0)
            
            # Skip if below minimum score
            if min_score is not None and score < min_score:
                return False
            
            # Validate DOI format if requested
            if validate_doi and doi:
                if not is_doi(doi):
                    return False
            
            # Deduplicate by DOI
            if doi and doi in seen_dois:
                return False
            
            # Ensure score is included
            if 'score' not in record:
                record['score'] = score
            
            all_candidates.append(record)
            if doi:
                seen_dois.add(doi)
            return True
        
        # Strategy 1: Bibliographic query
        logger.info(f"Strategy 1: Trying bibliographic query for: {citation[:60]}...")
        bibliographic_query = f"query.bibliographic={citation}"
        params = {"rows": max_results}
        
        for page_data in self.client.search_iter(bibliographic_query, params):
            if page_data.get('error') and page_data.get('error') != 'too_many_results':
                break
            
            results = page_data.get('results')
            if results:
                for record in results:
                    add_candidate(record)
            break
        
        logger.info(f"Bibliographic query found {len(all_candidates)} candidates")
        
        # Strategy 2: General query (if enabled and we want more candidates)
        if try_general_query:
            logger.info(f"Strategy 2: Trying general query for: {citation[:60]}...")
            params = {"rows": max_results}
            
            for page_data in self.client.search_iter(citation, params):
                if page_data.get('error') and page_data.get('error') != 'too_many_results':
                    break
                
                results = page_data.get('results')
                if results:
                    added_count = 0
                    for record in results:
                        if add_candidate(record):
                            added_count += 1
                    logger.info(f"General query added {added_count} new candidates")
                break
        
        # Sort by score (highest first)
        all_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Format as DataFrame
        df = pd.DataFrame([{
            'ID': citation,
            'page': 1,
            'num_hits': len(all_candidates),
            'data': all_candidates,  # All candidates from all strategies, sorted by score
            'error': None
        }])
        
        logger.info(f"Total: {len(all_candidates)} unique candidates from all strategies")
        
        return df
    
    def resolve_candidates_extended(
        self,
        citation: str,
        force_refresh: bool = False,
        max_results: int = 20,
        min_score: Optional[float] = None,
        validate_doi: bool = True
    ) -> pd.DataFrame:
        """
        Extended candidate search with multiple strategies and higher result limits.
        
        This method tries:
        1. Bibliographic query (up to max_results candidates)
        2. General query (up to max_results candidates)
        3. Removes duplicates and sorts by score
        
        Use this when regular resolve_candidates() doesn't find the correct DOI.
        
        Args:
            citation: Bibliographic citation string
            force_refresh: If True, bypass cache and fetch fresh data
            max_results: Maximum candidates per strategy (default: 20, higher = more candidates)
            min_score: Minimum score threshold (default: None)
            validate_doi: If True, only return results with valid DOI format
        
        Returns:
            DataFrame with all unique candidates from all strategies, sorted by score
        
        Example:
            >>> fetcher = CrossrefBibliographicFetcher()
            >>> # Try with more candidates and multiple strategies
            >>> df = fetcher.resolve_candidates_extended("citation", max_results=30)
            >>> candidates = df.iloc[0]['data']
            >>> print(f"Found {len(candidates)} candidates to review")
        """
        return self.resolve_candidates_with_fallback(
            citation,
            force_refresh=force_refresh,
            max_results=max_results,
            min_score=min_score,
            validate_doi=validate_doi,
            try_general_query=True
        )
    
    def fetch_by_doi(
        self,
        doi: str,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch full Crossref metadata for a specific DOI.
        
        This is the most reliable way to get metadata when you already know the DOI.
        Uses caching to avoid redundant API calls.
        
        Args:
            doi: DOI string (e.g., "10.1126/science.abe1107")
            force_refresh: If True, bypass cache and fetch fresh data
        
        Returns:
            Full Crossref metadata dict, or None if DOI not found.
            Contains all metadata fields: title, authors, journal, year, etc.
        
        Example:
            >>> fetcher = CrossrefBibliographicFetcher()
            >>> metadata = fetcher.fetch_by_doi("10.1126/science.abe1107")
            >>> if metadata:
            ...     print(f"Title: {metadata.get('title', [None])[0]}")
            ...     print(f"Journal: {metadata.get('container-title', [None])[0]}")
            ...     print(f"Year: {metadata.get('published-print', {}).get('date-parts', [[None]])[0][0]}")
        """
        # Check cache first
        cache_key = f"doi:{doi}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for DOI: {doi}")
                # Cache might store as DataFrame or dict
                if isinstance(cached, pd.DataFrame) and len(cached) > 0:
                    if 'data' in cached.columns:
                        data = cached.iloc[0]['data']
                        if isinstance(data, list) and len(data) > 0:
                            return data[0]
                        return data
                    return cached.iloc[0].to_dict()
                elif isinstance(cached, dict):
                    return cached
        
        # Fetch from API
        logger.info(f"Fetching metadata for DOI: {doi}")
        
        # Build URL
        url = f"https://api.crossref.org/works/{doi}"
        
        # Use the client's session for proper headers and rate limiting
        try:
            response = self.client._make_request(url)
            if response and response.ok:
                data = response.json()
                if 'message' in data:
                    metadata = data['message']
                    
                    # Store in cache
                    cache_df = pd.DataFrame([{
                        'ID': cache_key,
                        'page': 1,
                        'num_hits': 1,
                        'data': [metadata],
                        'error': None
                    }])
                    self.cache.store(cache_key, cache_df, total_results=1, num_pages=1)
                    
                    return metadata
            else:
                logger.warning(f"DOI not found or error: {doi}")
                return None
        except Exception as e:
            logger.error(f"Error fetching DOI {doi}: {e}")
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


    def expand(
        self,
        results: pd.DataFrame,
        clean_title: bool = True,
        clean_authors: bool = True,
        extract_year: bool = True,
        extract_citations: bool = True
    ) -> pd.DataFrame:
        """
        Expand Crossref results with automatic data cleaning.
        
        Args:
            results: DataFrame from fetch()
            clean_title: Extract title from list format (default: True)
            clean_authors: Format author names as readable strings (default: True)
            extract_year: Add publication_year column (default: True)
            extract_citations: Add citation_count column (default: True)
            
        Returns:
            Cleaned DataFrame with one row per paper
            
        Example:
            >>> results = crossref.fetch("machine learning")
            >>> papers = crossref.expand(results, clean_title=True, extract_year=True)
            >>> papers[['DOI', 'title', 'publication_year', 'author_names']].head()
        """
        # Call base implementation
        df = super().expand(results)
        
        if df.empty:
            return df
        
        # Clean title (Crossref returns as list)
        if clean_title and 'title' in df.columns:
            df['title'] = df['title'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
            )
        
        # Format authors
        if clean_authors and 'author' in df.columns:
            df['author_names'] = df['author'].apply(self._format_author_names)
            df['first_author'] = df['author'].apply(self._get_first_author)
        
        # Extract publication year
        if extract_year and 'published-print' in df.columns:
            df['publication_year'] = df['published-print'].apply(
                lambda x: x.get('date-parts', [[None]])[0][0] 
                if isinstance(x, dict) and 'date-parts' in x 
                else None
            )
        
        # Extract citation count
        if extract_citations and 'is-referenced-by-count' in df.columns:
            df['citation_count'] = df['is-referenced-by-count']
        
        return df

    @staticmethod
    def _format_author_names(authors):
        """Format author list to readable names."""
        if not authors or not isinstance(authors, list):
            return []
        
        names = []
        for author in authors:
            given = author.get('given', '')
            family = author.get('family', '')
            if given and family:
                names.append(f"{given} {family}")
            elif family:
                names.append(family)
        
        return names

    @staticmethod
    def _get_first_author(authors):
        """Get first author name."""
        if not authors or not isinstance(authors, list) or len(authors) == 0:
            return None
        
        first = authors[0]
        given = first.get('given', '')
        family = first.get('family', '')
        
        if given and family:
            return f"{given} {family}"
        elif family:
            return family
        return None

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
