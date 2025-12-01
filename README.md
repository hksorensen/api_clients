# api-clients

Unified API client framework for scholarly metadata APIs with shared infrastructure for rate limiting, caching, and error handling.

## Features

✅ **Unified Architecture** - Same patterns for all APIs  
✅ **Fast Local Caching** - 5-10x faster than database caching  
✅ **Proactive Rate Limiting** - Prevents API violations  
✅ **Exponential Backoff** - Smart error recovery  
✅ **Zero Configuration** - Auto-loads credentials  
✅ **Extensible** - Easy to add new APIs

## Supported APIs

- **Scopus** - Elsevier's abstract and citation database
- **Crossref** - DOI metadata and citations  
- **(Future: OpenAlex, PubMed, arXiv, etc.)**

## Installation

```bash
cd api_clients
pip install .
```

For development (editable mode):
```bash
pip install -e .
```

## Quick Start

### Crossref (Default - Easiest)

```python
from api_clients import CrossrefSearchFetcher

# Simple - auto-loads email from config file
crossref = CrossrefSearchFetcher()
results = crossref.fetch("machine learning")

# Or with explicit email
crossref = CrossrefSearchFetcher(mailto="your@email.com")
results = crossref.fetch("machine learning")

# Search with filters
results = crossref.search_with_filters(
    "deep learning",
    filters={'has-abstract': True, 'type': 'journal-article'}
)

# Get specific DOI
metadata = crossref.search_by_doi("10.1371/journal.pone.0033693")
```

### Scopus

```python
from api_clients import ScopusSearchFetcher

# Initialize (auto-loads API key from yaml)
scopus = ScopusSearchFetcher()

# Search
results = scopus.fetch("TITLE-ABS-KEY(machine learning)")

# Process results
print(f"Found {results.iloc[0]['num_hits']} papers")
```

## API Key / Email Configuration

All API config files are stored in: **`~/Documents/dh4pmp/api_keys/`** (by default)

Config files use **YAML format** for consistency. You can also place them in the current directory (`.`) as a fallback.

### Crossref Email Setup (Optional but Recommended)

Crossref works **without** any configuration, but providing your email gets you into the "polite pool" with 10x faster rate limits.

**Option 1: With email (recommended)** - Create `~/Documents/dh4pmp/api_keys/crossref.yaml`:
```yaml
mailto: your@email.com
```

Or use `email:` instead:
```yaml
email: your@email.com
```

**Option 2: Explicitly use public pool** - Create `~/Documents/dh4pmp/api_keys/crossref.yaml`:
```yaml
# Empty file or:
mailto: 
```

**Option 3: No file** - Just don't create `crossref.yaml` at all. Crossref will work fine with public pool.

**Why provide email?**
- ✅ Gets you into the "polite pool" with **10x faster rate limits** (50 req/sec vs 5 req/sec)
- ✅ Better API performance
- ✅ Crossref can contact you if there are issues
- ✅ No registration required - just provide your email

### Scopus API Key Setup (Required)

Scopus requires an API key. Get one from https://dev.elsevier.com/

Create `~/Documents/dh4pmp/api_keys/scopus.yaml`:
```yaml
X-ELS-APIKey: your_scopus_api_key_here
```

### Directory Structure

```
~/Documents/dh4pmp/api_keys/
├── crossref.yaml     # Optional: mailto: your@email.com
└── scopus.yaml       # Required: X-ELS-APIKey: your_key
```

Or use current directory:
```
./crossref.yaml
./scopus.yaml
```

### Custom API Key Directory

```python
# Use a different directory for API keys
crossref = CrossrefSearchFetcher(api_key_dir="/path/to/keys")
scopus = ScopusSearchFetcher(api_key_dir="/path/to/keys")
```

## Common Operations

### Progress Bars

Progress bars are **enabled by default** and work in both terminal and Jupyter notebooks!

```python
# Automatic progress bar (default)
results = crossref.fetch("machine learning")
# Shows: Fetching machine learning: |████████░░| 1234/5000 [25%]

# Disable if you want
results = crossref.fetch("AI", show_progress=False)

# Batch operations show progress too
results = crossref.provide(queries)
# Shows: Fetching queries: 3/5 [60%]
```

**Works in:**
- ✅ Terminal (text progress bar)
- ✅ Jupyter Notebook (widget progress bar)
- ✅ JupyterLab (widget progress bar)
- ✅ VSCode notebooks (widget progress bar)
- ✅ Google Colab (widget progress bar)

### Passing Parameters to fetch()

Both Scopus and Crossref support passing additional parameters:

```python
# Scopus with custom parameters
results = scopus.fetch(
    "TITLE-ABS-KEY(machine learning)",
    count=50,          # Results per page
    sort="+coverDate"  # Sort by date
)

# Crossref with custom parameters
results = crossref.fetch(
    "machine learning",
    rows=200,                    # Results per page
    sort="published",            # Sort order
    order="desc"                 # Descending
)

# Crossref with filters (easier syntax)
results = crossref.search_with_filters(
    "deep learning",
    filters={'has-abstract': True, 'type': 'journal-article'},
    rows=100
)
```

### Batch Fetching

```python
# Multiple queries
queries = [
    "TITLE-ABS-KEY(machine learning)",
    "TITLE-ABS-KEY(deep learning)",
]

results = scopus.provide(queries)
```

### Cache Management

```python
# Get cache stats
stats = scopus.get_stats()
print(f"Cached: {stats['num_entries']} queries")
print(f"Size: {stats['total_size_mb']:.1f} MB")

# List cached queries
queries = scopus.get_ID_list()

# Force refresh (bypass cache)
results = scopus.fetch(query, force_refresh=True)

# Clear cache
scopus.clear_cache()
```

### Crossref-Specific Features

```python
# Search with filters
results = crossref.search_with_filters(
    "climate change",
    filters={
        'has-abstract': True,
        'type': 'journal-article',
        'from-pub-date': '2020-01-01'
    }
)

# Field queries
results = crossref.fetch("query.title=neural networks")
results = crossref.fetch("query.author=Smith")

# Get DOI metadata
metadata = crossref.search_by_doi("10.1371/journal.pone.0033693")
```

## Architecture

### Package Structure

```
api_clients/
├── __init__.py           # Package exports
├── base_client.py        # Base classes (shared infrastructure)
├── local_cache.py        # File-based caching
├── scopus_client.py      # Scopus implementation
├── crossref_client.py    # Crossref implementation
├── setup.py              # Installation script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

### Class Hierarchy

```
BaseAPIClient (abstract)
├── ScopusSearchClient
└── CrossrefSearchClient

BaseSearchFetcher
├── ScopusSearchFetcher
└── CrossrefSearchFetcher
```

### Shared Components

- **TokenBucket** - Rate limiting algorithm
- **RateLimiter** - Coordinates local and API rate limits
- **LocalCache** - Fast file-based caching
- **BaseAPIClient** - Common API interaction patterns
- **BaseSearchFetcher** - Common caching/fetching patterns

## Adding New APIs

To add a new API, extend `BaseAPIClient`:

```python
from api_clients.base_client import BaseAPIClient, APIConfig

@dataclass
class MyAPIConfig(APIConfig):
    api_key: str = ""
    # ... your config

class MyAPIClient(BaseAPIClient):
    def _setup_session(self):
        # Setup authentication
        pass
    
    def _build_search_url(self, query, params):
        # Build URL
        pass
    
    def _parse_page_response(self, response_data, page):
        # Parse response
        return {
            'page': page,
            'total_results': ...,
            'results': ...,
        }
    
    def _get_next_page_url(self, response_data, current_url):
        # Get next page
        return next_url or None
```

See `scopus_client.py` and `crossref_client.py` for complete examples.

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Check cache | <1ms | Very fast |
| Read from cache | 1-3ms | Fast |
| Store to cache | 2-5ms | Fast |
| API request | 500-2000ms | Network dependent |

## File Locations

### Caches
```
~/.cache/scopus/search/     # Scopus search results
~/.cache/crossref/search/   # Crossref search results
```

### API Keys
```
./scopus.yaml               # Scopus key (local)
~/Documents/dh4pmp/api_keys/scopus.yaml  # Scopus key (shared)
```

## Comparison: Before vs After Refactoring

### Before (Separate Implementations)
```
scopus_client/
├── scopus_modern.py (19KB)
├── scopus_improved.py (17KB) 
├── local_cache.py (11KB)
└── ... (47KB total)

To add Crossref:
└── Duplicate everything → another 47KB
```

### After (Unified Framework)
```
api_clients/
├── base_client.py (15KB) ← Shared
├── local_cache.py (11KB) ← Shared
├── scopus_client.py (12KB) ← Scopus-specific only
├── crossref_client.py (12KB) ← Crossref-specific only
└── ... (50KB total for BOTH APIs)

To add new API:
└── Just ~10KB of API-specific code!
```

**Benefits:**
- ✅ Less code duplication
- ✅ Consistent patterns across APIs
- ✅ Fix bugs once, works everywhere
- ✅ Easy to add new APIs

## Dependencies

Required:
- pandas >= 1.0.0
- requests >= 2.25.0
- PyYAML >= 5.4.0

Optional:
- tqdm >= 4.60.0 (for progress bars)

## Examples

See the `examples/` directory for:
- Basic usage
- Batch processing
- Combining Scopus + Crossref data
- Custom configurations
- Extending with new APIs

## Troubleshooting

### Scopus Issues

**"API key file not found"**
```bash
# Create API key file
echo "X-ELS-APIKey: your_key" > scopus.yaml
```

**"Rate limit exceeded"**
```python
# Reduce rate
scopus = ScopusSearchFetcher(requests_per_second=1.0)
```

### Crossref Issues

**Slow responses**
```python
# Use polite pool for faster rate limits
crossref = CrossrefSearchFetcher(mailto="your@email.com")
```

**"Too many results"**
```python
# Use filters to narrow search
results = crossref.search_with_filters(
    query,
    filters={'from-pub-date': '2024-01-01'}
)
```

## License

MIT License - feel free to use in your research projects.

## Version

1.0.0 - Initial release with Scopus and Crossref support
