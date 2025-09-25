# Web Scraping Complete Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Web Scraping Basics](#web-scraping-basics)
3. [Simple HTML Extraction](#simple-html-extraction)
4. [JavaScript-Rendered Pages](#javascript-rendered-pages)
5. [Recursive Crawling](#recursive-crawling)
6. [Advanced Scraping Techniques](#advanced-scraping-techniques)
7. [Rate Limiting and Politeness](#rate-limiting-and-politeness)
8. [Handling Authentication](#handling-authentication)
9. [Data Extraction Strategies](#data-extraction-strategies)
10. [Error Handling and Retries](#error-handling-and-retries)
11. [Best Practices](#best-practices)
12. [Legal and Ethical Considerations](#legal-and-ethical-considerations)

## Introduction

Web scraping with Dataseter allows you to extract structured data from websites for AI training datasets. This guide covers everything from basic HTML extraction to complex JavaScript-rendered sites.

### Key Features

- **Static HTML Extraction**: Fast extraction from regular HTML pages
- **JavaScript Rendering**: Handle dynamic content with Selenium/Playwright
- **Recursive Crawling**: Follow links to specified depth
- **Parallel Processing**: Scrape multiple pages simultaneously
- **Smart Rate Limiting**: Respect server resources
- **Robot.txt Compliance**: Automatic robots.txt checking
- **Session Management**: Handle cookies and authentication
- **Error Recovery**: Automatic retries and fallbacks

## Web Scraping Basics

### Understanding Web Content

```python
# Different types of web content
content_types = {
    'static_html': 'Content in initial HTML response',
    'dynamic_js': 'Content loaded by JavaScript',
    'api_data': 'Data fetched from APIs',
    'media': 'Images, videos, audio files',
    'documents': 'PDFs, DOCs linked from pages'
}
```

### Basic Scraping Workflow

```
URL Input → Fetch Page → Parse HTML → Extract Data → Clean Text → Store Result
     ↓          ↓            ↓            ↓            ↓            ↓
  Validate   Request     BeautifulSoup  Selectors   Process    Dataset
   URL        Page          Parse        Extract     Content     Save
```

## Simple HTML Extraction

### Single Page Extraction

#### Command Line

```bash
# Basic web page extraction
dataseter create --website https://example.com/article -o output.jsonl

# With custom user agent
dataseter create \
  --website https://example.com \
  --user-agent "MyBot/1.0" \
  -o output.jsonl

# Extract specific elements
dataseter create \
  --website https://example.com \
  --css-selector "article.content" \
  -o output.jsonl
```

#### Python API

```python
from dataseter import WebExtractor

# Initialize extractor
extractor = WebExtractor({
    'user_agent': 'Dataseter/1.0',
    'timeout': 30
})

# Extract single page
result = extractor.extract('https://example.com/article')

print(f"Title: {result['metadata']['title']}")
print(f"Text length: {len(result['text'])}")
print(f"Links found: {len(result['links'])}")
```

### Multiple Pages

```python
from dataseter import DatasetCreator

# Create dataset from multiple URLs
creator = DatasetCreator()

urls = [
    'https://example.com/page1',
    'https://example.com/page2',
    'https://example.com/page3'
]

for url in urls:
    creator.add_website(url)

# Process all pages
dataset = creator.process()
dataset.to_jsonl('websites.jsonl')
```

### Custom Extraction Rules

```python
from dataseter import WebExtractor
from bs4 import BeautifulSoup

class CustomExtractor(WebExtractor):
    def extract_content(self, html, url):
        soup = BeautifulSoup(html, 'html.parser')

        # Custom extraction logic
        data = {
            'title': soup.find('h1').text if soup.find('h1') else '',
            'author': soup.find('span', class_='author').text if soup.find('span', class_='author') else '',
            'date': soup.find('time')['datetime'] if soup.find('time') else '',
            'content': soup.find('div', class_='article-body').text if soup.find('div', class_='article-body') else '',
            'tags': [tag.text for tag in soup.find_all('a', class_='tag')]
        }

        return data

# Use custom extractor
extractor = CustomExtractor()
result = extractor.extract('https://example.com/article')
```

## JavaScript-Rendered Pages

### Setting Up JavaScript Rendering

#### Using Selenium

```python
from dataseter import WebExtractor

# Configure for Selenium
config = {
    'javascript_rendering': True,
    'browser': 'chrome',  # or 'firefox', 'safari'
    'headless': True,     # Run without GUI
    'wait_time': 5,       # Wait for JS to load
}

extractor = WebExtractor(config)
result = extractor.extract('https://spa-website.com')
```

#### Using Playwright

```python
# Configure for Playwright (faster, more reliable)
config = {
    'javascript_rendering': True,
    'renderer': 'playwright',
    'browser': 'chromium',
    'headless': True,
    'viewport': {'width': 1920, 'height': 1080}
}

extractor = WebExtractor(config)
result = extractor.extract('https://modern-webapp.com')
```

### Waiting for Dynamic Content

```python
from dataseter import WebExtractor

config = {
    'javascript_rendering': True,
    'wait_conditions': [
        {'type': 'element', 'selector': 'div.content'},
        {'type': 'text', 'text': 'Loading complete'},
        {'type': 'javascript', 'script': 'return document.readyState === "complete"'},
    ],
    'max_wait_time': 30
}

extractor = WebExtractor(config)
```

### Interacting with Pages

```python
from dataseter import WebExtractor

# Interact with page before extraction
config = {
    'javascript_rendering': True,
    'interactions': [
        {'action': 'click', 'selector': 'button.load-more'},
        {'action': 'scroll', 'distance': 1000},
        {'action': 'wait', 'seconds': 2},
        {'action': 'input', 'selector': 'input.search', 'text': 'query'},
    ]
}

extractor = WebExtractor(config)
result = extractor.extract('https://interactive-site.com')
```

## Recursive Crawling

### Basic Crawling

```bash
# Crawl website to depth 2
dataseter create \
  --website https://docs.example.com \
  --max-depth 2 \
  --max-pages 100 \
  -o documentation.jsonl
```

### Advanced Crawling Configuration

```python
from dataseter import WebExtractor

config = {
    'max_depth': 3,
    'max_pages': 1000,
    'follow_links': True,
    'link_patterns': [
        r'/docs/.*',      # Only follow documentation links
        r'/api/.*',       # And API reference links
    ],
    'exclude_patterns': [
        r'.*\.(jpg|png|gif|pdf)$',  # Skip media files
        r'.*/download/.*',           # Skip download links
        r'.*/login.*',               # Skip login pages
    ],
    'allowed_domains': ['docs.example.com', 'api.example.com'],
    'crawl_delay': 1.0  # Delay between requests
}

extractor = WebExtractor(config)
```

### Sitemap-Based Crawling

```python
from dataseter import WebExtractor

# Use sitemap for efficient crawling
config = {
    'use_sitemap': True,
    'sitemap_url': 'https://example.com/sitemap.xml',
    'sitemap_filter': {
        'lastmod': '2024-01-01',  # Only pages modified after
        'priority': 0.5,          # Minimum priority
    }
}

extractor = WebExtractor(config)
results = extractor.crawl_sitemap('https://example.com')
```

### Breadth-First vs Depth-First

```python
# Breadth-first crawling (default)
config_bfs = {
    'crawl_strategy': 'breadth-first',
    'max_depth': 3
}

# Depth-first crawling
config_dfs = {
    'crawl_strategy': 'depth-first',
    'max_depth': 5,
    'max_branch_pages': 10  # Limit per branch
}
```

## Advanced Scraping Techniques

### Handling Pagination

```python
from dataseter import WebExtractor

class PaginatedExtractor(WebExtractor):
    def extract_paginated(self, base_url, max_pages=None):
        results = []
        page = 1

        while True:
            # Construct page URL
            url = f"{base_url}?page={page}"

            # Extract page
            result = self.extract(url)
            if not result or not result.get('text'):
                break

            results.append(result)

            # Check for next page
            if max_pages and page >= max_pages:
                break

            # Look for "next" button
            soup = BeautifulSoup(result['html'], 'html.parser')
            next_button = soup.find('a', class_='next')
            if not next_button:
                break

            page += 1

        return results

# Use paginated extraction
extractor = PaginatedExtractor()
all_results = extractor.extract_paginated(
    'https://blog.example.com/posts',
    max_pages=10
)
```

### AJAX and API Extraction

```python
import requests
from dataseter import WebExtractor

class APIExtractor(WebExtractor):
    def extract_via_api(self, page_url):
        # First, get the page to find API endpoints
        page_result = self.extract(page_url)

        # Extract API calls from JavaScript
        api_endpoints = self.find_api_endpoints(page_result['html'])

        # Call APIs directly
        api_data = []
        for endpoint in api_endpoints:
            response = requests.get(endpoint, headers={
                'User-Agent': self.config['user_agent'],
                'Referer': page_url
            })
            if response.ok:
                api_data.append(response.json())

        # Combine page and API data
        result = {
            'page_content': page_result['text'],
            'api_data': api_data
        }

        return result

    def find_api_endpoints(self, html):
        # Parse JavaScript to find API calls
        import re
        pattern = r'fetch\(["\']([^"\']+)["\']'
        endpoints = re.findall(pattern, html)
        return endpoints
```

### Infinite Scroll Handling

```python
from dataseter import WebExtractor
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

class InfiniteScrollExtractor(WebExtractor):
    def extract_infinite_scroll(self, url):
        # Setup driver
        driver = self.get_driver()
        driver.get(url)

        last_height = driver.execute_script("return document.body.scrollHeight")
        content = []

        while True:
            # Scroll to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait for new content
            time.sleep(2)

            # Extract newly loaded content
            elements = driver.find_elements(By.CLASS_NAME, "content-item")
            for element in elements:
                if element.text not in content:
                    content.append(element.text)

            # Check if more content loaded
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        driver.quit()
        return {'text': '\n'.join(content)}
```

## Rate Limiting and Politeness

### Respecting robots.txt

```python
from dataseter import WebExtractor

config = {
    'respect_robots': True,  # Default: True
    'robots_cache_timeout': 3600,  # Cache for 1 hour
}

extractor = WebExtractor(config)

# Check if URL is allowed
if extractor.is_allowed('https://example.com/page'):
    result = extractor.extract('https://example.com/page')
```

### Rate Limiting Strategies

```python
# Fixed delay between requests
config_fixed = {
    'rate_limit': 1.0,  # 1 request per second
}

# Adaptive rate limiting
config_adaptive = {
    'rate_limit': 'adaptive',
    'min_delay': 0.5,
    'max_delay': 5.0,
    'target_response_time': 1.0,  # Adjust based on server response
}

# Exponential backoff
config_backoff = {
    'rate_limit': 'exponential',
    'initial_delay': 1.0,
    'max_delay': 60.0,
    'backoff_factor': 2.0,
}
```

### Concurrent Requests with Limits

```python
from dataseter import WebExtractor
import asyncio
from aiohttp import ClientSession

class ConcurrentExtractor(WebExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.semaphore = asyncio.Semaphore(config.get('max_concurrent', 5))

    async def extract_async(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                html = await response.text()
                return self.parse_html(html, url)

    async def extract_many(self, urls):
        async with ClientSession() as session:
            tasks = [self.extract_async(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return results

# Use concurrent extraction
extractor = ConcurrentExtractor({'max_concurrent': 10})
urls = ['https://example.com/page{}'.format(i) for i in range(100)]
results = asyncio.run(extractor.extract_many(urls))
```

## Handling Authentication

### Basic Authentication

```python
from dataseter import WebExtractor

config = {
    'auth': {
        'type': 'basic',
        'username': 'user',
        'password': 'pass'
    }
}

extractor = WebExtractor(config)
result = extractor.extract('https://protected.example.com')
```

### Cookie-Based Authentication

```python
# Login and save cookies
config = {
    'auth': {
        'type': 'cookie',
        'login_url': 'https://example.com/login',
        'credentials': {
            'username': 'user',
            'password': 'pass'
        },
        'form_selectors': {
            'username': 'input[name="username"]',
            'password': 'input[name="password"]',
            'submit': 'button[type="submit"]'
        }
    },
    'persist_cookies': True,
    'cookie_file': 'cookies.json'
}

extractor = WebExtractor(config)
```

### Token-Based Authentication

```python
# API token authentication
config = {
    'auth': {
        'type': 'bearer',
        'token': 'your-api-token'
    },
    'headers': {
        'Accept': 'application/json',
        'X-API-Version': 'v1'
    }
}

extractor = WebExtractor(config)
```

### OAuth2 Authentication

```python
from dataseter import WebExtractor
import requests

class OAuth2Extractor(WebExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.token = self.get_oauth_token()

    def get_oauth_token(self):
        response = requests.post(
            self.config['oauth']['token_url'],
            data={
                'grant_type': 'client_credentials',
                'client_id': self.config['oauth']['client_id'],
                'client_secret': self.config['oauth']['client_secret']
            }
        )
        return response.json()['access_token']

    def extract(self, url):
        headers = {
            'Authorization': f'Bearer {self.token}'
        }
        return super().extract(url, headers=headers)
```

## Data Extraction Strategies

### CSS Selectors

```python
from dataseter import WebExtractor

config = {
    'extraction_rules': {
        'title': 'h1.article-title',
        'author': 'span.author-name',
        'date': 'time[datetime]',
        'content': 'div.article-body',
        'tags': 'a.tag',  # Multiple elements
        'comments': 'div.comment',  # Multiple elements
    }
}

extractor = WebExtractor(config)
result = extractor.extract('https://example.com/article')

# Access extracted data
print(result['extracted']['title'])
print(result['extracted']['tags'])  # List of tags
```

### XPath Expressions

```python
config = {
    'use_xpath': True,
    'extraction_rules': {
        'title': '//h1[@class="title"]/text()',
        'author': '//div[@class="author"]//span/text()',
        'content': '//article//p/text()',
        'links': '//a[@href]/@href',
    }
}
```

### Regular Expressions

```python
import re
from dataseter import WebExtractor

class RegexExtractor(WebExtractor):
    def extract_with_regex(self, html, patterns):
        results = {}
        for name, pattern in patterns.items():
            matches = re.findall(pattern, html)
            results[name] = matches[0] if len(matches) == 1 else matches
        return results

# Use regex extraction
extractor = RegexExtractor()
patterns = {
    'emails': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phones': r'\+?[\d\s\-\(\)]+',
    'prices': r'\$[\d,]+\.?\d*',
    'dates': r'\d{4}-\d{2}-\d{2}',
}

result = extractor.extract('https://example.com/contact')
extracted = extractor.extract_with_regex(result['html'], patterns)
```

### Structured Data Extraction

```python
from dataseter import WebExtractor
import json

class StructuredDataExtractor(WebExtractor):
    def extract_structured(self, url):
        result = self.extract(url)
        soup = BeautifulSoup(result['html'], 'html.parser')

        structured = {}

        # Extract JSON-LD
        json_ld = soup.find('script', type='application/ld+json')
        if json_ld:
            structured['json_ld'] = json.loads(json_ld.string)

        # Extract OpenGraph
        og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
        structured['opengraph'] = {
            tag.get('property'): tag.get('content')
            for tag in og_tags
        }

        # Extract Twitter Cards
        twitter_tags = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
        structured['twitter'] = {
            tag.get('name'): tag.get('content')
            for tag in twitter_tags
        }

        # Extract microdata
        items = soup.find_all(attrs={'itemscope': True})
        structured['microdata'] = []
        for item in items:
            data = {
                'type': item.get('itemtype'),
                'properties': {}
            }
            props = item.find_all(attrs={'itemprop': True})
            for prop in props:
                data['properties'][prop.get('itemprop')] = prop.get_text()
            structured['microdata'].append(data)

        return structured
```

## Error Handling and Retries

### Retry Configuration

```python
from dataseter import WebExtractor

config = {
    'retry_attempts': 3,
    'retry_delay': 5,  # seconds
    'retry_backoff': 2,  # exponential backoff factor
    'retry_on_status': [500, 502, 503, 504, 429],
    'retry_on_errors': ['ConnectionError', 'Timeout', 'SSLError']
}

extractor = WebExtractor(config)
```

### Custom Error Handling

```python
from dataseter import WebExtractor
import logging

class RobustExtractor(WebExtractor):
    def extract_with_fallback(self, url):
        strategies = [
            ('javascript', {'javascript_rendering': True}),
            ('static', {'javascript_rendering': False}),
            ('cached', {'use_cache': True}),
        ]

        for strategy_name, config in strategies:
            try:
                logging.info(f"Trying {strategy_name} strategy for {url}")
                self.config.update(config)
                result = self.extract(url)
                if result and result.get('text'):
                    return result
            except Exception as e:
                logging.warning(f"{strategy_name} failed: {e}")
                continue

        logging.error(f"All strategies failed for {url}")
        return None
```

### Connection Pool Management

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataseter import WebExtractor

class PooledExtractor(WebExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.session = self.create_session()

    def create_session(self):
        session = requests.Session()

        # Configure retry strategy
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )

        # Configure connection pool
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=100,
            pool_maxsize=100
        )

        session.mount('http://', adapter)
        session.mount('https://', adapter)

        return session
```

## Best Practices

### 1. Respect Website Policies

```python
# Always check and respect robots.txt
from urllib.robotparser import RobotFileParser

def can_fetch(url, user_agent='Dataseter/1.0'):
    rp = RobotFileParser()
    rp.set_url(url + '/robots.txt')
    rp.read()
    return rp.can_fetch(user_agent, url)

# Check before scraping
if can_fetch('https://example.com/page'):
    # Proceed with scraping
    pass
```

### 2. Use Caching

```python
from dataseter import WebExtractor
import hashlib
import json
import os

class CachedExtractor(WebExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dir = config.get('cache_dir', '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def extract(self, url):
        # Generate cache key
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        # Check cache
        if os.path.exists(cache_file):
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < self.config.get('cache_ttl', 3600):
                with open(cache_file, 'r') as f:
                    return json.load(f)

        # Extract and cache
        result = super().extract(url)
        with open(cache_file, 'w') as f:
            json.dump(result, f)

        return result
```

### 3. Monitor Performance

```python
from dataseter import WebExtractor
import time
from collections import defaultdict

class MonitoredExtractor(WebExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.stats = defaultdict(int)
        self.response_times = []

    def extract(self, url):
        start_time = time.time()

        try:
            result = super().extract(url)
            self.stats['success'] += 1
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            return result
        except Exception as e:
            self.stats['errors'] += 1
            self.stats[f'error_{type(e).__name__}'] += 1
            raise

    def get_statistics(self):
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
        else:
            avg_time = 0

        return {
            'total_requests': self.stats['success'] + self.stats['errors'],
            'successful': self.stats['success'],
            'failed': self.stats['errors'],
            'avg_response_time': avg_time,
            'error_breakdown': {k: v for k, v in self.stats.items() if k.startswith('error_')}
        }
```

### 4. Handle Dynamic Content Properly

```python
from dataseter import WebExtractor

def extract_spa_website(url):
    # For Single Page Applications
    config = {
        'javascript_rendering': True,
        'wait_conditions': [
            # Wait for specific element
            {'type': 'element', 'selector': 'div#content'},
            # Wait for network idle
            {'type': 'network_idle', 'timeout': 2000},
            # Custom JavaScript check
            {'type': 'javascript', 'script': 'return window.APP_READY === true'},
        ],
        'screenshot': True,  # Take screenshot for debugging
        'viewport': {'width': 1920, 'height': 1080}
    }

    extractor = WebExtractor(config)
    return extractor.extract(url)
```

### 5. Data Quality Validation

```python
def validate_extracted_data(result):
    """Validate extracted data quality"""
    issues = []

    # Check text length
    if len(result.get('text', '')) < 100:
        issues.append("Text too short")

    # Check for common extraction failures
    failure_indicators = ['404', 'Not Found', 'Access Denied', 'Please enable JavaScript']
    text_lower = result.get('text', '').lower()
    for indicator in failure_indicators:
        if indicator.lower() in text_lower:
            issues.append(f"Possible extraction failure: {indicator}")

    # Check encoding issues
    if '�' in result.get('text', ''):
        issues.append("Encoding issues detected")

    # Validate metadata
    if not result.get('metadata', {}).get('title'):
        issues.append("Missing page title")

    return len(issues) == 0, issues
```

## Legal and Ethical Considerations

### Legal Guidelines

1. **Check Terms of Service**: Always read and comply with website ToS
2. **Respect Copyright**: Don't republish copyrighted content
3. **Personal Data**: Be careful with PII and comply with GDPR/CCPA
4. **Rate Limiting**: Don't overwhelm servers with requests

### Ethical Scraping

```python
# Ethical scraping configuration
ethical_config = {
    'respect_robots': True,
    'rate_limit': 1.0,  # Max 1 request per second
    'user_agent': 'YourBot/1.0 (contact@example.com)',  # Identify yourself
    'max_pages': 1000,  # Limit scope
    'timeout': 30,  # Don't hang connections
    'max_retries': 2,  # Don't hammer failed endpoints
}
```

### Attribution

Always provide attribution when required:

```python
def add_attribution(dataset, source_url):
    """Add source attribution to dataset"""
    dataset.metadata['source'] = source_url
    dataset.metadata['attribution'] = f"Data sourced from {source_url}"
    dataset.metadata['scraping_date'] = datetime.now().isoformat()
    return dataset
```

## Summary

This guide covered:

- Basic HTML extraction techniques
- JavaScript rendering with Selenium/Playwright
- Recursive crawling strategies
- Advanced scraping patterns
- Rate limiting and politeness
- Authentication methods
- Data extraction strategies
- Error handling and retries
- Best practices and ethics

For more information:
- [PDF Extraction Guide](PDF_EXTRACTION_GUIDE.md)
- [Processing Pipeline Guide](PROCESSING_GUIDE.md)
- [API Reference](API_REFERENCE.md)