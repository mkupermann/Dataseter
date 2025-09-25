"""
Web scraping module with recursive crawling support
"""

import logging
import time
import re
from typing import Dict, Any, Optional, List, Set
from urllib.parse import urljoin, urlparse
from collections import deque
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class WebExtractor(BaseExtractor):
    """Extract text and data from websites with recursive crawling"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize web extractor

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_depth = config.get('max_depth', 3)
        self.max_pages = config.get('max_pages', 1000)
        self.respect_robots = config.get('respect_robots', True)
        self.user_agent = config.get('user_agent', 'Dataseter/1.0')
        self.javascript_rendering = config.get('javascript_rendering', False)
        self.rate_limit = config.get('rate_limit', 1.0)  # requests per second
        self.timeout = config.get('timeout', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.follow_redirects = config.get('follow_redirects', True)
        self.allowed_domains = set(config.get('allowed_domains', []))
        self.blocked_domains = set(config.get('blocked_domains', []))

        # Setup session with retry strategy
        self.session = self._setup_session()

        # Track visited URLs
        self.visited_urls = set()
        self.url_queue = deque()

    def _setup_session(self) -> requests.Session:
        """Setup requests session with retry strategy"""
        session = requests.Session()
        retry = Retry(
            total=self.retry_attempts,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({'User-Agent': self.user_agent})
        return session

    def extract(self, source: str, **kwargs) -> Dict[str, Any]:
        """Extract text and metadata from website

        Args:
            source: URL to scrape
            **kwargs: Additional extraction options

        Returns:
            Dictionary with extracted text and metadata
        """
        max_depth = kwargs.get('max_depth', self.max_depth)

        try:
            # Reset tracking for new extraction
            self.visited_urls.clear()
            self.url_queue.clear()
            self.url_queue.append((source, 0))

            all_text = []
            all_metadata = []
            pages_processed = 0

            while self.url_queue and pages_processed < self.max_pages:
                url, depth = self.url_queue.popleft()

                if url in self.visited_urls or depth > max_depth:
                    continue

                # Check domain restrictions
                if not self._is_allowed_domain(url):
                    logger.debug(f"Skipping blocked/unallowed domain: {url}")
                    continue

                # Rate limiting
                if pages_processed > 0:
                    time.sleep(1.0 / self.rate_limit)

                # Extract page content
                if self.javascript_rendering and (SELENIUM_AVAILABLE or PLAYWRIGHT_AVAILABLE):
                    content = self._extract_with_javascript(url)
                else:
                    content = self._extract_static(url)

                if content:
                    all_text.append(content['text'])
                    all_metadata.append(content['metadata'])
                    self.visited_urls.add(url)
                    pages_processed += 1

                    # Find and queue links if not at max depth
                    if depth < max_depth:
                        links = self._extract_links(content.get('html', ''), url)
                        for link in links:
                            if link not in self.visited_urls:
                                self.url_queue.append((link, depth + 1))

                    logger.info(f"Processed {url} (depth={depth}, pages={pages_processed})")

            # Combine results
            combined_text = '\n\n'.join(all_text)
            combined_metadata = {
                'source_url': source,
                'pages_scraped': pages_processed,
                'max_depth': max_depth,
                'total_links_found': len(self.visited_urls),
                'page_metadata': all_metadata
            }

            self.update_stats(success=True, bytes_processed=len(combined_text))

            return {
                'text': combined_text,
                'metadata': combined_metadata
            }

        except Exception as e:
            logger.error(f"Error extracting from {source}: {e}")
            self.update_stats(success=False)
            return {
                'text': '',
                'metadata': {'source_url': source},
                'error': str(e)
            }

    def _extract_static(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from static HTML"""
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=self.follow_redirects
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Extract metadata
            metadata = {
                'url': url,
                'title': soup.title.string if soup.title else None,
                'description': self._get_meta_content(soup, 'description'),
                'keywords': self._get_meta_content(soup, 'keywords'),
                'author': self._get_meta_content(soup, 'author'),
                'language': soup.html.get('lang') if soup.html else None,
                'status_code': response.status_code,
                'content_length': len(response.content)
            }

            return {
                'text': text,
                'html': response.text,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error extracting static content from {url}: {e}")
            return None

    def _extract_with_javascript(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from JavaScript-rendered pages"""
        if PLAYWRIGHT_AVAILABLE:
            return asyncio.run(self._extract_with_playwright(url))
        elif SELENIUM_AVAILABLE:
            return self._extract_with_selenium(url)
        else:
            logger.warning("JavaScript rendering requested but no library available")
            return self._extract_static(url)

    def _extract_with_selenium(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract using Selenium"""
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={self.user_agent}')

            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(self.timeout)

            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # Get page source after JavaScript execution
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()

                text = soup.get_text(separator='\n', strip=True)

                metadata = {
                    'url': url,
                    'title': driver.title,
                    'description': self._get_meta_content(soup, 'description'),
                    'keywords': self._get_meta_content(soup, 'keywords'),
                    'rendered_with': 'selenium'
                }

                return {
                    'text': text,
                    'html': html,
                    'metadata': metadata
                }

            finally:
                driver.quit()

        except Exception as e:
            logger.error(f"Error extracting with Selenium from {url}: {e}")
            return None

    async def _extract_with_playwright(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract using Playwright"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=self.user_agent
                )
                page = await context.new_page()

                await page.goto(url, wait_until='networkidle', timeout=self.timeout * 1000)
                await page.wait_for_selector('body')

                # Get content
                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')

                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                    element.decompose()

                text = soup.get_text(separator='\n', strip=True)

                metadata = {
                    'url': url,
                    'title': await page.title(),
                    'description': self._get_meta_content(soup, 'description'),
                    'keywords': self._get_meta_content(soup, 'keywords'),
                    'rendered_with': 'playwright'
                }

                await browser.close()

                return {
                    'text': text,
                    'html': html,
                    'metadata': metadata
                }

        except Exception as e:
            logger.error(f"Error extracting with Playwright from {url}: {e}")
            return None

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content"""
        links = []
        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup.find_all(['a', 'link']):
            href = tag.get('href')
            if href:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)

                # Filter out non-HTTP(S) links
                if absolute_url.startswith(('http://', 'https://')):
                    # Remove fragment identifiers
                    absolute_url = absolute_url.split('#')[0]
                    links.append(absolute_url)

        return list(set(links))  # Remove duplicates

    def _is_allowed_domain(self, url: str) -> bool:
        """Check if domain is allowed"""
        domain = urlparse(url).netloc

        # Check blocked domains
        if domain in self.blocked_domains:
            return False

        # If allowed domains specified, check membership
        if self.allowed_domains:
            return domain in self.allowed_domains

        return True

    def _get_meta_content(self, soup: BeautifulSoup, name: str) -> Optional[str]:
        """Extract meta tag content"""
        meta = soup.find('meta', attrs={'name': name})
        if meta:
            return meta.get('content')

        # Try property attribute (for Open Graph tags)
        meta = soup.find('meta', attrs={'property': f'og:{name}'})
        if meta:
            return meta.get('content')

        return None

    def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        if not self.respect_robots:
            return True

        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

            response = self.session.get(robots_url, timeout=5)
            if response.status_code == 200:
                # Simple robots.txt parser (basic implementation)
                lines = response.text.split('\n')
                user_agent_match = False

                for line in lines:
                    line = line.strip()
                    if line.startswith('User-agent:'):
                        agent = line.split(':', 1)[1].strip()
                        user_agent_match = (agent == '*' or agent.lower() in self.user_agent.lower())
                    elif user_agent_match and line.startswith('Disallow:'):
                        path = line.split(':', 1)[1].strip()
                        if path and url.startswith(urljoin(url, path)):
                            return False

            return True

        except Exception:
            # If can't fetch robots.txt, assume allowed
            return True