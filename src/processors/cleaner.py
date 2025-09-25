"""
Text cleaning and normalization module
"""

import re
import logging
from typing import Dict, Any
import ftfy
import unicodedata

logger = logging.getLogger(__name__)


class Cleaner:
    """Text cleaning and normalization"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lowercase = config.get('lowercase', False)
        self.remove_html = config.get('remove_html', True)
        self.remove_urls = config.get('remove_urls', True)
        self.remove_emails = config.get('remove_emails', True)
        self.fix_unicode = config.get('fix_unicode', True)
        self.remove_extra_whitespace = config.get('remove_extra_whitespace', True)
        self.remove_special_chars = config.get('remove_special_chars', False)

    def process(self, document: Any, **kwargs) -> Any:
        """Clean document text"""
        if hasattr(document, 'text'):
            document.text = self.clean_text(document.text, **kwargs)
        if hasattr(document, 'chunks'):
            for chunk in document.chunks:
                if 'text' in chunk:
                    chunk['text'] = self.clean_text(chunk['text'], **kwargs)
        return document

    def clean_text(self, text: str, **kwargs) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Fix unicode issues
        if self.fix_unicode:
            text = ftfy.fix_text(text)
            text = unicodedata.normalize('NFKC', text)

        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<[^>]+>', ' ', text)

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', '\n', text)
            text = text.strip()

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        return text