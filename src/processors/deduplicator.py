"""
Deduplication module for removing duplicate content
"""

import hashlib
import logging
from typing import Dict, Any, Set, List
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class Deduplicator:
    """Remove duplicate content from dataset"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold = config.get('duplicate_threshold', 0.95)
        self.method = config.get('method', 'hash')  # hash, similarity
        self.seen_hashes = set()
        self.seen_texts = []

    def process(self, document: Any, **kwargs) -> Any:
        """Remove duplicates from document"""
        if hasattr(document, 'chunks'):
            unique_chunks = []
            for chunk in document.chunks:
                if 'text' in chunk and not self.is_duplicate(chunk['text']):
                    unique_chunks.append(chunk)
            document.chunks = unique_chunks
        elif hasattr(document, 'text'):
            if self.is_duplicate(document.text):
                return None
        return document

    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate"""
        if self.method == 'hash':
            return self._is_duplicate_hash(text)
        else:
            return self._is_duplicate_similarity(text)

    def _is_duplicate_hash(self, text: str) -> bool:
        """Check duplicate using hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def _is_duplicate_similarity(self, text: str) -> bool:
        """Check duplicate using similarity"""
        for seen_text in self.seen_texts:
            similarity = SequenceMatcher(None, text, seen_text).ratio()
            if similarity >= self.threshold:
                return True
        self.seen_texts.append(text)
        return False

    def reset(self):
        """Reset deduplication state"""
        self.seen_hashes.clear()
        self.seen_texts.clear()