"""
Plain text extraction module
"""

import logging
import chardet
from typing import Dict, Any
from pathlib import Path

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class TextExtractor(BaseExtractor):
    """Extract text from plain text files"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.encoding = config.get('encoding', 'auto')
        self.max_size = config.get('max_size', 100 * 1024 * 1024)  # 100MB

    def extract(self, source: str, **kwargs) -> Dict[str, Any]:
        if not self.validate_source(source):
            return {'text': '', 'metadata': {}, 'error': f'Invalid source: {source}'}

        try:
            file_size = Path(source).stat().st_size
            if file_size > self.max_size:
                return {'text': '', 'metadata': {}, 'error': f'File too large: {file_size} bytes'}

            # Detect encoding if auto
            if self.encoding == 'auto':
                with open(source, 'rb') as f:
                    raw = f.read(10000)
                    result = chardet.detect(raw)
                    encoding = result['encoding'] or 'utf-8'
            else:
                encoding = self.encoding

            # Read text
            with open(source, 'r', encoding=encoding, errors='ignore') as f:
                text = f.read()

            metadata = self.extract_metadata(source)
            metadata['encoding'] = encoding

            self.update_stats(True, file_size)

            return {
                'text': text,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error extracting text from {source}: {e}")
            self.update_stats(False)
            return {'text': '', 'metadata': self.extract_metadata(source), 'error': str(e)}