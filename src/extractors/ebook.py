"""
eBook extraction module (EPUB, MOBI, AZW3)
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import html2text

try:
    import ebooklib
    from ebooklib import epub
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False

try:
    import mobi
    MOBI_AVAILABLE = True
except ImportError:
    MOBI_AVAILABLE = False

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class EbookExtractor(BaseExtractor):
    """Extract text and metadata from eBooks"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.extract_metadata = config.get('extract_metadata', True)
        self.extract_toc = config.get('extract_toc', True)
        self.preserve_chapter_structure = config.get('preserve_chapter_structure', True)
        self.convert_to_text = config.get('convert_to_text', True)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True

    def extract(self, source: str, **kwargs) -> Dict[str, Any]:
        if not self.validate_source(source):
            return {'text': '', 'metadata': {}, 'error': f'Invalid source: {source}'}

        file_ext = Path(source).suffix.lower()

        try:
            if file_ext == '.epub':
                result = self._extract_epub(source, **kwargs)
            elif file_ext in ['.mobi', '.azw', '.azw3']:
                result = self._extract_mobi(source, **kwargs)
            else:
                return {'text': '', 'metadata': {}, 'error': f'Unsupported format: {file_ext}'}

            self.update_stats(True, Path(source).stat().st_size)
            return result

        except Exception as e:
            logger.error(f"Error extracting ebook {source}: {e}")
            self.update_stats(False)
            return {'text': '', 'metadata': self.extract_metadata(source), 'error': str(e)}

    def _extract_epub(self, source: str, **kwargs) -> Dict[str, Any]:
        if not EBOOKLIB_AVAILABLE:
            return {'text': '', 'metadata': {}, 'error': 'ebooklib not installed'}

        book = epub.read_epub(source)
        text_parts = []
        metadata = self.extract_metadata(source)
        chapters = []

        # Extract metadata
        if self.extract_metadata:
            metadata.update({
                'title': book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else None,
                'author': book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else None,
                'language': book.get_metadata('DC', 'language')[0][0] if book.get_metadata('DC', 'language') else None,
                'publisher': book.get_metadata('DC', 'publisher')[0][0] if book.get_metadata('DC', 'publisher') else None,
            })

        # Extract chapters
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_body_content().decode('utf-8')
                text = self.h2t.handle(content) if self.convert_to_text else content

                chapter_data = {
                    'title': item.get_name(),
                    'text': text
                }
                chapters.append(chapter_data)

                if self.preserve_chapter_structure:
                    text_parts.append(f"\n[Chapter: {item.get_name()}]\n{text}")
                else:
                    text_parts.append(text)

        return {
            'text': '\n'.join(text_parts),
            'metadata': metadata,
            'chapters': chapters if self.preserve_chapter_structure else None
        }

    def _extract_mobi(self, source: str, **kwargs) -> Dict[str, Any]:
        if not MOBI_AVAILABLE:
            return {'text': '', 'metadata': {}, 'error': 'mobi library not installed'}

        tempdir, filepath = mobi.extract(source)
        text = open(filepath, 'r', encoding='utf-8').read()

        metadata = self.extract_metadata(source)

        return {
            'text': text,
            'metadata': metadata
        }