"""
PDF extraction module with OCR support
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import shutil

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from .base import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """Extract text and metadata from PDF files"""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PDF extractor

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.ocr_enabled = config.get('ocr_enabled', False) and OCR_AVAILABLE
        self.ocr_language = config.get('ocr_language', 'eng')
        self.extract_tables = config.get('extract_tables', True)
        self.extract_images = config.get('extract_images', False)
        self.preserve_layout = config.get('preserve_layout', False)
        self.max_pages = config.get('max_pages')
        self.dpi = config.get('dpi', 300)

        # Select best available PDF library
        if fitz:
            self.backend = 'pymupdf'
        elif pdfplumber:
            self.backend = 'pdfplumber'
        elif PyPDF2:
            self.backend = 'pypdf2'
        else:
            raise ImportError(
                "No PDF library available. Install PyPDF2, pdfplumber, or PyMuPDF"
            )

        logger.info(f"PDF extractor initialized with backend: {self.backend}")

    def extract(self, source: str, **kwargs) -> Dict[str, Any]:
        """Extract text and metadata from PDF file

        Args:
            source: Path to PDF file
            **kwargs: Additional extraction options

        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.validate_source(source):
            return {
                'text': '',
                'metadata': {},
                'error': f'Invalid source: {source}'
            }

        try:
            # Extract based on backend
            if self.backend == 'pymupdf':
                result = self._extract_pymupdf(source, **kwargs)
            elif self.backend == 'pdfplumber':
                result = self._extract_pdfplumber(source, **kwargs)
            else:
                result = self._extract_pypdf2(source, **kwargs)

            # Try OCR if enabled and text extraction failed
            if self.ocr_enabled and len(result.get('text', '').strip()) < 100:
                logger.info(f"Text extraction yielded little content, trying OCR for {source}")
                ocr_result = self._extract_with_ocr(source, **kwargs)
                if len(ocr_result.get('text', '')) > len(result.get('text', '')):
                    result = ocr_result

            # Update statistics
            self.update_stats(
                success=True,
                bytes_processed=Path(source).stat().st_size
            )

            return result

        except Exception as e:
            logger.error(f"Error extracting PDF {source}: {e}")
            self.update_stats(success=False)
            return {
                'text': '',
                'metadata': self.extract_metadata(source),
                'error': str(e)
            }

    def _extract_pymupdf(self, source: str, **kwargs) -> Dict[str, Any]:
        """Extract using PyMuPDF"""
        import fitz

        text_parts = []
        metadata = self.extract_metadata(source)
        tables = []

        with fitz.open(source) as pdf:
            # Extract document metadata
            pdf_metadata = pdf.metadata
            if pdf_metadata:
                metadata.update({
                    'title': pdf_metadata.get('title'),
                    'author': pdf_metadata.get('author'),
                    'subject': pdf_metadata.get('subject'),
                    'keywords': pdf_metadata.get('keywords'),
                    'creator': pdf_metadata.get('creator'),
                    'producer': pdf_metadata.get('producer'),
                })

            metadata['page_count'] = len(pdf)

            # Extract text from each page
            max_pages = self.max_pages or len(pdf)
            for page_num in range(min(max_pages, len(pdf))):
                page = pdf[page_num]

                # Extract text
                if self.preserve_layout:
                    text = page.get_text("text")
                else:
                    text = page.get_text()

                text_parts.append(text)

                # Extract tables if enabled
                if self.extract_tables:
                    tabs = page.find_tables()
                    for tab in tabs:
                        tables.append({
                            'page': page_num + 1,
                            'data': tab.extract()
                        })

        return {
            'text': '\n'.join(text_parts),
            'metadata': metadata,
            'tables': tables if tables else None
        }

    def _extract_pdfplumber(self, source: str, **kwargs) -> Dict[str, Any]:
        """Extract using pdfplumber"""
        import pdfplumber

        text_parts = []
        metadata = self.extract_metadata(source)
        tables = []

        with pdfplumber.open(source) as pdf:
            # Extract metadata
            pdf_metadata = pdf.metadata
            if pdf_metadata:
                metadata.update({
                    'title': pdf_metadata.get('Title'),
                    'author': pdf_metadata.get('Author'),
                    'subject': pdf_metadata.get('Subject'),
                    'creator': pdf_metadata.get('Creator'),
                })

            metadata['page_count'] = len(pdf.pages)

            # Extract text from each page
            max_pages = self.max_pages or len(pdf.pages)
            for i in range(min(max_pages, len(pdf.pages))):
                page = pdf.pages[i]

                # Extract text
                text = page.extract_text()
                if text:
                    text_parts.append(text)

                # Extract tables if enabled
                if self.extract_tables:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        tables.append({
                            'page': i + 1,
                            'data': table
                        })

        return {
            'text': '\n'.join(text_parts),
            'metadata': metadata,
            'tables': tables if tables else None
        }

    def _extract_pypdf2(self, source: str, **kwargs) -> Dict[str, Any]:
        """Extract using PyPDF2"""
        import PyPDF2

        text_parts = []
        metadata = self.extract_metadata(source)

        with open(source, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Extract metadata
            pdf_metadata = pdf_reader.metadata
            if pdf_metadata:
                metadata.update({
                    'title': pdf_metadata.get('/Title'),
                    'author': pdf_metadata.get('/Author'),
                    'subject': pdf_metadata.get('/Subject'),
                    'creator': pdf_metadata.get('/Creator'),
                })

            metadata['page_count'] = len(pdf_reader.pages)

            # Extract text from each page
            max_pages = self.max_pages or len(pdf_reader.pages)
            for i in range(min(max_pages, len(pdf_reader.pages))):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                text_parts.append(text)

        return {
            'text': '\n'.join(text_parts),
            'metadata': metadata,
            'tables': None  # PyPDF2 doesn't support table extraction
        }

    def _extract_with_ocr(self, source: str, **kwargs) -> Dict[str, Any]:
        """Extract text using OCR"""
        if not OCR_AVAILABLE:
            logger.warning("OCR libraries not available")
            return {'text': '', 'metadata': {}}

        text_parts = []
        metadata = self.extract_metadata(source)

        try:
            # Convert PDF to images
            images = convert_from_path(
                source,
                dpi=self.dpi,
                first_page=1,
                last_page=self.max_pages
            )

            metadata['page_count'] = len(images)
            metadata['extraction_method'] = 'OCR'

            # Extract text from each image
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(
                    image,
                    lang=self.ocr_language
                )
                text_parts.append(text)
                logger.debug(f"OCR extracted {len(text)} characters from page {i+1}")

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {'text': '', 'metadata': metadata, 'error': str(e)}

        return {
            'text': '\n'.join(text_parts),
            'metadata': metadata
        }

    def extract_batch(self, sources: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Extract from multiple PDF files

        Args:
            sources: List of PDF file paths
            **kwargs: Additional extraction options

        Returns:
            List of extraction results
        """
        results = []
        for source in sources:
            result = self.extract(source, **kwargs)
            results.append(result)
        return results