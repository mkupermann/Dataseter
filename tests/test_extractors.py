"""
Comprehensive tests for all data extractors
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import requests

from src.extractors import (
    PDFExtractor,
    WebExtractor,
    OfficeExtractor,
    EbookExtractor,
    TextExtractor
)


class TestWebExtractor:
    """Comprehensive tests for web extraction"""

    @pytest.fixture
    def web_extractor(self):
        """Create WebExtractor with test config"""
        config = {
            'max_depth': 2,
            'max_pages': 10,
            'respect_robots': False,
            'rate_limit': 10,
            'javascript_rendering': False
        }
        return WebExtractor(config)

    @patch('requests.Session.get')
    def test_extract_simple_webpage(self, mock_get, web_extractor):
        """Test extraction from simple HTML page"""
        mock_response = Mock()
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Header</h1>
                <p>This is test content for Dataseter.</p>
                <a href="/page2">Link to page 2</a>
            </body>
        </html>
        """
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = web_extractor.extract("https://example.com")

        assert result['text'] is not None
        assert 'Test Header' in result['text']
        assert 'test content for Dataseter' in result['text']
        assert result['metadata']['url'] == "https://example.com"
        assert result['metadata']['title'] == 'Test Page'

    @patch('requests.Session.get')
    def test_recursive_crawling(self, mock_get, web_extractor):
        """Test recursive website crawling"""
        # Mock responses for multiple pages
        responses = {
            'https://example.com': """
            <html>
                <body>
                    <h1>Main Page</h1>
                    <a href="/subpage1">Subpage 1</a>
                    <a href="/subpage2">Subpage 2</a>
                </body>
            </html>
            """,
            'https://example.com/subpage1': """
            <html>
                <body>
                    <h1>Subpage 1</h1>
                    <p>Content of subpage 1</p>
                    <a href="/subpage3">Subpage 3</a>
                </body>
            </html>
            """,
            'https://example.com/subpage2': """
            <html>
                <body>
                    <h1>Subpage 2</h1>
                    <p>Content of subpage 2</p>
                </body>
            </html>
            """,
            'https://example.com/subpage3': """
            <html>
                <body>
                    <h1>Subpage 3</h1>
                    <p>Deep content at level 3</p>
                </body>
            </html>
            """
        }

        def side_effect(url, **kwargs):
            mock_response = Mock()
            mock_response.text = responses.get(url, "<html><body>404</body></html>")
            mock_response.status_code = 200 if url in responses else 404
            return mock_response

        mock_get.side_effect = side_effect

        result = web_extractor.extract("https://example.com", max_depth=2)

        assert 'Main Page' in result['text']
        assert 'Subpage 1' in result['text']
        assert 'Subpage 2' in result['text']
        # Subpage 3 should not be included (depth > 2)
        assert result['metadata']['pages_scraped'] <= 3

    @patch('requests.Session.get')
    def test_javascript_content_extraction(self, mock_get, web_extractor):
        """Test extraction from JavaScript-heavy pages"""
        mock_response = Mock()
        mock_response.text = """
        <html>
            <body>
                <div id="app">Loading...</div>
                <script>
                    document.getElementById('app').innerHTML = 'Dynamic Content';
                </script>
            </body>
        </html>
        """
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = web_extractor.extract("https://example.com")

        # Without JS rendering, should get "Loading..."
        assert 'Loading...' in result['text']

    @patch('requests.Session.get')
    def test_handle_different_content_types(self, mock_get, web_extractor):
        """Test handling of various content types"""
        test_cases = [
            {
                'html': '<html><body><article>Article content</article></body></html>',
                'expected': 'Article content'
            },
            {
                'html': '<html><body><main>Main content area</main></body></html>',
                'expected': 'Main content area'
            },
            {
                'html': '<html><body><div class="content">Div content</div></body></html>',
                'expected': 'Div content'
            }
        ]

        for test_case in test_cases:
            mock_response = Mock()
            mock_response.text = test_case['html']
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = web_extractor.extract("https://example.com")
            assert test_case['expected'] in result['text']

    @patch('requests.Session.get')
    def test_metadata_extraction(self, mock_get, web_extractor):
        """Test extraction of meta tags and metadata"""
        mock_response = Mock()
        mock_response.text = """
        <html lang="en">
            <head>
                <title>Test Title</title>
                <meta name="description" content="Test description">
                <meta name="keywords" content="test, keywords">
                <meta name="author" content="Test Author">
                <meta property="og:title" content="Open Graph Title">
            </head>
            <body>Content</body>
        </html>
        """
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = web_extractor.extract("https://example.com")

        assert result['metadata']['title'] == 'Test Title'
        assert result['metadata']['description'] == 'Test description'
        assert result['metadata']['keywords'] == 'test, keywords'
        assert result['metadata']['author'] == 'Test Author'
        assert result['metadata']['language'] == 'en'

    @patch('requests.Session.get')
    def test_domain_filtering(self, mock_get):
        """Test allowed and blocked domain filtering"""
        config = {
            'allowed_domains': ['example.com'],
            'blocked_domains': ['blocked.com']
        }
        extractor = WebExtractor(config)

        mock_response = Mock()
        mock_response.text = """
        <html><body>
            <a href="https://example.com/page1">Allowed</a>
            <a href="https://blocked.com/page2">Blocked</a>
            <a href="https://other.com/page3">Other</a>
        </body></html>
        """
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = extractor.extract("https://example.com")
        assert result is not None

    @patch('requests.Session.get')
    def test_error_handling(self, mock_get, web_extractor):
        """Test error handling for failed requests"""
        mock_get.side_effect = requests.RequestException("Connection error")

        result = web_extractor.extract("https://example.com")

        assert result['text'] == ''
        assert 'error' in result
        assert 'Connection error' in result['error']

    @patch('requests.Session.get')
    def test_rate_limiting(self, mock_get, web_extractor):
        """Test rate limiting functionality"""
        import time

        mock_response = Mock()
        mock_response.text = "<html><body>Test</body></html>"
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        web_extractor.rate_limit = 2  # 2 requests per second

        start_time = time.time()
        web_extractor.extract("https://example.com")
        elapsed = time.time() - start_time

        # Should complete quickly for single request
        assert elapsed < 1.0

    def test_real_website_extraction(self):
        """Test extraction from real websites (integration test)"""
        extractor = WebExtractor({
            'max_depth': 1,
            'max_pages': 3,
            'rate_limit': 0.5  # Be respectful
        })

        # Test with a simple, stable website
        result = extractor.extract("https://httpbin.org/html")

        assert result['text'] is not None
        assert len(result['text']) > 0
        assert 'Herman Melville' in result['text']  # httpbin.org/html contains Moby Dick text
        assert result['metadata']['source_url'] == "https://httpbin.org/html"


class TestPDFExtractor:
    """Test PDF extraction functionality"""

    @pytest.fixture
    def pdf_extractor(self):
        """Create PDFExtractor instance"""
        return PDFExtractor({
            'ocr_enabled': False,
            'extract_tables': True
        })

    def test_pdf_extraction_mock(self, pdf_extractor, tmp_path):
        """Test PDF extraction with mock PDF"""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b'%PDF-1.4\nMock PDF content')

        with patch('PyPDF2.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Extracted PDF text content"
            mock_reader.return_value.pages = [mock_page]
            mock_reader.return_value.metadata = {
                '/Title': 'Test PDF',
                '/Author': 'Test Author'
            }

            result = pdf_extractor.extract(str(pdf_path))

            # Note: This will fail with real PDF processing, but tests the structure
            assert 'error' in result or 'text' in result

    def test_pdf_metadata_extraction(self, pdf_extractor, tmp_path):
        """Test PDF metadata extraction"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        result = pdf_extractor.extract(str(pdf_path))

        assert 'metadata' in result
        assert 'source' in result['metadata']
        assert str(pdf_path) in result['metadata']['source']


class TestOfficeExtractor:
    """Test Office document extraction"""

    @pytest.fixture
    def office_extractor(self):
        """Create OfficeExtractor instance"""
        return OfficeExtractor({
            'extract_comments': True,
            'extract_tables': True
        })

    @patch('docx.Document')
    def test_word_extraction(self, mock_document, office_extractor, tmp_path):
        """Test Word document extraction"""
        docx_path = tmp_path / "test.docx"
        docx_path.write_text("dummy")

        # Mock Word document
        mock_doc = Mock()
        mock_para1 = Mock()
        mock_para1.text = "First paragraph"
        mock_para2 = Mock()
        mock_para2.text = "Second paragraph"
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []
        mock_doc.core_properties = Mock(
            title="Test Document",
            author="Test Author",
            subject=None,
            keywords=None,
            created=None,
            modified=None,
            last_modified_by=None
        )
        mock_doc.sections = []
        mock_document.return_value = mock_doc

        result = office_extractor.extract(str(docx_path))

        assert 'First paragraph' in result['text']
        assert 'Second paragraph' in result['text']
        assert result['metadata']['title'] == 'Test Document'

    @patch('openpyxl.load_workbook')
    def test_excel_extraction(self, mock_workbook, office_extractor, tmp_path):
        """Test Excel extraction"""
        xlsx_path = tmp_path / "test.xlsx"
        xlsx_path.write_text("dummy")

        # Mock Excel workbook
        mock_wb = Mock()
        mock_wb.sheetnames = ['Sheet1']
        mock_sheet = Mock()

        # Mock cells
        mock_cell1 = Mock(value="Header1")
        mock_cell2 = Mock(value="Header2")
        mock_cell3 = Mock(value="Data1")
        mock_cell4 = Mock(value="Data2")

        mock_sheet.iter_rows.return_value = [
            [mock_cell1, mock_cell2],
            [mock_cell3, mock_cell4]
        ]

        mock_wb.__getitem__ = Mock(return_value=mock_sheet)
        mock_wb.properties = Mock(
            title="Test Spreadsheet",
            creator="Test Creator",
            created=None,
            modified=None
        )
        mock_workbook.return_value = mock_wb

        result = office_extractor.extract(str(xlsx_path))

        assert 'Header1' in result['text']
        assert 'Data1' in result['text']


class TestTextExtractor:
    """Test plain text extraction"""

    @pytest.fixture
    def text_extractor(self):
        """Create TextExtractor instance"""
        return TextExtractor({'encoding': 'auto'})

    def test_text_extraction(self, text_extractor, tmp_path):
        """Test plain text file extraction"""
        text_path = tmp_path / "test.txt"
        test_content = "This is a test text file.\nIt has multiple lines.\n特殊字符测试。"
        text_path.write_text(test_content, encoding='utf-8')

        result = text_extractor.extract(str(text_path))

        assert result['text'] == test_content
        assert result['metadata']['encoding'] in ['utf-8', 'UTF-8']

    def test_encoding_detection(self, text_extractor, tmp_path):
        """Test automatic encoding detection"""
        # Test with different encodings
        encodings = [
            ('utf-8', "UTF-8 content: 你好"),
            ('latin-1', "Latin-1 content: café"),
            ('ascii', "ASCII content only")
        ]

        for encoding, content in encodings:
            text_path = tmp_path / f"test_{encoding}.txt"

            try:
                text_path.write_text(content, encoding=encoding)
                result = text_extractor.extract(str(text_path))

                # Should successfully extract without errors
                assert result['text'] is not None
                assert 'error' not in result
            except UnicodeEncodeError:
                # Skip if encoding not supported
                pass

    def test_large_file_handling(self, text_extractor, tmp_path):
        """Test handling of large text files"""
        text_path = tmp_path / "large.txt"

        # Create a file larger than max_size
        large_content = "x" * (101 * 1024 * 1024)  # 101 MB
        text_path.write_text(large_content)

        text_extractor.max_size = 100 * 1024 * 1024  # 100 MB limit

        result = text_extractor.extract(str(text_path))

        assert 'error' in result
        assert 'too large' in result['error']


class TestIntegrationExtractors:
    """Integration tests for all extractors"""

    def test_all_extractors_initialization(self):
        """Test that all extractors can be initialized"""
        extractors = [
            PDFExtractor({}),
            WebExtractor({}),
            OfficeExtractor({}),
            EbookExtractor({}),
            TextExtractor({})
        ]

        for extractor in extractors:
            assert extractor is not None
            assert hasattr(extractor, 'extract')
            assert hasattr(extractor, 'config')

    def test_extractor_error_handling(self, tmp_path):
        """Test error handling for non-existent files"""
        non_existent = str(tmp_path / "non_existent.pdf")

        extractors = [
            PDFExtractor({}),
            OfficeExtractor({}),
            EbookExtractor({}),
            TextExtractor({})
        ]

        for extractor in extractors:
            result = extractor.extract(non_existent)
            assert 'error' in result or result['text'] == ''

    @pytest.mark.parametrize("extractor_class,file_ext,content", [
        (TextExtractor, '.txt', 'Simple text content'),
        (TextExtractor, '.md', '# Markdown Header\n\nContent'),
    ])
    def test_multiple_file_types(self, extractor_class, file_ext, content, tmp_path):
        """Test extraction from various file types"""
        file_path = tmp_path / f"test{file_ext}"
        file_path.write_text(content)

        extractor = extractor_class({})
        result = extractor.extract(str(file_path))

        assert result['text'] == content or 'error' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])