"""
Comprehensive tests for all text processors
"""

import pytest
from unittest.mock import Mock, patch
import hashlib

from src.processors import (
    Chunker,
    Cleaner,
    Deduplicator,
    PrivacyProtector,
    QualityFilter
)


class TestChunker:
    """Test text chunking functionality"""

    @pytest.fixture
    def chunker(self):
        return Chunker({
            'strategy': 'sliding_window',
            'size': 10,
            'overlap': 2
        })

    def test_sliding_window_chunking(self, chunker):
        """Test sliding window chunking with overlap"""
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = chunker.chunk_text(text, strategy='sliding_window', size=10, overlap=2)

        assert len(chunks) > 0
        # Check overlap exists
        for i in range(len(chunks) - 1):
            assert chunks[i]['end_index'] > chunks[i+1]['start_index']

    def test_fixed_chunking(self, chunker):
        """Test fixed size chunking without overlap"""
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = chunker.chunk_text(text, strategy='fixed', size=10)

        assert len(chunks) == 5  # 50 words / 10 words per chunk
        # Check no overlap
        for i in range(len(chunks) - 1):
            assert chunks[i]['end_index'] == chunks[i+1]['start_index']

    def test_sentence_chunking(self, chunker):
        """Test sentence-based chunking"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk_text(text, strategy='sentence', size=10)

        assert len(chunks) > 0
        for chunk in chunks:
            assert 'sentence_count' in chunk
            assert chunk['sentence_count'] > 0

    def test_paragraph_chunking(self, chunker):
        """Test paragraph-based chunking"""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk_text(text, strategy='paragraph', size=20)

        assert len(chunks) > 0
        for chunk in chunks:
            assert 'paragraph_count' in chunk

    def test_semantic_chunking(self, chunker):
        """Test semantic chunking"""
        text = """
# Header 1
Content under header 1.

# Header 2
Content under header 2.

## Subheader
More detailed content.
"""
        chunks = chunker.chunk_text(text, strategy='semantic', size=50)

        assert len(chunks) > 0
        for chunk in chunks:
            assert 'type' in chunk or 'text' in chunk

    def test_minimum_chunk_size(self):
        """Test minimum chunk size filtering"""
        chunker = Chunker({'min_chunk_size': 20})
        text = "Short"  # Less than min size

        chunks = chunker.chunk_text(text, strategy='fixed', size=10)

        assert len(chunks) == 0  # Should filter out small chunks

    def test_document_processing(self, chunker):
        """Test processing document objects"""
        doc = Mock()
        doc.text = " ".join([f"word{i}" for i in range(30)])

        processed = chunker.process(doc, size=10)

        assert hasattr(processed, 'chunks')
        assert len(processed.chunks) > 0


class TestCleaner:
    """Test text cleaning functionality"""

    @pytest.fixture
    def cleaner(self):
        return Cleaner({
            'lowercase': False,
            'remove_html': True,
            'remove_urls': True,
            'remove_emails': True,
            'fix_unicode': True,
            'remove_extra_whitespace': True
        })

    def test_html_removal(self, cleaner):
        """Test HTML tag removal"""
        text = "<p>This is <b>HTML</b> content</p>"
        cleaned = cleaner.clean_text(text)

        assert '<p>' not in cleaned
        assert '<b>' not in cleaned
        assert 'This is' in cleaned
        assert 'HTML' in cleaned
        assert 'content' in cleaned

    def test_url_removal(self, cleaner):
        """Test URL removal"""
        text = "Visit https://example.com or http://test.org for more"
        cleaned = cleaner.clean_text(text)

        assert 'https://example.com' not in cleaned
        assert 'http://test.org' not in cleaned
        assert 'Visit' in cleaned
        assert 'for more' in cleaned

    def test_email_removal(self, cleaner):
        """Test email removal"""
        text = "Contact us at test@example.com or admin@site.org"
        cleaned = cleaner.clean_text(text)

        assert 'test@example.com' not in cleaned
        assert 'admin@site.org' not in cleaned
        assert 'Contact us at' in cleaned

    def test_whitespace_normalization(self, cleaner):
        """Test extra whitespace removal"""
        text = "Too    many     spaces\n\n\nand lines"
        cleaned = cleaner.clean_text(text)

        assert cleaned == "Too many spaces\nand lines"

    def test_lowercase_conversion(self):
        """Test lowercase conversion"""
        cleaner = Cleaner({'lowercase': True})
        text = "UPPER and Lower CaSe"
        cleaned = cleaner.clean_text(text)

        assert cleaned == "upper and lower case"

    def test_special_chars_removal(self):
        """Test special character removal"""
        cleaner = Cleaner({'remove_special_chars': True})
        text = "Hello! @#$% World?"
        cleaned = cleaner.clean_text(text)

        assert '@' not in cleaned
        assert '#' not in cleaned
        assert '$' not in cleaned

    def test_unicode_fixing(self, cleaner):
        """Test Unicode normalization"""
        text = "Broken — unicode – text"
        cleaned = cleaner.clean_text(text)

        assert cleaned is not None
        # Text should be normalized


class TestDeduplicator:
    """Test deduplication functionality"""

    @pytest.fixture
    def deduplicator(self):
        return Deduplicator({
            'duplicate_threshold': 0.9,
            'method': 'hash'
        })

    def test_hash_deduplication(self, deduplicator):
        """Test hash-based deduplication"""
        texts = [
            "Unique text 1",
            "Unique text 2",
            "Unique text 1",  # Duplicate
            "Unique text 3"
        ]

        seen = []
        for text in texts:
            if not deduplicator.is_duplicate(text):
                seen.append(text)

        assert len(seen) == 3  # One duplicate removed
        assert "Unique text 1" in seen
        assert "Unique text 2" in seen
        assert "Unique text 3" in seen

    def test_similarity_deduplication(self):
        """Test similarity-based deduplication"""
        dedup = Deduplicator({
            'method': 'similarity',
            'duplicate_threshold': 0.8
        })

        texts = [
            "This is a test sentence.",
            "This is a test sentence!",  # Very similar
            "Completely different text here",
            "This is a test sentenc"  # Similar with typo
        ]

        seen = []
        for text in texts:
            if not dedup.is_duplicate(text):
                seen.append(text)

        assert len(seen) < 4  # Some duplicates removed

    def test_document_deduplication(self, deduplicator):
        """Test document-level deduplication"""
        doc = Mock()
        doc.chunks = [
            {'text': 'Unique chunk 1'},
            {'text': 'Unique chunk 2'},
            {'text': 'Unique chunk 1'},  # Duplicate
        ]

        processed = deduplicator.process(doc)

        assert len(processed.chunks) == 2

    def test_reset_functionality(self, deduplicator):
        """Test resetting deduplicator state"""
        deduplicator.is_duplicate("Test text")
        assert len(deduplicator.seen_hashes) == 1

        deduplicator.reset()
        assert len(deduplicator.seen_hashes) == 0


class TestPrivacyProtector:
    """Test PII detection and redaction"""

    @pytest.fixture
    def privacy_protector(self):
        return PrivacyProtector({
            'detect_pii': True,
            'redaction_method': 'mask'
        })

    def test_email_redaction(self, privacy_protector):
        """Test email address redaction"""
        text = "Contact John at john.doe@example.com for details"
        protected = privacy_protector._protect_with_regex(text)

        assert 'john.doe@example.com' not in protected
        assert '[EMAIL]' in protected

    def test_phone_redaction(self, privacy_protector):
        """Test phone number redaction"""
        text = "Call me at 555-123-4567 or 555.987.6543"
        protected = privacy_protector._protect_with_regex(text)

        assert '555-123-4567' not in protected
        assert '555.987.6543' not in protected
        assert '[PHONE]' in protected

    def test_ssn_redaction(self, privacy_protector):
        """Test SSN redaction"""
        text = "SSN: 123-45-6789"
        protected = privacy_protector._protect_with_regex(text)

        assert '123-45-6789' not in protected
        assert '[SSN]' in protected

    def test_credit_card_redaction(self, privacy_protector):
        """Test credit card number redaction"""
        text = "Card: 4111 1111 1111 1111"
        protected = privacy_protector._protect_with_regex(text)

        assert '4111 1111 1111 1111' not in protected
        assert '[CREDIT_CARD]' in protected

    def test_ip_address_redaction(self, privacy_protector):
        """Test IP address redaction"""
        text = "Server IP: 192.168.1.1"
        protected = privacy_protector._protect_with_regex(text)

        assert '192.168.1.1' not in protected
        assert '[IP_ADDRESS]' in protected

    def test_hash_redaction(self):
        """Test hash-based redaction"""
        protector = PrivacyProtector({
            'detect_pii': True,
            'redaction_method': 'hash'
        })

        text = "Email: test@example.com"
        protected = protector._protect_with_regex(text)

        assert 'test@example.com' not in protected
        # Should contain hashed version
        assert 'EMAIL_' in protected

    def test_removal_redaction(self):
        """Test complete removal redaction"""
        protector = PrivacyProtector({
            'detect_pii': True,
            'redaction_method': 'remove'
        })

        text = "Contact at test@example.com today"
        protected = protector._protect_with_regex(text)

        assert 'test@example.com' not in protected
        assert 'Contact at  today' in protected or 'Contact at today' in protected


class TestQualityFilter:
    """Test quality filtering and scoring"""

    @pytest.fixture
    def quality_filter(self):
        return QualityFilter({
            'min_score': 0.7,
            'detect_language': True,
            'allowed_languages': ['en'],
            'min_word_count': 10,
            'max_word_count': 1000
        })

    def test_quality_score_calculation(self, quality_filter):
        """Test quality score calculation"""
        # Good quality text
        good_text = " ".join(["word" for _ in range(50)])
        score = quality_filter.calculate_quality_score(good_text)
        assert score > 0.5

        # Poor quality text (too short)
        poor_text = "Too short"
        score = quality_filter.calculate_quality_score(poor_text)
        assert score < 0.5

        # Empty text
        score = quality_filter.calculate_quality_score("")
        assert score == 0.0

    def test_word_count_filtering(self, quality_filter):
        """Test filtering based on word count"""
        # Too few words
        short_text = "Only five words here now"
        score = quality_filter.calculate_quality_score(short_text)
        assert score < 1.0

        # Good word count
        good_text = " ".join(["word" for _ in range(50)])
        score = quality_filter.calculate_quality_score(good_text)
        assert score > 0.5

    def test_repetition_detection(self, quality_filter):
        """Test detection of repetitive content"""
        # Highly repetitive text
        repetitive = " ".join(["repeat"] * 100)
        score = quality_filter.calculate_quality_score(repetitive)

        # Non-repetitive text
        varied = " ".join([f"word{i}" for i in range(50)])
        varied_score = quality_filter.calculate_quality_score(varied)

        assert varied_score > score  # Varied text should score higher

    def test_special_char_ratio(self, quality_filter):
        """Test special character ratio scoring"""
        # Too many special chars
        special = "###!!!@@@###!!!@@@" * 10
        score = quality_filter.calculate_quality_score(special)

        # Normal text
        normal = "This is normal text with proper punctuation."
        normal_score = quality_filter.calculate_quality_score(normal)

        assert normal_score > score

    def test_uppercase_ratio(self, quality_filter):
        """Test uppercase ratio scoring"""
        # Too much uppercase
        uppercase = "THIS IS ALL UPPERCASE TEXT"
        score = quality_filter.calculate_quality_score(uppercase)

        # Normal case
        normal = "This is normal cased text"
        normal_score = quality_filter.calculate_quality_score(normal)

        assert normal_score > score

    @patch('langdetect.detect')
    def test_language_detection(self, mock_detect, quality_filter):
        """Test language detection and filtering"""
        mock_detect.return_value = 'en'

        doc = Mock()
        doc.text = "This is English text that should pass the filter"

        processed = quality_filter.process(doc)
        assert processed is not None
        assert processed.language == 'en'

        # Test non-allowed language
        mock_detect.return_value = 'fr'
        doc.text = "Ceci est du texte français"

        processed = quality_filter.process(doc)
        assert processed is None  # Filtered out

    def test_document_filtering(self, quality_filter):
        """Test document-level quality filtering"""
        doc = Mock()
        doc.text = "Short"  # Too short, should be filtered

        processed = quality_filter.process(doc)
        assert processed is None  # Should be filtered out

        # Good quality document
        doc.text = " ".join(["word" for _ in range(50)])
        processed = quality_filter.process(doc)
        assert processed is not None
        assert hasattr(processed, 'quality_score')


class TestProcessorIntegration:
    """Integration tests for processor pipeline"""

    def test_full_processing_pipeline(self):
        """Test complete processing pipeline"""
        # Create processors
        cleaner = Cleaner({'remove_urls': True, 'lowercase': True})
        quality_filter = QualityFilter({'min_score': 0.5})
        chunker = Chunker({'strategy': 'fixed', 'size': 10})
        privacy = PrivacyProtector({'detect_pii': True})
        dedup = Deduplicator({'method': 'hash'})

        # Create test document
        doc = Mock()
        doc.text = """
        Visit https://example.com for more info.
        Contact john@example.com for details.
        This is good quality content with enough words to pass filters.
        """

        # Process through pipeline
        doc = cleaner.process(doc)
        assert 'https://example.com' not in doc.text
        assert doc.text.islower()

        doc = privacy.process(doc)
        assert 'john@example.com' not in doc.text

        doc = quality_filter.process(doc)
        assert doc is not None  # Should pass quality check

        doc = chunker.process(doc)
        assert hasattr(doc, 'chunks')
        assert len(doc.chunks) > 0

        # Process chunks through deduplication
        original_chunks = len(doc.chunks)
        doc = dedup.process(doc)
        assert len(doc.chunks) <= original_chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])