"""
Basic tests for Dataseter core functionality
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.core import DatasetCreator, Document, Dataset, Pipeline
from src.processors import Chunker, Cleaner


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file"""
    file_path = Path(temp_dir) / "sample.txt"
    file_path.write_text("This is a sample text file for testing Dataseter.")
    return str(file_path)


@pytest.fixture
def dataset_creator():
    """Create a DatasetCreator instance"""
    return DatasetCreator()


class TestDocument:
    """Test Document class"""

    def test_document_creation(self):
        doc = Document(
            id="test123",
            text="Sample text",
            source="test.txt"
        )
        assert doc.id == "test123"
        assert doc.text == "Sample text"
        assert doc.source == "test.txt"

    def test_document_metadata(self):
        doc = Document(
            id="test",
            text="Text",
            metadata={"author": "Test Author"}
        )
        assert doc.metadata["author"] == "Test Author"


class TestDataset:
    """Test Dataset class"""

    def test_dataset_creation(self):
        dataset = Dataset()
        assert len(dataset) == 0

    def test_dataset_add_document(self):
        dataset = Dataset()
        doc = Document(id="1", text="Test")
        dataset.documents.append(doc)
        assert len(dataset) == 1

    def test_dataset_export_methods(self):
        dataset = Dataset()
        # Check that export methods exist
        assert hasattr(dataset, 'to_jsonl')
        assert hasattr(dataset, 'to_parquet')
        assert hasattr(dataset, 'to_csv')
        assert hasattr(dataset, 'to_huggingface')


class TestPipeline:
    """Test Pipeline class"""

    def test_pipeline_creation(self):
        pipeline = Pipeline()
        assert len(pipeline.steps) == 0

    def test_pipeline_add_step(self):
        pipeline = Pipeline()
        pipeline.add_step(lambda x: x)
        assert len(pipeline.steps) == 1

    def test_pipeline_clear(self):
        pipeline = Pipeline()
        pipeline.add_step(lambda x: x)
        pipeline.clear()
        assert len(pipeline.steps) == 0


class TestDatasetCreator:
    """Test DatasetCreator class"""

    def test_creator_initialization(self, dataset_creator):
        assert dataset_creator is not None
        assert len(dataset_creator.sources) == 0

    def test_add_pdf(self, dataset_creator, temp_dir):
        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.write_text("dummy")  # Create dummy file

        dataset_creator.add_pdf(str(pdf_path))
        assert len(dataset_creator.sources) == 1
        assert dataset_creator.sources[0]['type'] == 'pdf'

    def test_add_website(self, dataset_creator):
        dataset_creator.add_website("https://example.com", max_depth=2)
        assert len(dataset_creator.sources) == 1
        assert dataset_creator.sources[0]['type'] == 'web'

    def test_add_directory(self, dataset_creator, temp_dir):
        # Create some test files
        Path(temp_dir, "file1.txt").write_text("content1")
        Path(temp_dir, "file2.txt").write_text("content2")

        dataset_creator.add_directory(temp_dir, pattern="*.txt")
        assert len(dataset_creator.sources) == 2

    def test_detect_file_type(self, dataset_creator):
        assert dataset_creator._detect_file_type("test.pdf") == "pdf"
        assert dataset_creator._detect_file_type("test.docx") == "office"
        assert dataset_creator._detect_file_type("test.epub") == "ebook"
        assert dataset_creator._detect_file_type("test.txt") == "text"

    def test_clear(self, dataset_creator):
        dataset_creator.add_website("https://example.com")
        dataset_creator.clear()
        assert len(dataset_creator.sources) == 0


class TestChunker:
    """Test Chunker processor"""

    def test_chunker_initialization(self):
        chunker = Chunker({'size': 100, 'overlap': 10})
        assert chunker.chunk_size == 100
        assert chunker.overlap == 10

    def test_fixed_chunking(self):
        chunker = Chunker({'strategy': 'fixed', 'size': 5})
        text = "This is a test text for chunking purposes only"
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)

    def test_sliding_window_chunking(self):
        chunker = Chunker({'strategy': 'sliding_window', 'size': 5, 'overlap': 2})
        text = "This is a test text for chunking with overlap"
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 0


class TestCleaner:
    """Test Cleaner processor"""

    def test_cleaner_initialization(self):
        cleaner = Cleaner({'lowercase': True})
        assert cleaner.lowercase is True

    def test_clean_text_lowercase(self):
        cleaner = Cleaner({'lowercase': True})
        text = "HELLO World"
        cleaned = cleaner.clean_text(text)
        assert cleaned == "hello world"

    def test_clean_text_remove_urls(self):
        cleaner = Cleaner({'remove_urls': True})
        text = "Visit https://example.com for more"
        cleaned = cleaner.clean_text(text)
        assert "https://example.com" not in cleaned

    def test_clean_text_remove_extra_whitespace(self):
        cleaner = Cleaner({'remove_extra_whitespace': True})
        text = "Too    many     spaces"
        cleaned = cleaner.clean_text(text)
        assert cleaned == "Too many spaces"


# Integration tests
class TestIntegration:
    """Integration tests"""

    def test_end_to_end_text_processing(self, dataset_creator, sample_text_file):
        """Test complete pipeline from file to dataset"""
        dataset_creator.add_directory(
            os.path.dirname(sample_text_file),
            pattern="*.txt"
        )

        # This would normally process the files, but we need mock extractors
        # for testing without actual file processing
        assert len(dataset_creator.sources) > 0

    @pytest.mark.slow
    def test_parallel_processing(self, dataset_creator, temp_dir):
        """Test parallel processing capability"""
        # Create multiple files
        for i in range(5):
            Path(temp_dir, f"file{i}.txt").write_text(f"Content {i}")

        dataset_creator.add_directory(temp_dir, pattern="*.txt")
        assert len(dataset_creator.sources) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])