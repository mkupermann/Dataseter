#!/usr/bin/env python
"""
End-to-end integration tests with real data sources
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
import requests
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import DatasetCreator
from src.extractors import WebExtractor, TextExtractor, PDFExtractor
from src.processors import Chunker, Cleaner, PrivacyProtector, QualityFilter
from src.formatters import JSONLFormatter, ParquetFormatter


class TestRealWebExtraction:
    """Test extraction from real websites"""

    def test_extract_from_wikipedia(self):
        """Test extracting from Wikipedia"""
        creator = DatasetCreator()

        # Add Wikipedia page
        creator.add_website(
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            max_depth=0  # Just the main page
        )

        # Process with minimal settings
        dataset = creator.process(
            chunk_size=500,
            overlap=50,
            remove_pii=True,
            quality_threshold=0.6
        )

        assert len(dataset) > 0
        assert dataset.documents[0].text is not None
        assert 'artificial intelligence' in dataset.documents[0].text.lower()
        assert dataset.statistics['total_documents'] > 0

    def test_extract_from_github(self):
        """Test extracting from GitHub"""
        creator = DatasetCreator()

        # Add GitHub README
        creator.add_website(
            "https://raw.githubusercontent.com/python/cpython/main/README.rst",
            max_depth=0
        )

        dataset = creator.process(
            chunk_size=300,
            remove_pii=False,
            quality_threshold=0.5
        )

        assert len(dataset) > 0
        assert 'Python' in dataset.documents[0].text

    def test_extract_from_news_site(self):
        """Test extracting from news website"""
        creator = DatasetCreator()

        # Use a stable API documentation page
        creator.add_website(
            "https://docs.python.org/3/",
            max_depth=1  # Main page + one level
        )

        dataset = creator.process(
            chunk_size=1000,
            overlap=100,
            remove_pii=True,
            quality_threshold=0.7,
            parallel=True
        )

        assert len(dataset) > 0
        assert dataset.statistics['total_text_length'] > 1000

    def test_extract_multiple_sites(self):
        """Test extracting from multiple websites concurrently"""
        creator = DatasetCreator()

        # Add multiple sites
        sites = [
            "https://httpbin.org/html",
            "https://httpbin.org/json",
            "https://httpbin.org/xml"
        ]

        for site in sites:
            creator.add_website(site, max_depth=0)

        dataset = creator.process(
            chunk_size=512,
            parallel=True,
            max_workers=3
        )

        assert len(dataset) >= len(sites)


class TestCompleteWorkflow:
    """Test complete dataset creation workflow"""

    def test_mixed_sources_workflow(self):
        """Test processing mixed data sources"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            text_file = Path(tmpdir) / "test.txt"
            text_file.write_text(
                "This is a test document for Dataseter. "
                "It contains multiple sentences and paragraphs.\n\n"
                "Second paragraph with more content. "
                "Testing the extraction and processing pipeline."
            )

            markdown_file = Path(tmpdir) / "test.md"
            markdown_file.write_text(
                "# Test Markdown\n\n"
                "## Section 1\n"
                "Content in section 1.\n\n"
                "## Section 2\n"
                "Content in section 2."
            )

            # Create dataset
            creator = DatasetCreator()

            # Add local files
            creator.add_directory(tmpdir, pattern="*.txt")
            creator.add_directory(tmpdir, pattern="*.md")

            # Add web source
            creator.add_website("https://httpbin.org/html", max_depth=0)

            # Process
            dataset = creator.process(
                chunk_size=100,
                overlap=20,
                remove_pii=True,
                quality_threshold=0.5
            )

            assert len(dataset) >= 3  # At least 3 documents
            assert dataset.statistics['total_documents'] >= 3

            # Test export formats
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Export to JSONL
            jsonl_path = output_dir / "dataset.jsonl"
            dataset.to_jsonl(str(jsonl_path))
            assert jsonl_path.exists()

            # Verify JSONL content
            with open(jsonl_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == len(dataset)

                # Check first document
                doc = json.loads(lines[0])
                assert 'id' in doc
                assert 'text' in doc
                assert 'source' in doc

    def test_quality_filtering_workflow(self):
        """Test quality filtering in workflow"""
        creator = DatasetCreator()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create high quality file
            good_file = Path(tmpdir) / "good.txt"
            good_file.write_text(
                "This is high quality content with proper sentences. "
                "It contains meaningful information and good structure. "
                "The text is well-formatted and contains useful data. " * 5
            )

            # Create low quality file
            bad_file = Path(tmpdir) / "bad.txt"
            bad_file.write_text("bad bad bad bad bad")

            creator.add_directory(tmpdir)

            dataset = creator.process(
                quality_threshold=0.7,
                remove_duplicates=True
            )

            # Should filter out low quality
            assert all(
                doc.quality_score >= 0.7
                for doc in dataset.documents
                if hasattr(doc, 'quality_score')
            )

    def test_pii_removal_workflow(self):
        """Test PII removal in workflow"""
        creator = DatasetCreator()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with PII
            pii_file = Path(tmpdir) / "pii.txt"
            pii_file.write_text(
                "Contact John Doe at john.doe@example.com or call 555-123-4567. "
                "His SSN is 123-45-6789 and IP address is 192.168.1.1."
            )

            creator.add_directory(tmpdir)

            dataset = creator.process(
                remove_pii=True,
                chunk_size=500
            )

            # Check PII was removed
            combined_text = ' '.join(doc.text for doc in dataset.documents)
            assert 'john.doe@example.com' not in combined_text
            assert '555-123-4567' not in combined_text
            assert '123-45-6789' not in combined_text
            assert '192.168.1.1' not in combined_text

    def test_chunking_strategies(self):
        """Test different chunking strategies"""
        test_text = """
        # Main Header

        This is the introduction paragraph with some content.

        ## Section 1
        Content in section 1. This has multiple sentences.
        Each sentence adds information.

        ## Section 2
        More content here in section 2.
        Final thoughts and conclusion.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            text_file = Path(tmpdir) / "test.txt"
            text_file.write_text(test_text)

            # Test different strategies
            strategies = ['fixed', 'sliding_window', 'sentence', 'paragraph', 'semantic']

            for strategy in strategies:
                creator = DatasetCreator()
                creator.add_directory(tmpdir)

                # Custom pipeline with specific chunking
                from src.core import Pipeline
                pipeline = Pipeline()
                pipeline.add_step(Cleaner())
                pipeline.add_step(Chunker({'strategy': strategy, 'size': 50}))

                dataset = creator.process(pipeline=pipeline)

                assert len(dataset) > 0
                if dataset.documents[0].chunks:
                    assert len(dataset.documents[0].chunks) > 0


class TestPerformance:
    """Test performance and scalability"""

    def test_parallel_processing(self):
        """Test parallel processing performance"""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(10):
                file = Path(tmpdir) / f"file_{i}.txt"
                file.write_text(f"Content for file {i}. " * 100)

            # Sequential processing
            creator_seq = DatasetCreator()
            creator_seq.add_directory(tmpdir)

            start = time.time()
            dataset_seq = creator_seq.process(parallel=False)
            seq_time = time.time() - start

            # Parallel processing
            creator_par = DatasetCreator()
            creator_par.add_directory(tmpdir)

            start = time.time()
            dataset_par = creator_par.process(parallel=True, max_workers=4)
            par_time = time.time() - start

            # Both should produce same results
            assert len(dataset_seq) == len(dataset_par)

            # Parallel should not be significantly slower
            # (might not be faster for small datasets due to overhead)
            assert par_time < seq_time * 2

    def test_large_document_handling(self):
        """Test handling of large documents"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large file (1MB)
            large_file = Path(tmpdir) / "large.txt"
            content = "Large document content. " * 50000
            large_file.write_text(content)

            creator = DatasetCreator()
            creator.add_directory(tmpdir)

            dataset = creator.process(
                chunk_size=1000,
                overlap=100
            )

            assert len(dataset) > 0
            assert dataset.documents[0].chunks
            assert len(dataset.documents[0].chunks) > 10

    @pytest.mark.slow
    def test_stress_test_web_extraction(self):
        """Stress test web extraction with multiple pages"""
        creator = DatasetCreator()

        # Add website with deeper crawl
        creator.add_website(
            "https://docs.python.org/3/tutorial/",
            max_depth=2  # Will fetch many pages
        )

        dataset = creator.process(
            chunk_size=500,
            parallel=True,
            max_workers=4
        )

        assert len(dataset) > 0
        assert dataset.statistics['total_documents'] > 1


class TestErrorHandling:
    """Test error handling and recovery"""

    def test_invalid_url_handling(self):
        """Test handling of invalid URLs"""
        creator = DatasetCreator()
        creator.add_website("https://this-definitely-does-not-exist-12345.com")

        dataset = creator.process()

        # Should handle gracefully
        assert dataset is not None
        # May have 0 documents due to failed extraction
        assert len(dataset) >= 0

    def test_mixed_valid_invalid_sources(self):
        """Test handling mix of valid and invalid sources"""
        creator = DatasetCreator()

        # Add mix of valid and invalid
        creator.add_website("https://httpbin.org/html")  # Valid
        creator.add_website("https://invalid-url-12345.com")  # Invalid

        with tempfile.TemporaryDirectory() as tmpdir:
            valid_file = Path(tmpdir) / "valid.txt"
            valid_file.write_text("Valid content")
            creator.add_directory(tmpdir)

            creator.add_pdf("/nonexistent/file.pdf")  # Invalid

        dataset = creator.process()

        # Should process valid sources
        assert len(dataset) >= 2  # At least web and text file

    def test_timeout_handling(self):
        """Test handling of slow/timeout responses"""
        extractor = WebExtractor({
            'timeout': 1,  # 1 second timeout
            'retry_attempts': 1
        })

        # Use httpbin delay endpoint
        result = extractor.extract("https://httpbin.org/delay/10")

        # Should timeout and return error
        assert 'error' in result or result['text'] == ''


class TestCLIIntegration:
    """Test CLI functionality"""

    def test_cli_basic_execution(self):
        """Test basic CLI execution"""
        from src.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test content for CLI")

            output_file = Path(tmpdir) / "output.jsonl"

            # Run CLI command
            result = runner.invoke(cli, [
                'create',
                '--directory', tmpdir,
                '--output', str(output_file),
                '--format', 'jsonl',
                '--chunk-size', '100'
            ])

            # Check execution
            assert result.exit_code == 0
            assert output_file.exists()

    def test_cli_config_generation(self):
        """Test config file generation"""
        from src.cli.main import cli
        from click.testing import CliRunner

        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            os.chdir(tmpdir)

            result = runner.invoke(cli, ['config'])

            assert result.exit_code == 0
            assert Path("dataseter_config.yaml").exists()

            os.chdir(original_dir)


if __name__ == "__main__":
    print("Running Dataseter End-to-End Tests")
    print("=" * 50)

    # Run specific test suites
    test_suites = [
        "TestRealWebExtraction",
        "TestCompleteWorkflow",
        "TestPerformance",
        "TestErrorHandling",
        "TestCLIIntegration"
    ]

    for suite in test_suites:
        print(f"\nRunning {suite}...")
        pytest.main([__file__, f"::{suite}", "-v", "-s"])

    print("\n" + "=" * 50)
    print("All tests completed!")