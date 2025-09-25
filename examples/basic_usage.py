#!/usr/bin/env python
"""
Basic usage examples for Dataseter
"""

from dataseter import DatasetCreator

def example_pdf_extraction():
    """Example: Extract data from PDF files"""
    print("Example 1: PDF Extraction")

    creator = DatasetCreator()

    # Add PDF files
    creator.add_pdf("documents/report1.pdf")
    creator.add_pdf("documents/report2.pdf")

    # Process with custom settings
    dataset = creator.process(
        chunk_size=512,
        overlap=50,
        remove_pii=True,
        quality_threshold=0.7
    )

    # Export to different formats
    dataset.to_jsonl("output/pdf_dataset.jsonl")
    dataset.to_parquet("output/pdf_dataset.parquet")

    print(f"Created dataset with {len(dataset)} documents")
    print(f"Statistics: {dataset.statistics}")


def example_web_scraping():
    """Example: Scrape websites recursively"""
    print("\nExample 2: Web Scraping")

    creator = DatasetCreator()

    # Add websites with different depths
    creator.add_website("https://docs.python.org", max_depth=2)
    creator.add_website("https://wikipedia.org/wiki/Machine_learning", max_depth=1)

    # Process with web-specific settings
    dataset = creator.process(
        chunk_size=1024,
        overlap=100,
        remove_pii=True,
        quality_threshold=0.8
    )

    # Export for HuggingFace
    dataset.to_huggingface("my-web-dataset")

    print(f"Scraped {len(dataset)} pages")


def example_mixed_sources():
    """Example: Combine multiple source types"""
    print("\nExample 3: Mixed Sources")

    creator = DatasetCreator()

    # Add various sources
    creator.add_pdf("research/paper.pdf")
    creator.add_website("https://arxiv.org", max_depth=1)
    creator.add_directory("./documents", recursive=True, pattern="*.txt")
    creator.add_office_document("presentations/slides.pptx")
    creator.add_ebook("books/textbook.epub")

    # Process with balanced settings
    dataset = creator.process(
        chunk_size=768,
        overlap=75,
        remove_pii=True,
        quality_threshold=0.75,
        remove_duplicates=True
    )

    # Export with compression
    dataset.to_jsonl("output/mixed_dataset.jsonl.gz", compress=True)

    print(f"Processed {len(dataset)} documents from mixed sources")


def example_custom_pipeline():
    """Example: Create custom processing pipeline"""
    print("\nExample 4: Custom Pipeline")

    from dataseter import Pipeline, DatasetCreator
    from dataseter.processors import Chunker, Cleaner, QualityFilter

    # Create custom pipeline
    pipeline = Pipeline()
    pipeline.add_step(Cleaner({'lowercase': True, 'remove_urls': True}))
    pipeline.add_step(QualityFilter({'min_score': 0.8}))
    pipeline.add_step(Chunker({'strategy': 'semantic', 'size': 1024}))

    # Use custom pipeline
    creator = DatasetCreator()
    creator.add_directory("./corpus", recursive=True)

    dataset = creator.process(pipeline=pipeline)

    print(f"Processed with custom pipeline: {len(dataset)} documents")


def example_analysis():
    """Example: Analyze existing dataset"""
    print("\nExample 5: Dataset Analysis")

    from dataseter.analyzers import DatasetAnalyzer, Visualizer

    # Load and analyze dataset
    analyzer = DatasetAnalyzer()

    # Assuming you have an existing dataset
    creator = DatasetCreator()
    creator.add_directory("./sample_texts", pattern="*.txt")
    dataset = creator.process()

    # Analyze
    stats = analyzer.analyze(dataset)

    print("Dataset Analysis:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Average quality: {stats['quality_stats']['mean']:.2f}")
    print(f"  Language distribution: {stats['languages']}")

    # Generate visualizations
    visualizer = Visualizer()
    visualizer.create_report(dataset, "analysis_report.html")


def example_parallel_processing():
    """Example: Parallel processing for large datasets"""
    print("\nExample 6: Parallel Processing")

    creator = DatasetCreator()

    # Add many sources
    for i in range(100):
        creator.add_pdf(f"documents/doc_{i}.pdf")

    # Process in parallel with multiple workers
    dataset = creator.process(
        chunk_size=512,
        parallel=True,
        max_workers=8  # Use 8 CPU cores
    )

    print(f"Processed {len(dataset)} documents in parallel")


def example_streaming():
    """Example: Stream processing for memory efficiency"""
    print("\nExample 7: Streaming Large Files")

    from dataseter import StreamingDatasetCreator  # Hypothetical streaming version

    # For very large datasets that don't fit in memory
    creator = StreamingDatasetCreator()

    # Process large files in chunks
    creator.add_large_file("huge_corpus.txt", chunk_mb=100)

    # Stream process and write directly to disk
    creator.stream_process(
        output_path="output/streaming_dataset.jsonl",
        chunk_size=512,
        batch_size=1000  # Write every 1000 documents
    )

    print("Streaming processing completed")


if __name__ == "__main__":
    # Run examples
    print("Dataseter Usage Examples")
    print("=" * 50)

    # Note: These examples assume you have the necessary files
    # Uncomment the examples you want to run

    # example_pdf_extraction()
    # example_web_scraping()
    # example_mixed_sources()
    # example_custom_pipeline()
    # example_analysis()
    # example_parallel_processing()
    # example_streaming()

    print("\nFor more examples, see the documentation.")