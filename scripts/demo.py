#!/usr/bin/env python
"""
Dataseter Demo - Comprehensive demonstration of all features
"""

import time
import tempfile
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

console = Console()


def demo_header():
    """Display demo header"""
    console.print(Panel.fit("""
    [bold cyan]🚀 Dataseter Demo[/bold cyan]
    [yellow]Advanced AI Training Dataset Creator[/yellow]

    This demo showcases all major features:
    • Multi-source extraction (Web, PDF, Office, eBooks)
    • Advanced text processing
    • Quality filtering & PII removal
    • Multiple output formats
    • Real-time web scraping
    """, title="Welcome"))


def demo_web_extraction():
    """Demonstrate web extraction"""
    console.print("\n[bold magenta]📡 Demo 1: Web Extraction[/bold magenta]")
    console.print("Extracting content from live websites...\n")

    from src.core import DatasetCreator

    urls = [
        ("Wikipedia AI Article", "https://en.wikipedia.org/wiki/Artificial_intelligence"),
        ("Python Documentation", "https://docs.python.org/3/tutorial/index.html"),
        ("HTTPBin Test Page", "https://httpbin.org/html")
    ]

    creator = DatasetCreator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        for name, url in urls:
            task = progress.add_task(f"Extracting {name}...", total=None)
            creator.add_website(url, max_depth=0)
            progress.update(task, completed=100)
            console.print(f"  [green]✓[/green] Added {name}")

    console.print("\n[cyan]Processing dataset...[/cyan]")

    dataset = creator.process(
        chunk_size=500,
        overlap=50,
        remove_pii=True,
        quality_threshold=0.7
    )

    # Display results
    table = Table(title="Web Extraction Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Pages Extracted", str(len(dataset)))
    table.add_row("Total Text Length", f"{dataset.statistics.get('total_text_length', 0):,}")
    table.add_row("Chunks Created", str(dataset.statistics.get('total_chunks', 0)))
    table.add_row("Quality Score", f"{dataset.statistics.get('quality_stats', {}).get('mean', 0):.2f}")

    console.print(table)


def demo_file_processing():
    """Demonstrate file processing"""
    console.print("\n[bold magenta]📁 Demo 2: File Processing[/bold magenta]")
    console.print("Processing various file formats...\n")

    from src.core import DatasetCreator

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files
        files = {
            "document.txt": "This is a sample document with important information about AI and machine learning.",
            "report.md": "# Technical Report\n\n## Introduction\nThis report covers dataset creation techniques.\n\n## Methods\nWe use advanced NLP processing.",
            "data.csv": "name,value\nitem1,100\nitem2,200\nitem3,300"
        }

        for filename, content in files.items():
            filepath = Path(tmpdir) / filename
            filepath.write_text(content)
            console.print(f"  [green]✓[/green] Created {filename}")

        creator = DatasetCreator()
        creator.add_directory(tmpdir, recursive=True)

        dataset = creator.process(
            chunk_size=100,
            remove_duplicates=True
        )

        console.print(f"\n[green]Processed {len(dataset)} documents from files[/green]")


def demo_text_processing():
    """Demonstrate text processing pipeline"""
    console.print("\n[bold magenta]🔧 Demo 3: Text Processing Pipeline[/bold magenta]")
    console.print("Demonstrating advanced text processing...\n")

    from src.processors import Cleaner, PrivacyProtector, QualityFilter, Chunker

    sample_text = """
    Contact our team at support@example.com or call 555-123-4567.
    Visit https://example.com for more information.

    This is HIGH QUALITY content with proper formatting.
    John Doe's SSN is 123-45-6789 (fake for demo).
    """

    console.print("[yellow]Original text:[/yellow]")
    console.print(Panel(sample_text, expand=False))

    # Clean text
    cleaner = Cleaner({
        'remove_urls': True,
        'remove_emails': True,
        'remove_extra_whitespace': True
    })
    cleaned = cleaner.clean_text(sample_text)
    console.print("\n[yellow]After cleaning:[/yellow]")
    console.print(Panel(cleaned, expand=False))

    # Remove PII
    privacy = PrivacyProtector({'detect_pii': True})
    protected = privacy.protect_text(cleaned)
    console.print("\n[yellow]After PII removal:[/yellow]")
    console.print(Panel(protected, expand=False))

    # Quality scoring
    quality = QualityFilter()
    score = quality.calculate_quality_score(protected)
    console.print(f"\n[cyan]Quality Score: {score:.2f}[/cyan]")


def demo_chunking_strategies():
    """Demonstrate different chunking strategies"""
    console.print("\n[bold magenta]✂️ Demo 4: Chunking Strategies[/bold magenta]")

    from src.processors import Chunker

    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines.
    It involves the simulation of natural intelligence in machines.
    These machines are programmed to think like humans and mimic their actions.

    Machine learning is a subset of AI.
    It focuses on the use of data and algorithms.
    The goal is to imitate the way humans learn.
    """

    strategies = ['fixed', 'sliding_window', 'sentence', 'paragraph']

    for strategy in strategies:
        chunker = Chunker({'strategy': strategy, 'size': 50, 'overlap': 10})
        chunks = chunker.chunk_text(text)
        console.print(f"\n[cyan]{strategy.title()} Chunking:[/cyan] {len(chunks)} chunks")

        if chunks:
            console.print(f"  First chunk: {chunks[0]['text'][:50]}...")


def demo_output_formats():
    """Demonstrate output formats"""
    console.print("\n[bold magenta]💾 Demo 5: Output Formats[/bold magenta]")
    console.print("Exporting dataset in multiple formats...\n")

    from src.core import DatasetCreator, Document, Dataset
    import tempfile

    # Create sample dataset
    dataset = Dataset()
    for i in range(3):
        doc = Document(
            id=f"doc_{i}",
            text=f"Sample document {i} content for demonstration.",
            source=f"demo_{i}.txt"
        )
        dataset.documents.append(doc)

    with tempfile.TemporaryDirectory() as tmpdir:
        formats = {
            'jsonl': 'dataset.jsonl',
            'csv': 'dataset.csv',
            'parquet': 'dataset.parquet'
        }

        for format_name, filename in formats.items():
            filepath = Path(tmpdir) / filename

            if format_name == 'jsonl':
                dataset.to_jsonl(str(filepath))
            elif format_name == 'csv':
                dataset.to_csv(str(filepath))
            elif format_name == 'parquet':
                try:
                    dataset.to_parquet(str(filepath))
                except:
                    console.print(f"  [yellow]⚠[/yellow] Parquet export requires pyarrow")
                    continue

            if filepath.exists():
                size = filepath.stat().st_size
                console.print(f"  [green]✓[/green] Exported to {format_name.upper()}: {size:,} bytes")


def demo_parallel_processing():
    """Demonstrate parallel processing"""
    console.print("\n[bold magenta]⚡ Demo 6: Parallel Processing[/bold magenta]")
    console.print("Comparing sequential vs parallel processing...\n")

    from src.core import DatasetCreator
    import time

    # Create multiple sources
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/xml",
    ]

    # Sequential processing
    creator_seq = DatasetCreator()
    for url in urls:
        creator_seq.add_website(url, max_depth=0)

    start = time.time()
    with console.status("[cyan]Sequential processing..."):
        dataset_seq = creator_seq.process(parallel=False)
    seq_time = time.time() - start

    console.print(f"  Sequential: {len(dataset_seq)} docs in {seq_time:.2f}s")

    # Parallel processing
    creator_par = DatasetCreator()
    for url in urls:
        creator_par.add_website(url, max_depth=0)

    start = time.time()
    with console.status("[cyan]Parallel processing..."):
        dataset_par = creator_par.process(parallel=True, max_workers=3)
    par_time = time.time() - start

    console.print(f"  Parallel:   {len(dataset_par)} docs in {par_time:.2f}s")

    if par_time < seq_time:
        speedup = (seq_time / par_time - 1) * 100
        console.print(f"\n[green]Parallel processing {speedup:.0f}% faster![/green]")


def demo_quality_analysis():
    """Demonstrate quality analysis"""
    console.print("\n[bold magenta]📊 Demo 7: Quality Analysis[/bold magenta]")
    console.print("Analyzing dataset quality...\n")

    from src.core import DatasetCreator
    from src.analyzers import DatasetAnalyzer

    # Create a diverse dataset
    creator = DatasetCreator()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files with different quality
        high_quality = Path(tmpdir) / "high.txt"
        high_quality.write_text(
            "This is high-quality content with proper grammar and structure. "
            "It contains meaningful information that would be valuable for training. " * 10
        )

        low_quality = Path(tmpdir) / "low.txt"
        low_quality.write_text("bad bad bad " * 20)

        creator.add_directory(tmpdir)
        creator.add_website("https://httpbin.org/html", max_depth=0)

        dataset = creator.process(quality_threshold=0.5)

        # Analyze
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze(dataset)

        # Display analysis
        table = Table(title="Dataset Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Documents", str(stats.get('total_documents', 0)))
        table.add_row("Total Words", f"{stats.get('total_words', 0):,}")
        table.add_row("Vocabulary Size", f"{stats.get('vocabulary_size', 0):,}")

        if 'quality_stats' in stats:
            table.add_row("Avg Quality Score", f"{stats['quality_stats']['mean']:.2f}")
            table.add_row("Min Quality Score", f"{stats['quality_stats']['min']:.2f}")
            table.add_row("Max Quality Score", f"{stats['quality_stats']['max']:.2f}")

        if 'languages' in stats:
            langs = ", ".join(f"{k}:{v}" for k, v in list(stats['languages'].items())[:3])
            table.add_row("Languages", langs)

        console.print(table)


def run_demo():
    """Run the complete demo"""
    demo_header()

    demos = [
        ("Web Extraction", demo_web_extraction),
        ("File Processing", demo_file_processing),
        ("Text Processing", demo_text_processing),
        ("Chunking Strategies", demo_chunking_strategies),
        ("Output Formats", demo_output_formats),
        ("Parallel Processing", demo_parallel_processing),
        ("Quality Analysis", demo_quality_analysis)
    ]

    console.print("\n[bold]Select demo to run:[/bold]")
    console.print("0. Run all demos")

    for i, (name, _) in enumerate(demos, 1):
        console.print(f"{i}. {name}")

    try:
        choice = input("\nEnter choice (0-7): ").strip()
        choice = int(choice)

        if choice == 0:
            console.print("\n[bold cyan]Running all demos...[/bold cyan]")
            for name, demo_func in demos:
                try:
                    demo_func()
                    time.sleep(1)
                except Exception as e:
                    console.print(f"[red]Error in {name}: {e}[/red]")
        elif 1 <= choice <= len(demos):
            name, demo_func = demos[choice - 1]
            console.print(f"\n[bold cyan]Running {name} demo...[/bold cyan]")
            demo_func()
        else:
            console.print("[red]Invalid choice[/red]")

    except (ValueError, KeyboardInterrupt):
        console.print("\n[yellow]Demo cancelled[/yellow]")

    console.print("\n" + "=" * 50)
    console.print(Panel.fit("""
    [bold green]✅ Demo Complete![/bold green]

    To get started with Dataseter:

    1. Install: pip install -e .
    2. CLI: dataseter create --help
    3. Web UI: dataseter web
    4. API: See documentation

    Visit: https://github.com/mkupermann/dataseter
    """, title="Next Steps"))


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
        import traceback
        if input("Show traceback? (y/n): ").lower() == 'y':
            traceback.print_exc()