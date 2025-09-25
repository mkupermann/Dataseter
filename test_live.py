#!/usr/bin/env python
"""
Live testing script for Dataseter - Tests real website extraction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core import DatasetCreator
from src.extractors import WebExtractor
import json
import time
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def test_single_website(url, max_depth=0):
    """Test extraction from a single website"""
    console.print(f"\n[bold cyan]Testing: {url}[/bold cyan]")

    creator = DatasetCreator()
    creator.add_website(url, max_depth=max_depth)

    start_time = time.time()

    try:
        dataset = creator.process(
            chunk_size=500,
            overlap=50,
            remove_pii=True,
            quality_threshold=0.6,
            remove_duplicates=True
        )

        elapsed = time.time() - start_time

        # Display results
        if len(dataset) > 0:
            console.print(f"[green]✓ Success![/green] Extracted in {elapsed:.2f}s")

            doc = dataset.documents[0]
            console.print(f"\nFirst 500 characters of extracted text:")
            console.print("-" * 50)
            console.print(doc.text[:500] + "...")
            console.print("-" * 50)

            # Show statistics
            stats = dataset.statistics
            console.print(f"\nStatistics:")
            console.print(f"  Documents: {len(dataset)}")
            console.print(f"  Total text: {stats.get('total_text_length', 0):,} chars")
            console.print(f"  Vocabulary: {stats.get('vocabulary_size', 0):,} unique words")

            if hasattr(doc, 'quality_score'):
                console.print(f"  Quality score: {doc.quality_score:.2f}")

            if hasattr(doc, 'language'):
                console.print(f"  Language: {doc.language}")

            return True
        else:
            console.print(f"[yellow]⚠ No content extracted[/yellow]")
            return False

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return False


def test_multiple_websites():
    """Test extraction from multiple websites"""
    console.print("\n[bold magenta]Testing Multiple Websites[/bold magenta]")

    test_sites = [
        {
            "name": "Wikipedia - Machine Learning",
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "max_depth": 0
        },
        {
            "name": "Python.org Tutorial",
            "url": "https://docs.python.org/3/tutorial/index.html",
            "max_depth": 1
        },
        {
            "name": "HTTPBin HTML",
            "url": "https://httpbin.org/html",
            "max_depth": 0
        },
        {
            "name": "GitHub REST API Docs",
            "url": "https://docs.github.com/en/rest",
            "max_depth": 0
        },
        {
            "name": "Example.com",
            "url": "https://example.com",
            "max_depth": 0
        }
    ]

    results = []

    for site in track(test_sites, description="Testing websites..."):
        console.print(f"\n[cyan]Testing {site['name']}...[/cyan]")

        creator = DatasetCreator()
        creator.add_website(site['url'], max_depth=site['max_depth'])

        try:
            start = time.time()
            dataset = creator.process(
                chunk_size=500,
                overlap=50,
                remove_pii=True,
                quality_threshold=0.5
            )
            elapsed = time.time() - start

            if len(dataset) > 0:
                results.append({
                    "name": site['name'],
                    "status": "✓",
                    "docs": len(dataset),
                    "chars": dataset.statistics.get('total_text_length', 0),
                    "time": elapsed
                })
                console.print(f"  [green]✓[/green] Extracted {len(dataset)} documents")
            else:
                results.append({
                    "name": site['name'],
                    "status": "⚠",
                    "docs": 0,
                    "chars": 0,
                    "time": elapsed
                })
                console.print(f"  [yellow]⚠[/yellow] No content extracted")

        except Exception as e:
            results.append({
                "name": site['name'],
                "status": "✗",
                "docs": 0,
                "chars": 0,
                "time": 0
            })
            console.print(f"  [red]✗[/red] Error: {str(e)[:50]}")

    # Display summary table
    console.print("\n[bold]Summary:[/bold]")

    table = Table()
    table.add_column("Website", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Docs", style="green")
    table.add_column("Characters", style="yellow")
    table.add_column("Time (s)", style="magenta")

    for result in results:
        table.add_row(
            result['name'],
            result['status'],
            str(result['docs']),
            f"{result['chars']:,}",
            f"{result['time']:.2f}"
        )

    console.print(table)

    # Save test results
    output_file = "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_file}[/green]")


def test_deep_crawling():
    """Test deep website crawling"""
    console.print("\n[bold magenta]Testing Deep Crawling[/bold magenta]")

    url = "https://docs.python.org/3/tutorial/"
    max_depth = 2

    console.print(f"URL: {url}")
    console.print(f"Max depth: {max_depth}")
    console.print("[yellow]This may take a while...[/yellow]\n")

    creator = DatasetCreator()
    creator.add_website(url, max_depth=max_depth)

    start = time.time()

    dataset = creator.process(
        chunk_size=1000,
        overlap=100,
        quality_threshold=0.6,
        parallel=True,
        max_workers=4
    )

    elapsed = time.time() - start

    console.print(f"\n[green]Crawling completed in {elapsed:.2f}s[/green]")
    console.print(f"Pages extracted: {len(dataset)}")
    console.print(f"Total text: {dataset.statistics.get('total_text_length', 0):,} characters")

    # Show source distribution
    if dataset.statistics.get('sources'):
        console.print("\nPages crawled:")
        for source, count in list(dataset.statistics['sources'].items())[:10]:
            console.print(f"  • {source}: {count}")


def test_javascript_rendering():
    """Test JavaScript rendering capability"""
    console.print("\n[bold magenta]Testing JavaScript Rendering[/bold magenta]")

    # Test with a site that requires JS
    extractor = WebExtractor({
        'javascript_rendering': True,
        'max_pages': 1
    })

    test_urls = [
        "https://example.com",  # Simple HTML
        "https://httpbin.org/html",  # Dynamic content
    ]

    for url in test_urls:
        console.print(f"\n[cyan]Testing {url}...[/cyan]")

        result = extractor.extract(url)

        if result.get('text'):
            console.print(f"  [green]✓[/green] Extracted {len(result['text'])} characters")
            console.print(f"  Rendered with: {result.get('metadata', {}).get('rendered_with', 'static')}")
        else:
            console.print(f"  [red]✗[/red] No content extracted")


def test_error_handling():
    """Test error handling with problematic URLs"""
    console.print("\n[bold magenta]Testing Error Handling[/bold magenta]")

    problematic_urls = [
        "https://this-site-definitely-does-not-exist-12345.com",
        "https://httpbin.org/status/404",
        "https://httpbin.org/status/500",
        "https://httpbin.org/delay/10",  # Timeout test
        "not-a-valid-url",
    ]

    for url in problematic_urls:
        console.print(f"\n[cyan]Testing {url}...[/cyan]")

        extractor = WebExtractor({'timeout': 2, 'retry_attempts': 1})
        result = extractor.extract(url)

        if result.get('error'):
            console.print(f"  [green]✓[/green] Error handled: {result['error'][:50]}")
        elif result.get('text'):
            console.print(f"  [yellow]⚠[/yellow] Unexpectedly extracted content")
        else:
            console.print(f"  [green]✓[/green] Returned empty content")


def main():
    """Main test execution"""
    console.print("""
╔════════════════════════════════════════════════╗
║     Dataseter Live Website Testing Suite      ║
╚════════════════════════════════════════════════╝
    """)

    tests = [
        ("Single Website", lambda: test_single_website("https://en.wikipedia.org/wiki/Python_(programming_language)")),
        ("Multiple Websites", test_multiple_websites),
        ("Deep Crawling", test_deep_crawling),
        ("JavaScript Rendering", test_javascript_rendering),
        ("Error Handling", test_error_handling),
    ]

    console.print("[bold]Available tests:[/bold]")
    console.print("0. Run all tests")
    for i, (name, _) in enumerate(tests, 1):
        console.print(f"{i}. {name}")

    console.print("q. Quit")

    choice = input("\nSelect test (0-5, q): ").strip()

    if choice == 'q':
        return

    try:
        choice = int(choice)

        if choice == 0:
            console.print("\n[bold cyan]Running all tests...[/bold cyan]")
            for name, test_func in tests:
                console.print(f"\n{'='*50}")
                console.print(f"[bold]{name}[/bold]")
                console.print('='*50)
                try:
                    test_func()
                except Exception as e:
                    console.print(f"[red]Test failed: {e}[/red]")
                time.sleep(1)
        elif 1 <= choice <= len(tests):
            name, test_func = tests[choice - 1]
            console.print(f"\n[bold cyan]Running {name} test...[/bold cyan]")
            test_func()
        else:
            console.print("[red]Invalid choice[/red]")

    except ValueError:
        console.print("[red]Invalid input[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Testing cancelled[/yellow]")

    console.print("\n[bold green]Testing complete![/bold green]")
    console.print("Check test_results.json for detailed results")


if __name__ == "__main__":
    main()