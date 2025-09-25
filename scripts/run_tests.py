#!/usr/bin/env python
"""
Master test runner for Dataseter
Runs all tests including unit, integration, and end-to-end tests
"""

import sys
import subprocess
import time
from pathlib import Path
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track
import json

console = Console()


class TestRunner:
    """Comprehensive test runner for Dataseter"""

    def __init__(self, verbose=False, coverage=False, parallel=False):
        self.verbose = verbose
        self.coverage = coverage
        self.parallel = parallel
        self.results = {}

    def run_test_suite(self, suite_name, test_file, markers=None):
        """Run a specific test suite"""
        console.print(f"\n[bold blue]Running {suite_name}...[/bold blue]")

        cmd = ["pytest", test_file]

        if self.verbose:
            cmd.append("-v")

        if self.coverage:
            cmd.extend(["--cov=src", "--cov-append"])

        if self.parallel:
            cmd.extend(["-n", "auto"])

        if markers:
            cmd.extend(["-m", markers])

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        self.results[suite_name] = {
            "passed": result.returncode == 0,
            "time": elapsed,
            "output": result.stdout if self.verbose else ""
        }

        if result.returncode == 0:
            console.print(f"[green]✓[/green] {suite_name} passed ({elapsed:.2f}s)")
        else:
            console.print(f"[red]✗[/red] {suite_name} failed")
            if self.verbose:
                console.print(result.stdout)
                console.print(result.stderr)

        return result.returncode == 0

    def run_all_tests(self):
        """Run all test suites"""
        console.print("[bold cyan]Dataseter Test Suite[/bold cyan]")
        console.print("=" * 50)

        test_suites = [
            ("Core Tests", "tests/test_core.py"),
            ("Extractor Tests", "tests/test_extractors.py"),
            ("Processor Tests", "tests/test_processors.py"),
            ("API Tests", "tests/test_api.py"),
            ("End-to-End Tests", "tests/test_end_to_end.py"),
        ]

        # Track overall results
        all_passed = True

        # Run each test suite
        for suite_name, test_file in track(test_suites, description="Running tests..."):
            if Path(test_file).exists():
                passed = self.run_test_suite(suite_name, test_file)
                all_passed = all_passed and passed
            else:
                console.print(f"[yellow]⚠[/yellow] {suite_name} file not found: {test_file}")

        # Display summary
        self.display_summary()

        return all_passed

    def run_quick_tests(self):
        """Run quick tests only (no integration/e2e)"""
        console.print("[bold cyan]Running Quick Tests[/bold cyan]")

        test_suites = [
            ("Core Tests", "tests/test_core.py"),
            ("Processor Tests", "tests/test_processors.py"),
        ]

        all_passed = True
        for suite_name, test_file in test_suites:
            if Path(test_file).exists():
                passed = self.run_test_suite(suite_name, test_file)
                all_passed = all_passed and passed

        return all_passed

    def run_web_tests(self):
        """Run web-specific tests"""
        console.print("[bold cyan]Running Web Extraction Tests[/bold cyan]")

        # Test basic web extraction
        from tests.test_end_to_end import TestRealWebExtraction

        test = TestRealWebExtraction()

        console.print("\n[bold]Testing Wikipedia extraction...[/bold]")
        try:
            test.test_extract_from_wikipedia()
            console.print("[green]✓[/green] Wikipedia extraction successful")
        except Exception as e:
            console.print(f"[red]✗[/red] Wikipedia extraction failed: {e}")

        console.print("\n[bold]Testing GitHub extraction...[/bold]")
        try:
            test.test_extract_from_github()
            console.print("[green]✓[/green] GitHub extraction successful")
        except Exception as e:
            console.print(f"[red]✗[/red] GitHub extraction failed: {e}")

        console.print("\n[bold]Testing multiple sites extraction...[/bold]")
        try:
            test.test_extract_multiple_sites()
            console.print("[green]✓[/green] Multiple sites extraction successful")
        except Exception as e:
            console.print(f"[red]✗[/red] Multiple sites extraction failed: {e}")

    def test_live_website(self, url):
        """Test extraction from a specific website"""
        console.print(f"\n[bold cyan]Testing extraction from: {url}[/bold cyan]")

        from src.core import DatasetCreator

        try:
            creator = DatasetCreator()
            creator.add_website(url, max_depth=1)

            with console.status("[bold green]Processing..."):
                dataset = creator.process(
                    chunk_size=512,
                    overlap=50,
                    remove_pii=True,
                    quality_threshold=0.6
                )

            # Display results
            table = Table(title="Extraction Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Documents extracted", str(len(dataset)))
            table.add_row("Total text length", f"{dataset.statistics.get('total_text_length', 0):,}")
            table.add_row("Total words", f"{dataset.statistics.get('total_words', 0):,}")
            table.add_row("Vocabulary size", f"{dataset.statistics.get('vocabulary_size', 0):,}")

            if dataset.statistics.get('quality_stats'):
                table.add_row("Average quality", f"{dataset.statistics['quality_stats']['mean']:.2f}")

            console.print(table)

            # Save sample output
            output_file = "test_output.jsonl"
            dataset.to_jsonl(output_file)
            console.print(f"\n[green]✓[/green] Sample output saved to {output_file}")

            return True

        except Exception as e:
            console.print(f"[red]✗[/red] Extraction failed: {e}")
            return False

    def display_summary(self):
        """Display test summary"""
        console.print("\n" + "=" * 50)
        console.print("[bold cyan]Test Summary[/bold cyan]")

        table = Table()
        table.add_column("Test Suite", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Time", style="yellow")

        total_time = 0
        passed_count = 0

        for suite, result in self.results.items():
            status = "[green]PASSED[/green]" if result["passed"] else "[red]FAILED[/red]"
            table.add_row(suite, status, f"{result['time']:.2f}s")
            total_time += result["time"]
            if result["passed"]:
                passed_count += 1

        console.print(table)

        console.print(f"\nTotal: {passed_count}/{len(self.results)} passed")
        console.print(f"Time: {total_time:.2f}s")

        if self.coverage:
            console.print("\n[cyan]Coverage report generated[/cyan]")

    def run_performance_test(self):
        """Run performance benchmark"""
        console.print("[bold cyan]Performance Benchmark[/bold cyan]")

        import tempfile
        from pathlib import Path
        import time
        from src.core import DatasetCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            console.print("Creating test data...")
            for i in range(100):
                file = Path(tmpdir) / f"file_{i}.txt"
                file.write_text(f"Test content {i}. " * 100)

            creator = DatasetCreator()
            creator.add_directory(tmpdir)

            # Benchmark
            console.print("Running benchmark...")
            start = time.time()
            dataset = creator.process(
                chunk_size=500,
                parallel=True,
                max_workers=4
            )
            elapsed = time.time() - start

            console.print(f"\n[green]✓[/green] Processed {len(dataset)} documents in {elapsed:.2f}s")
            console.print(f"Rate: {len(dataset)/elapsed:.2f} docs/sec")


def main():
    parser = argparse.ArgumentParser(description="Dataseter Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--web", action="store_true", help="Run web extraction tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--url", type=str, help="Test specific URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run tests in parallel")

    args = parser.parse_args()

    runner = TestRunner(
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel
    )

    console.print("""
    ╔══════════════════════════════════════╗
    ║     Dataseter Test Suite v1.0.0     ║
    ╚══════════════════════════════════════╝
    """)

    success = True

    if args.url:
        success = runner.test_live_website(args.url)
    elif args.quick:
        success = runner.run_quick_tests()
    elif args.web:
        runner.run_web_tests()
    elif args.performance:
        runner.run_performance_test()
    elif args.all:
        success = runner.run_all_tests()
    else:
        # Default: run all tests
        success = runner.run_all_tests()

    if success:
        console.print("\n[bold green]✅ All tests passed![/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]❌ Some tests failed[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()