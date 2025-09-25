#!/usr/bin/env python
"""
Dataseter CLI - Command-line interface for dataset creation
"""

import click
import yaml
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint

from ..core import DatasetCreator

console = Console()

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Dataseter - Advanced AI Training Dataset Creator"""
    pass

@cli.command()
@click.option('--pdf', multiple=True, help='PDF files to process')
@click.option('--website', multiple=True, help='Websites to scrape')
@click.option('--directory', help='Directory of files to process')
@click.option('--config', help='Configuration file path')
@click.option('--output', '-o', default='dataset.jsonl', help='Output file path')
@click.option('--format', '-f',
              type=click.Choice(['jsonl', 'parquet', 'csv', 'huggingface']),
              default='jsonl', help='Output format')
@click.option('--chunk-size', default=512, help='Chunk size for text splitting')
@click.option('--overlap', default=50, help='Overlap between chunks')
@click.option('--quality-threshold', default=0.7, help='Minimum quality score')
@click.option('--remove-pii/--keep-pii', default=True, help='Remove PII from text')
@click.option('--remove-duplicates/--keep-duplicates', default=True, help='Remove duplicate content')
@click.option('--parallel/--sequential', default=True, help='Process in parallel')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def create(**kwargs):
    """Create a dataset from various sources"""
    console.print("[bold blue]Dataseter - Creating Dataset[/bold blue]")

    # Load configuration
    config = {}
    if kwargs['config']:
        with open(kwargs['config'], 'r') as f:
            config = yaml.safe_load(f)

    # Initialize creator
    creator = DatasetCreator(config_path=kwargs['config'])

    # Add sources
    source_count = 0

    # Add PDFs
    for pdf_path in kwargs['pdf']:
        creator.add_pdf(pdf_path)
        source_count += 1
        if kwargs['verbose']:
            console.print(f"[green]✓[/green] Added PDF: {pdf_path}")

    # Add websites
    for url in kwargs['website']:
        creator.add_website(url, max_depth=2)
        source_count += 1
        if kwargs['verbose']:
            console.print(f"[green]✓[/green] Added website: {url}")

    # Add directory
    if kwargs['directory']:
        creator.add_directory(kwargs['directory'], recursive=True)
        source_count += len(list(Path(kwargs['directory']).rglob('*')))
        if kwargs['verbose']:
            console.print(f"[green]✓[/green] Added directory: {kwargs['directory']}")

    if source_count == 0:
        console.print("[red]Error: No sources specified[/red]")
        return

    console.print(f"\n[cyan]Processing {source_count} sources...[/cyan]")

    # Process dataset
    with console.status("[bold green]Processing dataset...") as status:
        dataset = creator.process(
            chunk_size=kwargs['chunk_size'],
            overlap=kwargs['overlap'],
            quality_threshold=kwargs['quality_threshold'],
            remove_pii=kwargs['remove_pii'],
            remove_duplicates=kwargs['remove_duplicates'],
            parallel=kwargs['parallel']
        )

    # Save output
    output_path = kwargs['output']
    console.print(f"\n[cyan]Saving to {output_path}...[/cyan]")

    if kwargs['format'] == 'jsonl':
        dataset.to_jsonl(output_path)
    elif kwargs['format'] == 'parquet':
        dataset.to_parquet(output_path)
    elif kwargs['format'] == 'csv':
        dataset.to_csv(output_path)
    elif kwargs['format'] == 'huggingface':
        dataset.to_huggingface(output_path)

    # Display statistics
    display_statistics(dataset.statistics)

    console.print(f"\n[bold green]✓ Dataset created successfully![/bold green]")
    console.print(f"Output: {output_path}")
    console.print(f"Documents: {len(dataset)}")

@cli.command()
@click.argument('dataset_path')
@click.option('--report', help='Generate HTML report')
@click.option('--verbose', '-v', is_flag=True, help='Detailed analysis')
def analyze(dataset_path, report, verbose):
    """Analyze an existing dataset"""
    console.print(f"[bold blue]Analyzing dataset: {dataset_path}[/bold blue]\n")

    # Load dataset
    with open(dataset_path, 'r') as f:
        if dataset_path.endswith('.jsonl'):
            documents = [json.loads(line) for line in f]
        else:
            console.print("[red]Only JSONL format supported for analysis[/red]")
            return

    # Calculate statistics
    stats = {
        'total_documents': len(documents),
        'total_text_length': sum(len(doc.get('text', '')) for doc in documents),
        'avg_text_length': sum(len(doc.get('text', '')) for doc in documents) / len(documents) if documents else 0
    }

    # Display statistics
    display_statistics(stats)

    if report:
        console.print(f"\n[cyan]Generating report: {report}[/cyan]")
        generate_html_report(stats, report)

@cli.command()
@click.option('--port', '-p', default=8080, help='Port to run web interface')
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind to')
@click.option('--debug', is_flag=True, help='Run in debug mode')
def web(port, host, debug):
    """Start the web interface"""
    console.print(f"[bold blue]Starting Dataseter Web Interface[/bold blue]")
    console.print(f"Host: {host}:{port}")

    import uvicorn
    from ..api.main import app

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info"
    )

@cli.command()
def config():
    """Generate example configuration file"""
    config_path = "dataseter_config.yaml"

    default_config = {
        'extraction': {
            'pdf': {
                'ocr_enabled': True,
                'extract_tables': True
            },
            'web': {
                'max_depth': 3,
                'respect_robots': True,
                'javascript_rendering': False
            }
        },
        'processing': {
            'chunking': {
                'strategy': 'sliding_window',
                'size': 512,
                'overlap': 50
            },
            'quality': {
                'min_score': 0.7,
                'detect_language': True
            },
            'privacy': {
                'detect_pii': True,
                'redaction_method': 'mask'
            }
        },
        'output': {
            'formats': ['jsonl', 'parquet'],
            'compression': 'gzip'
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)

    console.print(f"[green]✓[/green] Configuration file created: {config_path}")

def display_statistics(stats):
    """Display statistics in a formatted table"""
    if not stats:
        return

    table = Table(title="Dataset Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                table.add_row(f"{key}.{sub_key}", str(sub_value))
        else:
            table.add_row(key, str(value))

    console.print(table)

def generate_html_report(stats, output_path):
    """Generate HTML analysis report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Dataset Analysis Report</h1>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stats.items())}
        </table>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html)

if __name__ == '__main__':
    cli()