# Dataseter - Advanced AI Training Dataset Creator

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

## Overview

Dataseter is a comprehensive, production-ready tool for creating high-quality datasets for AI/ML training from diverse sources including PDFs, websites, office documents, and eBooks. It features advanced text extraction, processing pipelines, quality control, and both CLI and web interfaces.

## Key Features

### Multi-Source Data Extraction
- **PDF Processing**: Advanced OCR, layout preservation, table extraction
- **Web Scraping**: Recursive crawling, JavaScript rendering, robots.txt compliance
- **Office Documents**: Word, Excel, PowerPoint, OpenOffice formats
- **eBooks**: EPUB, MOBI, AZW3, FB2 support
- **Plain Text**: TXT, Markdown, RTF, LaTeX

### Advanced Processing
- **Intelligent Chunking**: Semantic, sliding window, and custom strategies
- **Quality Control**: Deduplication, language detection, quality scoring
- **Privacy Protection**: PII detection and redaction
- **Metadata Preservation**: Source tracking, timestamps, authorship
- **Multi-Language Support**: 50+ languages with proper tokenization

### Output Formats
- **AI-Ready Formats**: HuggingFace datasets, TensorFlow records, PyTorch tensors
- **Standard Formats**: JSON, JSONL, CSV, Parquet, Arrow
- **Custom Templates**: Configurable output schemas
- **Token Counting**: GPT, BERT, T5, and custom tokenizers

### Analysis Tools
- **Dataset Statistics**: Size, diversity, quality metrics
- **Visualization**: Distribution plots, word clouds, embeddings
- **Validation**: Schema validation, consistency checks
- **Benchmarking**: Performance metrics, processing speed

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mkupermann/dataseter.git
cd dataseter

# Install with pip
pip install -e .

# Or use Docker
docker pull mkupermann/dataseter:latest
```

### Basic Usage

```python
from dataseter import DatasetCreator

# Initialize creator
creator = DatasetCreator()

# Add sources
creator.add_pdf("documents/report.pdf")
creator.add_website("https://example.com", max_depth=2)
creator.add_directory("./texts", recursive=True)

# Process and create dataset
dataset = creator.process(
    chunk_size=512,
    overlap=50,
    remove_pii=True,
    quality_threshold=0.7
)

# Export
dataset.to_huggingface("my-dataset")
dataset.to_jsonl("output/dataset.jsonl")
```

### CLI Usage

```bash
# Create dataset from multiple sources
dataseter create \
  --pdf documents/*.pdf \
  --website https://docs.example.com \
  --output dataset.jsonl \
  --chunk-size 512 \
  --remove-pii

# Analyze existing dataset
dataseter analyze dataset.jsonl --report analysis.html

# Start web interface
dataseter web --port 8080
```

## Web Interface

Access the intuitive web GUI at `http://localhost:8080` after running:

```bash
dataseter web
```

Features:
- Drag-and-drop file upload
- Real-time processing progress
- Interactive configuration
- Dataset preview and download
- Analysis dashboard

## Project Structure

```
dataseter/
├── src/
│   ├── extractors/       # Data extraction modules
│   │   ├── pdf.py
│   │   ├── web.py
│   │   ├── office.py
│   │   └── ebook.py
│   ├── processors/       # Text processing pipeline
│   │   ├── chunker.py
│   │   ├── cleaner.py
│   │   ├── deduplicator.py
│   │   └── privacy.py
│   ├── analyzers/        # Analysis and validation
│   │   ├── statistics.py
│   │   ├── quality.py
│   │   └── visualizer.py
│   ├── formatters/       # Output format handlers
│   │   ├── huggingface.py
│   │   ├── tensorflow.py
│   │   └── standard.py
│   ├── api/             # REST API
│   ├── cli/             # Command-line interface
│   └── web/             # Web GUI (React frontend)
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Usage examples
├── docker/              # Docker configuration
└── config/              # Configuration files
```

## Configuration

Configure via `config.yaml`:

```yaml
extraction:
  pdf:
    ocr_enabled: true
    extract_tables: true
    preserve_layout: false
  web:
    max_depth: 3
    respect_robots: true
    javascript_rendering: true
    rate_limit: 1.0  # requests per second

processing:
  chunking:
    strategy: "semantic"  # semantic, fixed, sliding_window
    size: 512
    overlap: 50
  quality:
    min_score: 0.7
    remove_duplicates: true
    detect_language: true
  privacy:
    detect_pii: true
    redaction_method: "mask"  # mask, remove, hash

output:
  formats: ["jsonl", "parquet"]
  compression: "gzip"
  include_metadata: true
```

## Advanced Features

### Custom Extractors

```python
from dataseter.extractors import BaseExtractor

class CustomExtractor(BaseExtractor):
    def extract(self, source):
        # Your extraction logic
        return extracted_data

# Register extractor
creator.register_extractor("custom", CustomExtractor())
```

### Pipeline Customization

```python
from dataseter import Pipeline

pipeline = Pipeline()
pipeline.add_step(custom_preprocessor)
pipeline.add_step(quality_filter, threshold=0.8)
pipeline.add_step(custom_chunker)

dataset = creator.process(pipeline=pipeline)
```

### Distributed Processing

```python
from dataseter.distributed import DistributedCreator

creator = DistributedCreator(
    workers=4,
    backend="ray"  # or "dask", "spark"
)
```

## Performance

| Source Type | Processing Speed | Memory Usage |
|------------|------------------|--------------|
| PDF        | ~100 pages/sec   | ~200 MB/1000 pages |
| Web        | ~50 pages/sec    | ~150 MB/1000 pages |
| Office     | ~200 docs/sec    | ~100 MB/1000 docs |
| eBooks     | ~500 pages/sec   | ~50 MB/book |

## Security & Privacy

- **PII Detection**: Automatic detection of names, emails, phone numbers, SSNs
- **Data Encryption**: Optional encryption for sensitive datasets
- **Access Control**: API key authentication, rate limiting
- **Audit Logging**: Complete processing history
- **GDPR Compliance**: Right to deletion, data minimization

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/extractors/

# With coverage
pytest --cov=dataseter --cov-report=html
```

## Documentation

- [Full Documentation](docs/README.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Extractor Guide](docs/extractors.md)
- [Processing Pipeline](docs/processing.md)
- [Examples](examples/README.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Roadmap

- [ ] GPU acceleration for large-scale processing
- [ ] Real-time streaming data support
- [ ] AutoML dataset optimization
- [ ] Multi-modal data support (images, audio)
- [ ] Federated dataset creation
- [ ] Advanced deduplication with MinHash
- [ ] Synthetic data augmentation

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [PyPDF2](https://github.com/py-pdf/pypdf) - PDF processing
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing
- [Scrapy](https://scrapy.org/) - Web scraping
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) - Dataset formatting
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [React](https://reactjs.org/) - Web UI

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/mkupermann/dataseter/issues)
- Email: dataseter@example.com
- Discord: [Join our community](https://discord.gg/dataseter)

---

**Made for the AI community**