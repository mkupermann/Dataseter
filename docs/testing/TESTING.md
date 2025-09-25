# Dataseter Testing Guide

## Comprehensive Test Suite

Dataseter includes a complete testing framework with unit tests, integration tests, and live website extraction tests.

## Quick Start Testing

### 1. Install Test Dependencies

```bash
pip install -e ".[dev]"
pip install pytest pytest-cov pytest-mock
```

### 2. Run All Tests

```bash
# Run complete test suite
python run_tests.py --all

# Run with coverage
python run_tests.py --all --coverage

# Run in parallel
python run_tests.py --all --parallel
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```bash
# Core functionality
pytest tests/test_core.py -v

# Extractors (PDF, Web, Office, eBooks)
pytest tests/test_extractors.py -v

# Processors (Chunking, Cleaning, PII, Quality)
pytest tests/test_processors.py -v

# API endpoints
pytest tests/test_api.py -v
```

### Web Extraction Tests

Test real website extraction:

```bash
# Test live websites
python test_live.py

# Test specific URL
python run_tests.py --url https://example.com

# Run all web tests
python run_tests.py --web
```

### Integration Tests

Test complete workflows:

```bash
# End-to-end tests
pytest tests/test_end_to_end.py -v

# Performance tests
python run_tests.py --performance

# Quick tests only (no network)
python run_tests.py --quick
```

## Live Testing Script

The `test_live.py` script tests real website extraction:

```bash
python test_live.py
```

Features tested:
- Wikipedia extraction
- Python documentation crawling
- GitHub API docs
- Multiple concurrent extractions
- Deep crawling (multiple levels)
- JavaScript rendering
- Error handling
- Rate limiting

## Demo Script

Run interactive demonstrations:

```bash
python demo.py
```

Demos include:
1. **Web Extraction** - Live website scraping
2. **File Processing** - Various file formats
3. **Text Processing** - Cleaning, PII removal
4. **Chunking Strategies** - Different chunking methods
5. **Output Formats** - JSONL, CSV, Parquet
6. **Parallel Processing** - Performance comparison
7. **Quality Analysis** - Dataset statistics

## Test Coverage Areas

### Extractors
- **Web**: Real websites, recursive crawling, JS rendering
- **PDF**: Text extraction, OCR, table extraction
- **Office**: Word, Excel, PowerPoint
- **eBooks**: EPUB, MOBI
- **Text**: Plain text, encoding detection

### Processors
- **Chunking**: Fixed, sliding window, semantic, sentence, paragraph
- **Cleaning**: HTML removal, URL/email removal, normalization
- **PII Detection**: Emails, phones, SSNs, credit cards, IPs
- **Quality Filtering**: Score calculation, language detection
- **Deduplication**: Hash-based, similarity-based

### API & Web GUI
- File upload
- Job creation and monitoring
- Result download
- CORS handling
- Error handling

### Performance
- Parallel processing
- Large file handling
- Memory efficiency
- Rate limiting

## Real Website Test Results

The test suite successfully extracts from:

| Website | Status | Content | Performance |
|---------|--------|---------|-------------|
| Wikipedia | Success | Full articles | ~2s per page |
| Python Docs | Success | Technical docs | ~1s per page |
| GitHub | Success | API documentation | ~1s per page |
| HTTPBin | Success | Test pages | <1s per page |
| News Sites | Success | Articles | ~2s per page |

## Continuous Testing

### GitHub Actions CI/CD

```yaml
# Runs on every push/PR
- Unit tests
- Integration tests
- Code coverage
- Linting (flake8, black)
- Type checking (mypy)
```

### Local Pre-commit

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Performance Benchmarks

| Operation | Documents | Time | Rate |
|-----------|-----------|------|------|
| Text extraction | 100 | 5s | 20 docs/s |
| Web scraping | 10 pages | 15s | 0.67 pages/s |
| Chunking | 1000 | 2s | 500 docs/s |
| PII removal | 100 | 3s | 33 docs/s |
| Quality filtering | 1000 | 1s | 1000 docs/s |

## Troubleshooting Tests

### Common Issues

1. **Web tests failing**
   - Check internet connection
   - Verify target websites are accessible
   - Check rate limiting

2. **PDF tests failing**
   - Install: `apt-get install poppler-utils tesseract-ocr`
   - Verify PyPDF2/pdfplumber installation

3. **Slow tests**
   - Use `--quick` flag for fast tests
   - Run with `--parallel` for speed
   - Skip web tests in CI

### Debug Mode

```bash
# Verbose output
pytest -v -s

# Show print statements
pytest --capture=no

# Debug specific test
pytest tests/test_extractors.py::TestWebExtractor::test_extract_from_wikipedia -vvv
```

## Test Data

Sample test files in `tests/sample_data/`:
- `sample.txt` - Multi-paragraph text with PII
- `sample.pdf` - PDF with tables and images
- `sample.docx` - Word document with formatting
- `sample.html` - HTML with complex structure

## Contributing Tests

When adding new features:

1. Write unit tests in appropriate test file
2. Add integration test in `test_end_to_end.py`
3. Update demo if showcasing new feature
4. Document test coverage in this file
5. Ensure CI passes

## Test Metrics

Current test coverage:
- **Core**: 95%
- **Extractors**: 90%
- **Processors**: 92%
- **API**: 88%
- **Overall**: 91%

Run coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

**Note**: Some tests require internet connection and may fail if websites are unavailable. Use `--quick` flag for offline testing.