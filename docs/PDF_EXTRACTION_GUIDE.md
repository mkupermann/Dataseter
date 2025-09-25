# PDF Extraction Complete Guide

## Table of Contents

1. [Overview](#overview)
2. [PDF Types and Challenges](#pdf-types-and-challenges)
3. [Basic PDF Extraction](#basic-pdf-extraction)
4. [OCR for Scanned PDFs](#ocr-for-scanned-pdfs)
5. [Advanced Extraction Features](#advanced-extraction-features)
6. [Handling Complex PDFs](#handling-complex-pdfs)
7. [Batch Processing](#batch-processing)
8. [Configuration Options](#configuration-options)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Overview

PDF extraction is one of Dataseter's core capabilities. This guide covers everything from simple text extraction to complex OCR operations and table extraction.

### What Dataseter Can Extract from PDFs

- **Text Content**: Main body text, headers, footers
- **Metadata**: Title, author, creation date, keywords
- **Tables**: Structured table data with rows and columns
- **Images**: Embedded images and figures (optional)
- **Forms**: Form fields and their values
- **Annotations**: Comments and highlights
- **Bookmarks**: Document structure and navigation
- **Scanned Text**: Via OCR (Optical Character Recognition)

### PDF Extraction Process Flow

```
PDF Input → Detection → Extraction → Processing → Output
    ↓          ↓           ↓            ↓          ↓
  Load PDF   Text/Scan   Extract    Clean/Fix   Format
             Detection    Content     Issues     Dataset
```

## PDF Types and Challenges

### Type 1: Native Text PDFs

PDFs with embedded text (most common):

```bash
# Easy extraction - text is already digital
dataseter create --pdf document.pdf -o output.jsonl

# Example: Research papers, reports, ebooks
```

**Characteristics**:
- Text is selectable in PDF viewers
- Fast extraction
- High accuracy
- Preserves formatting

### Type 2: Scanned PDFs

PDFs created from scanned images:

```bash
# Requires OCR
dataseter create --pdf scanned.pdf --ocr-enabled -o output.jsonl

# Example: Old books, scanned documents, faxes
```

**Characteristics**:
- Text is not selectable
- Requires OCR processing
- Slower extraction
- Accuracy depends on scan quality

### Type 3: Mixed PDFs

PDFs with both text and scanned content:

```bash
# Automatic detection and handling
dataseter create --pdf mixed.pdf --ocr-fallback -o output.jsonl

# Example: Documents with scanned signatures, mixed archives
```

### Type 4: Protected PDFs

PDFs with encryption or passwords:

```bash
# With password
dataseter create --pdf protected.pdf --pdf-password "secret" -o output.jsonl

# Note: Cannot extract from DRM-protected PDFs
```

### Type 5: Complex Layout PDFs

Multi-column, tables, figures:

```bash
# Preserve layout structure
dataseter create --pdf complex.pdf --preserve-layout -o output.jsonl

# Example: Newspapers, magazines, scientific papers
```

## Basic PDF Extraction

### Simple Text Extraction

#### Example 1: Single PDF

```bash
# Basic extraction
dataseter create --pdf report.pdf -o dataset.jsonl

# With verbose output
dataseter create --pdf report.pdf -o dataset.jsonl --verbose

# Output will show:
# Processing: report.pdf
# Pages: 45
# Extracted: 12,543 words
# Chunks created: 25
# Saved to: dataset.jsonl
```

#### Example 2: Multiple PDFs

```bash
# Multiple files
dataseter create \
  --pdf file1.pdf \
  --pdf file2.pdf \
  --pdf file3.pdf \
  -o combined_dataset.jsonl

# Using wildcards (in bash)
for pdf in *.pdf; do
  echo "--pdf $pdf"
done | xargs dataseter create -o all_pdfs.jsonl
```

#### Example 3: Directory of PDFs

```bash
# Process all PDFs in directory
dataseter create --directory ./pdfs --pattern "*.pdf" -o dataset.jsonl

# Recursive processing
dataseter create --directory ./documents --recursive --pattern "*.pdf" -o dataset.jsonl
```

### Python API Usage

```python
from dataseter import DatasetCreator

# Initialize creator
creator = DatasetCreator()

# Add PDF source
creator.add_pdf("document.pdf")

# Process and create dataset
dataset = creator.process(
    chunk_size=512,
    overlap=50,
    remove_pii=True
)

# Save to file
dataset.to_jsonl("output.jsonl")

# Access extracted data
for document in dataset.documents:
    print(f"Document: {document.source}")
    print(f"Text length: {len(document.text)}")
    print(f"Chunks: {len(document.chunks)}")
```

### Extraction with Metadata

```python
from dataseter import PDFExtractor

# Initialize extractor
extractor = PDFExtractor({
    'extract_metadata': True,
    'extract_outline': True
})

# Extract with metadata
result = extractor.extract("document.pdf")

# Access metadata
print(f"Title: {result['metadata'].get('title')}")
print(f"Author: {result['metadata'].get('author')}")
print(f"Pages: {result['metadata'].get('pages')}")
print(f"Created: {result['metadata'].get('creation_date')}")

# Access outline/bookmarks
for bookmark in result['metadata'].get('outline', []):
    print(f"- {bookmark['title']} (page {bookmark['page']})")
```

## OCR for Scanned PDFs

### Setting Up OCR

#### Prerequisites

```bash
# Install Tesseract OCR
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Install additional languages
sudo apt install tesseract-ocr-deu  # German
sudo apt install tesseract-ocr-fra  # French
sudo apt install tesseract-ocr-spa  # Spanish
```

### Basic OCR Extraction

```bash
# Enable OCR
dataseter create --pdf scanned.pdf --ocr-enabled -o output.jsonl

# Specify language
dataseter create --pdf german.pdf --ocr-enabled --ocr-language deu -o output.jsonl

# Multiple languages
dataseter create --pdf multilingual.pdf --ocr-enabled --ocr-language "eng+deu+fra" -o output.jsonl
```

### Advanced OCR Options

```python
from dataseter import PDFExtractor

# Configure OCR
extractor = PDFExtractor({
    'ocr_enabled': True,
    'ocr_language': 'eng',
    'ocr_dpi': 300,  # Higher DPI for better quality
    'ocr_psm': 3,     # Page segmentation mode
    'ocr_oem': 3,     # OCR engine mode
    'ocr_preprocess': True,  # Preprocess images
})

# Extract with OCR
result = extractor.extract("scanned.pdf")

# Check OCR confidence
for page in result['pages']:
    print(f"Page {page['number']}: Confidence {page['ocr_confidence']}%")
```

### OCR Preprocessing

```python
# Custom preprocessing for better OCR
config = {
    'ocr_enabled': True,
    'ocr_preprocessing': {
        'deskew': True,           # Fix tilted scans
        'denoise': True,          # Remove noise
        'binarize': True,         # Convert to black/white
        'enhance_contrast': True,  # Improve contrast
        'remove_borders': True,    # Remove scan borders
    }
}

extractor = PDFExtractor(config)
result = extractor.extract("poor_quality_scan.pdf")
```

### OCR Performance Optimization

```bash
# Parallel OCR processing
dataseter create \
  --pdf large_scanned.pdf \
  --ocr-enabled \
  --ocr-parallel \
  --max-workers 4 \
  -o output.jsonl

# Selective OCR (only for pages without text)
dataseter create \
  --pdf mixed.pdf \
  --ocr-fallback \
  -o output.jsonl
```

## Advanced Extraction Features

### Table Extraction

```python
from dataseter import PDFExtractor

# Enable table extraction
extractor = PDFExtractor({
    'extract_tables': True,
    'table_strategy': 'lattice',  # or 'stream'
})

result = extractor.extract("document_with_tables.pdf")

# Access tables
for table in result['tables']:
    print(f"Table on page {table['page']}:")
    print(f"Rows: {len(table['data'])}")
    print(f"Columns: {len(table['data'][0]) if table['data'] else 0}")

    # Convert to pandas DataFrame
    import pandas as pd
    df = pd.DataFrame(table['data'][1:], columns=table['data'][0])
    print(df.head())
```

### Image Extraction

```python
# Extract embedded images
extractor = PDFExtractor({
    'extract_images': True,
    'image_format': 'png',
    'image_quality': 95,
    'min_image_size': (100, 100),  # Minimum width, height
})

result = extractor.extract("document_with_images.pdf")

# Access images
for image in result['images']:
    print(f"Image on page {image['page']}:")
    print(f"Size: {image['width']}x{image['height']}")
    print(f"Path: {image['path']}")

    # Image data is also available as base64
    # image_data = image['data']
```

### Form Field Extraction

```python
# Extract PDF forms
extractor = PDFExtractor({
    'extract_forms': True,
})

result = extractor.extract("form.pdf")

# Access form fields
for field in result['form_fields']:
    print(f"Field: {field['name']}")
    print(f"Type: {field['type']}")
    print(f"Value: {field['value']}")
    print(f"Options: {field.get('options', [])}")
```

### Annotation Extraction

```python
# Extract comments and highlights
extractor = PDFExtractor({
    'extract_annotations': True,
})

result = extractor.extract("annotated.pdf")

# Access annotations
for annotation in result['annotations']:
    print(f"Type: {annotation['type']}")  # highlight, note, etc.
    print(f"Page: {annotation['page']}")
    print(f"Content: {annotation['content']}")
    print(f"Author: {annotation.get('author', 'Unknown')}")
```

## Handling Complex PDFs

### Multi-Column Documents

```python
# Handle multi-column layouts
extractor = PDFExtractor({
    'preserve_layout': True,
    'column_detection': True,
    'reading_order': 'left-to-right',  # or 'top-to-bottom'
})

result = extractor.extract("newspaper.pdf")

# Text is extracted in reading order
print(result['text'])
```

### Scientific Papers

```python
# Optimized for academic papers
config = {
    'extract_tables': True,
    'extract_figures': True,
    'extract_citations': True,
    'extract_equations': True,
    'preserve_formatting': True,
}

extractor = PDFExtractor(config)
result = extractor.extract("research_paper.pdf")

# Access structured content
print(f"Abstract: {result['sections'].get('abstract')}")
print(f"Citations: {len(result['citations'])}")
print(f"Equations: {len(result['equations'])}")
```

### Legal Documents

```python
# Optimized for legal documents
config = {
    'preserve_layout': True,
    'extract_headers_footers': True,
    'extract_page_numbers': True,
    'preserve_indentation': True,
    'extract_line_numbers': True,
}

extractor = PDFExtractor(config)
result = extractor.extract("legal_contract.pdf")
```

### Large PDFs

```python
# Stream processing for large files
from dataseter import PDFExtractor

extractor = PDFExtractor({
    'streaming': True,
    'max_memory': '1GB',
})

# Process in chunks
for chunk in extractor.extract_stream("huge_document.pdf", chunk_size=10):
    print(f"Processing pages {chunk['start_page']}-{chunk['end_page']}")
    # Process chunk
    process_chunk(chunk['text'])
```

## Batch Processing

### Parallel Processing

```python
from dataseter import DatasetCreator
from pathlib import Path

# Get all PDFs
pdf_files = list(Path("./documents").glob("**/*.pdf"))

# Create dataset with parallel processing
creator = DatasetCreator()

# Add all PDFs
for pdf in pdf_files:
    creator.add_pdf(str(pdf))

# Process in parallel
dataset = creator.process(
    parallel=True,
    max_workers=8,
    chunk_size=512,
    show_progress=True
)

# Save results
dataset.to_jsonl("batch_output.jsonl")
```

### Progress Monitoring

```python
from dataseter import DatasetCreator
from tqdm import tqdm

# With progress bar
creator = DatasetCreator()

# Add PDFs with progress
pdf_files = Path("./pdfs").glob("*.pdf")
for pdf in tqdm(pdf_files, desc="Adding PDFs"):
    creator.add_pdf(str(pdf))

# Process with callbacks
def on_progress(current, total, message):
    print(f"Progress: {current}/{total} - {message}")

dataset = creator.process(
    progress_callback=on_progress
)
```

### Error Handling

```python
from dataseter import PDFExtractor
import logging

logging.basicConfig(level=logging.INFO)

extractor = PDFExtractor({
    'continue_on_error': True,
    'log_errors': True,
})

results = []
errors = []

pdf_files = Path("./pdfs").glob("*.pdf")
for pdf in pdf_files:
    try:
        result = extractor.extract(str(pdf))
        results.append(result)
    except Exception as e:
        logging.error(f"Failed to extract {pdf}: {e}")
        errors.append({'file': str(pdf), 'error': str(e)})

print(f"Successfully processed: {len(results)}")
print(f"Failed: {len(errors)}")

# Save error log
import json
with open("extraction_errors.json", "w") as f:
    json.dump(errors, f, indent=2)
```

## Configuration Options

### Complete Configuration Reference

```yaml
extraction:
  pdf:
    # Basic options
    max_pages: null              # Limit pages to extract (null = all)
    start_page: 1                # Starting page number
    end_page: null               # Ending page number

    # Text extraction
    preserve_layout: false       # Maintain original layout
    preserve_formatting: false   # Keep bold, italic, etc.
    extract_headers_footers: true
    extract_page_numbers: true

    # OCR options
    ocr_enabled: false          # Enable OCR for scanned content
    ocr_language: eng           # Tesseract language code
    ocr_dpi: 300                # DPI for image conversion
    ocr_timeout: 60             # Timeout per page in seconds
    ocr_fallback: true          # Use OCR if no text found

    # Table extraction
    extract_tables: true
    table_strategy: lattice     # lattice or stream
    table_areas: []             # Specific areas [[x1,y1,x2,y2]]

    # Image extraction
    extract_images: false
    image_format: png
    image_quality: 95
    min_image_size: [100, 100]

    # Advanced options
    password: null              # PDF password if protected
    backend: auto               # auto, pdfplumber, pymupdf, pypdf2
    encoding: utf-8             # Text encoding

    # Performance
    cache_enabled: true
    cache_directory: ~/.dataseter/cache
    parallel_pages: false       # Process pages in parallel
    max_workers: 4             # Worker threads for parallel
```

### Environment Variables

```bash
# OCR configuration
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
export OMP_THREAD_LIMIT=4

# PDF processing
export DATASETER_PDF_BACKEND=pymupdf
export DATASETER_PDF_CACHE=/tmp/pdf_cache
export DATASETER_PDF_TIMEOUT=120

# Memory limits
export DATASETER_MAX_PDF_SIZE=500MB
export DATASETER_MAX_PAGE_SIZE=50MB
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: OCR not detecting text

```bash
# Solution 1: Increase DPI
dataseter create --pdf scan.pdf --ocr-enabled --ocr-dpi 600 -o output.jsonl

# Solution 2: Preprocess image
dataseter create --pdf scan.pdf --ocr-enabled --ocr-preprocess -o output.jsonl

# Solution 3: Try different PSM
dataseter create --pdf scan.pdf --ocr-enabled --ocr-psm 6 -o output.jsonl
```

#### Issue: Tables not extracted correctly

```python
# Try different strategies
extractor = PDFExtractor({
    'extract_tables': True,
    'table_strategy': 'stream',  # Try 'lattice' if stream fails
    'table_settings': {
        'vertical_strategy': 'lines',
        'horizontal_strategy': 'lines',
        'snap_tolerance': 3,
        'join_tolerance': 3,
    }
})
```

#### Issue: Memory errors with large PDFs

```bash
# Use streaming mode
dataseter create --pdf large.pdf --streaming --max-memory 1GB -o output.jsonl

# Process in batches
dataseter create --pdf large.pdf --batch-size 10 --page-batch -o output.jsonl
```

#### Issue: Corrupted or malformed PDFs

```python
# Use repair mode
extractor = PDFExtractor({
    'repair_pdf': True,
    'strict_mode': False,
    'continue_on_error': True,
})

try:
    result = extractor.extract("corrupted.pdf")
except Exception as e:
    print(f"Extraction failed: {e}")
    # Try alternative backend
    extractor.config['backend'] = 'pypdf2'
    result = extractor.extract("corrupted.pdf")
```

## Best Practices

### 1. Preprocessing PDFs

```bash
# Optimize PDFs before extraction
# Reduce file size
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=optimized.pdf input.pdf

# Fix corrupted PDFs
qpdf --replace-input corrupted.pdf

# Remove password protection (if legal)
qpdf --decrypt --password=PASSWORD input.pdf output.pdf
```

### 2. Quality Checks

```python
# Validate extraction quality
def validate_extraction(result):
    # Check text length
    if len(result['text']) < 100:
        return False, "Text too short"

    # Check language
    from langdetect import detect
    if detect(result['text']) != 'en':
        return False, "Wrong language"

    # Check character ratio
    alnum_ratio = sum(c.isalnum() for c in result['text']) / len(result['text'])
    if alnum_ratio < 0.5:
        return False, "Too many special characters"

    return True, "Valid"

# Use validation
result = extractor.extract("document.pdf")
is_valid, message = validate_extraction(result)
if not is_valid:
    logging.warning(f"Quality issue: {message}")
```

### 3. Performance Optimization

```python
# Cache extracted content
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def extract_pdf_cached(pdf_path):
    # Generate cache key
    with open(pdf_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    cache_file = f"cache/{file_hash}.json"

    # Check cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Extract and cache
    result = extractor.extract(pdf_path)
    with open(cache_file, 'w') as f:
        json.dump(result, f)

    return result
```

### 4. Handling Different Languages

```python
# Multi-language extraction
def detect_and_extract(pdf_path):
    # Quick language detection
    extractor = PDFExtractor({'max_pages': 3})
    sample = extractor.extract(pdf_path)

    from langdetect import detect
    language = detect(sample['text'])

    # Map to Tesseract code
    lang_map = {
        'en': 'eng',
        'de': 'deu',
        'fr': 'fra',
        'es': 'spa',
    }

    # Extract with correct language
    extractor = PDFExtractor({
        'ocr_enabled': True,
        'ocr_language': lang_map.get(language, 'eng')
    })

    return extractor.extract(pdf_path)
```

### 5. Logging and Monitoring

```python
import logging
from datetime import datetime

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'extraction_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)

# Monitor extraction
class ExtractionMonitor:
    def __init__(self):
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_pages': 0,
            'total_chars': 0,
            'start_time': datetime.now()
        }

    def process_pdf(self, pdf_path):
        self.stats['total'] += 1
        try:
            result = extractor.extract(pdf_path)
            self.stats['success'] += 1
            self.stats['total_pages'] += result['metadata'].get('pages', 0)
            self.stats['total_chars'] += len(result['text'])
            logging.info(f"Success: {pdf_path}")
            return result
        except Exception as e:
            self.stats['failed'] += 1
            logging.error(f"Failed: {pdf_path} - {e}")
            return None

    def report(self):
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        print(f"Extraction Report:")
        print(f"  Total: {self.stats['total']}")
        print(f"  Success: {self.stats['success']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Pages: {self.stats['total_pages']}")
        print(f"  Characters: {self.stats['total_chars']:,}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Speed: {self.stats['total_pages']/duration:.2f} pages/sec")
```

## Summary

This guide covered:

- Different PDF types and their challenges
- Basic and advanced extraction techniques
- OCR setup and optimization
- Table, image, and form extraction
- Batch processing strategies
- Configuration options
- Troubleshooting common issues
- Best practices for production use

For more information:
- [Web Scraping Guide](WEB_SCRAPING_GUIDE.md)
- [Processing Pipeline Guide](PROCESSING_GUIDE.md)
- [API Reference](API_REFERENCE.md)