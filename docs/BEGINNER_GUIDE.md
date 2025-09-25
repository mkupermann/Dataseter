# Dataseter Beginner's Guide

## Table of Contents

1. [Introduction](#introduction)
2. [What is Dataseter?](#what-is-dataseter)
3. [Core Concepts](#core-concepts)
4. [Getting Started](#getting-started)
5. [Your First Dataset](#your-first-dataset)
6. [Understanding the Workflow](#understanding-the-workflow)
7. [Common Use Cases](#common-use-cases)
8. [Tips for Beginners](#tips-for-beginners)

## Introduction

Welcome to Dataseter! This guide is designed for absolute beginners who want to create high-quality datasets for training AI models. Whether you're a data scientist, machine learning engineer, or researcher, this guide will help you get started quickly and effectively.

## What is Dataseter?

Dataseter is a powerful tool that automatically extracts and processes text data from various sources to create clean, structured datasets suitable for training AI models. Think of it as a smart data collector that:

- **Extracts** text from PDFs, websites, Office documents, and eBooks
- **Cleans** the extracted text to remove noise and irrelevant content
- **Processes** the text into uniform chunks suitable for AI training
- **Filters** content for quality and removes sensitive information
- **Exports** the final dataset in formats compatible with popular AI frameworks

### Why Use Dataseter?

Creating training datasets manually is:
- Time-consuming (can take weeks or months)
- Error-prone (human mistakes in formatting)
- Inconsistent (different people process data differently)
- Expensive (requires significant human resources)

Dataseter automates this entire process, reducing weeks of work to hours or minutes.

## Core Concepts

Before diving into the practical usage, let's understand the key concepts:

### 1. Data Sources

Dataseter can extract data from multiple source types:

- **PDF Files**: Research papers, reports, documentation, scanned documents
- **Websites**: News articles, blogs, documentation sites, forums
- **Office Documents**: Word documents, Excel spreadsheets, PowerPoint presentations
- **eBooks**: EPUB, MOBI, AZW formats containing structured text

### 2. Extraction Process

The extraction process involves:

- **Text Extraction**: Getting raw text from the source
- **Metadata Preservation**: Keeping important information like titles, authors, dates
- **Structure Recognition**: Understanding headings, paragraphs, lists, tables
- **OCR (Optical Character Recognition)**: Converting scanned images to text

### 3. Processing Pipeline

After extraction, text goes through a processing pipeline:

```
Raw Text → Cleaning → Chunking → Quality Check → PII Removal → Deduplication → Final Dataset
```

Each step refines the data:

- **Cleaning**: Removes extra whitespace, fixes encoding issues, normalizes text
- **Chunking**: Splits text into manageable pieces (e.g., 512 tokens)
- **Quality Check**: Filters out low-quality content
- **PII Removal**: Detects and removes personal information
- **Deduplication**: Removes duplicate or near-duplicate content

### 4. Output Formats

Dataseter supports multiple output formats:

- **JSONL**: Line-delimited JSON, perfect for streaming large datasets
- **Parquet**: Columnar format, efficient for analytics and machine learning
- **CSV**: Simple tabular format, compatible with spreadsheet software
- **HuggingFace**: Direct integration with HuggingFace datasets library

## Getting Started

### Prerequisites

Before installing Dataseter, ensure you have:

1. **Python 3.8 or higher**: Check with `python --version`
2. **pip package manager**: Usually comes with Python
3. **At least 4GB RAM**: For processing large documents
4. **10GB free disk space**: For temporary files and output

### Installation Methods

#### Method 1: Using pip (Recommended for Beginners)

```bash
# Install from PyPI
pip install dataseter

# Or install with all optional dependencies
pip install dataseter[all]
```

#### Method 2: Using Docker (No Python Required)

```bash
# Pull the Docker image
docker pull mkupermann/dataseter:latest

# Run with Docker
docker run -it mkupermann/dataseter:latest
```

#### Method 3: From Source (For Developers)

```bash
# Clone the repository
git clone https://github.com/mkupermann/dataseter.git
cd dataseter

# Install in development mode
pip install -e .
```

### Verifying Installation

After installation, verify everything works:

```bash
# Check installation
dataseter --version

# View available commands
dataseter --help

# Run a simple test
echo "Hello World" > test.txt
dataseter create --text test.txt -o test_output.jsonl
```

## Your First Dataset

Let's create your first dataset step by step. We'll start with a simple example and gradually add complexity.

### Example 1: Single PDF File

Suppose you have a PDF file called `research_paper.pdf`:

```bash
# Basic extraction
dataseter create --pdf research_paper.pdf -o my_first_dataset.jsonl

# What happens:
# 1. Dataseter reads the PDF file
# 2. Extracts all text content
# 3. Cleans and processes the text
# 4. Saves it to my_first_dataset.jsonl
```

### Example 2: Multiple PDFs

Processing multiple PDFs at once:

```bash
# Process multiple PDFs
dataseter create \
  --pdf paper1.pdf \
  --pdf paper2.pdf \
  --pdf paper3.pdf \
  -o research_dataset.jsonl

# Or process all PDFs in a directory
dataseter create --directory ./pdfs -o all_pdfs_dataset.jsonl
```

### Example 3: Website Scraping

Extract content from a website:

```bash
# Scrape a single page
dataseter create --website https://example.com/article -o web_dataset.jsonl

# Scrape with depth (follows links)
dataseter create \
  --website https://example.com \
  --max-depth 2 \
  -o website_dataset.jsonl
```

### Example 4: Mixed Sources

Combine different source types:

```bash
dataseter create \
  --pdf report.pdf \
  --website https://docs.example.com \
  --directory ./documents \
  -o combined_dataset.jsonl
```

### Understanding the Output

After running the command, you'll see output like:

```
Dataseter - Creating Dataset
✓ Added PDF: report.pdf
✓ Added website: https://docs.example.com
✓ Added directory: ./documents

Processing 15 sources...
Extracting sources: 100%|████████| 15/15
Processing documents: 100%|████████| 15/15

Dataset created with 15 documents
Saved to combined_dataset.jsonl

Statistics:
- Total documents: 15
- Total chunks: 450
- Average chunk size: 498 tokens
- Total size: 2.3 MB
```

## Understanding the Workflow

### Step 1: Planning Your Dataset

Before starting, ask yourself:

1. **What is the purpose?** Training a chatbot, document classifier, or something else?
2. **What sources do I need?** Technical documentation, news articles, research papers?
3. **How much data?** Start small (100-1000 documents) and scale up
4. **What quality level?** Higher quality is better than quantity

### Step 2: Preparing Your Sources

Organize your sources:

```
my_dataset_project/
├── pdfs/
│   ├── research_papers/
│   └── reports/
├── websites.txt        # List of URLs
├── config.yaml         # Configuration file
└── output/            # Output directory
```

### Step 3: Configuration

Create a configuration file (`config.yaml`) for reusable settings:

```yaml
general:
  log_level: INFO
  max_workers: 4

extraction:
  pdf:
    ocr_enabled: true
    extract_tables: true
  web:
    javascript_rendering: false
    max_depth: 2
    rate_limit: 1.0

processing:
  chunking:
    method: sliding_window
    chunk_size: 512
    chunk_overlap: 50
  quality:
    min_length: 100
    max_length: 10000
    language: en
  privacy:
    remove_emails: true
    remove_phone_numbers: true
    remove_ssn: true
```

### Step 4: Running Extraction

Use your configuration:

```bash
dataseter create --config config.yaml --directory ./pdfs -o output/dataset.jsonl
```

### Step 5: Monitoring Progress

Dataseter provides real-time progress updates:

- **Extraction Progress**: Shows which files are being processed
- **Processing Progress**: Shows pipeline stages
- **Error Messages**: Displays any issues encountered
- **Statistics**: Final summary of the dataset

### Step 6: Validating Output

Check your dataset:

```bash
# View first few entries
head -n 5 output/dataset.jsonl | jq '.'

# Check dataset statistics
dataseter analyze output/dataset.jsonl

# Validate format
dataseter validate output/dataset.jsonl
```

## Common Use Cases

### Use Case 1: Creating a Q&A Dataset

For training question-answering models:

```bash
# Extract from FAQ pages and documentation
dataseter create \
  --website https://docs.python.org/3/ \
  --website https://stackoverflow.com/questions/tagged/python \
  --chunk-size 256 \
  --format jsonl \
  -o python_qa_dataset.jsonl
```

### Use Case 2: Document Classification Dataset

For training document classifiers:

```bash
# Process documents with preserved metadata
dataseter create \
  --directory ./classified_docs \
  --preserve-metadata \
  --no-chunking \
  -o classification_dataset.jsonl
```

### Use Case 3: Language Model Training

For training language models:

```bash
# Large-scale text extraction
dataseter create \
  --directory ./books \
  --directory ./articles \
  --chunk-size 1024 \
  --remove-pii \
  --quality-threshold 0.8 \
  -o language_model_dataset.parquet
```

### Use Case 4: Multilingual Dataset

For multilingual models:

```bash
# Extract from multiple language sources
dataseter create \
  --website https://es.wikipedia.org \
  --website https://fr.wikipedia.org \
  --website https://de.wikipedia.org \
  --language-detection \
  -o multilingual_dataset.jsonl
```

## Tips for Beginners

### Start Small

Don't try to process thousands of documents on your first attempt:

1. Start with 5-10 documents
2. Verify the output quality
3. Adjust configuration as needed
4. Scale up gradually

### Quality Over Quantity

Better to have 1,000 high-quality examples than 10,000 poor ones:

```bash
# High quality settings
dataseter create \
  --pdf document.pdf \
  --quality-threshold 0.9 \
  --remove-duplicates \
  --remove-pii \
  -o high_quality.jsonl
```

### Use Appropriate Chunk Sizes

Different AI models require different chunk sizes:

- **Small models**: 128-256 tokens
- **BERT-like models**: 512 tokens
- **GPT-like models**: 1024-2048 tokens
- **Long-context models**: 4096+ tokens

### Handle Errors Gracefully

Some files might fail to process:

```bash
# Continue on errors
dataseter create \
  --directory ./mixed_documents \
  --continue-on-error \
  --log-errors error_log.txt \
  -o dataset.jsonl
```

### Organize Your Workflow

Create a project structure:

```bash
# Create project structure
mkdir my_ai_dataset
cd my_ai_dataset
mkdir -p sources/{pdfs,websites} output logs configs

# Save configuration
cat > configs/default.yaml << EOF
general:
  log_level: INFO
extraction:
  pdf:
    ocr_enabled: true
processing:
  chunking:
    chunk_size: 512
EOF

# Run with saved config
dataseter create \
  --config configs/default.yaml \
  --directory sources/pdfs \
  -o output/dataset_$(date +%Y%m%d).jsonl
```

### Monitor Resource Usage

Large datasets can consume significant resources:

```bash
# Limit parallel processing for low-memory systems
dataseter create \
  --directory ./large_dataset \
  --max-workers 2 \
  --batch-size 10 \
  -o dataset.jsonl
```

### Backup Your Work

Always keep copies of:
1. Original source files
2. Configuration files
3. Output datasets
4. Processing logs

### Learn from Examples

Study the example datasets provided:

```bash
# Run example datasets
dataseter examples list
dataseter examples run basic_pdf
dataseter examples run web_scraping
```

## Next Steps

Now that you understand the basics:

1. **Read the [Installation Guide](INSTALLATION.md)** for detailed setup instructions
2. **Explore the [CLI Reference](CLI_REFERENCE.md)** for all available commands
3. **Check the [Processing Guide](PROCESSING_GUIDE.md)** for advanced processing options
4. **Try the [Web GUI](WEB_GUI_TUTORIAL.md)** for a visual interface
5. **Learn about [API Usage](API_TUTORIAL.md)** for programmatic access

## Getting Help

If you encounter issues:

1. **Check the [FAQ](FAQ.md)** for common questions
2. **Read the [Troubleshooting Guide](TROUBLESHOOTING.md)**
3. **Search [GitHub Issues](https://github.com/mkupermann/dataseter/issues)**
4. **Join the community** on Discord or Slack
5. **Contact support** at dataseter@example.com

## Summary

You've learned:

- What Dataseter is and why it's useful
- Core concepts of data extraction and processing
- How to install and verify Dataseter
- Creating your first dataset
- Understanding the complete workflow
- Common use cases and best practices
- Tips for beginners

Remember: Start small, focus on quality, and gradually build your expertise. Good luck creating amazing datasets for your AI models!