# Dataseter Installation & Setup Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Dependency Installation](#dependency-installation)
4. [Platform-Specific Setup](#platform-specific-setup)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)
8. [Upgrading](#upgrading)
9. [Uninstallation](#uninstallation)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 10GB free space
- **Processor**: Dual-core 2GHz or better
- **Internet**: Required for web scraping and package installation

### Recommended Requirements

- **Operating System**: Ubuntu 20.04+, macOS 11+, or Windows 11
- **Python**: 3.10 or 3.11
- **RAM**: 16GB or more
- **Disk Space**: 50GB free space
- **Processor**: Quad-core 3GHz or better
- **GPU**: Optional, for accelerated OCR processing

### Python Version Check

Before installation, verify your Python version:

```bash
# Check Python version
python --version
# or
python3 --version

# Should output: Python 3.8.x or higher
```

If Python is not installed or outdated:
- **Windows**: Download from [python.org](https://python.org)
- **macOS**: Use Homebrew: `brew install python@3.11`
- **Linux**: Use package manager: `sudo apt install python3.11`

## Installation Methods

### Method 1: pip Installation (Simplest)

#### Basic Installation

```bash
# Install latest stable version
pip install dataseter

# Or with Python 3 specifically
python3 -m pip install dataseter
```

#### Full Installation (All Features)

```bash
# Install with all optional dependencies
pip install dataseter[all]

# Or install specific extras
pip install dataseter[ocr]        # OCR support
pip install dataseter[web]        # Advanced web scraping
pip install dataseter[ml]         # Machine learning features
pip install dataseter[dev]        # Development tools
```

#### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv dataseter_env

# Activate virtual environment
# On Linux/macOS:
source dataseter_env/bin/activate
# On Windows:
dataseter_env\Scripts\activate

# Install Dataseter
pip install dataseter[all]
```

### Method 2: Docker Installation (No Python Required)

#### Pull Pre-built Image

```bash
# Pull latest image
docker pull mkupermann/dataseter:latest

# Pull specific version
docker pull mkupermann/dataseter:1.0.0

# Verify image
docker images | grep dataseter
```

#### Run with Docker

```bash
# Basic run
docker run -it mkupermann/dataseter:latest

# With volume mounting for data persistence
docker run -it \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  mkupermann/dataseter:latest

# With all capabilities
docker run -it \
  --name dataseter \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  -v $(pwd)/config:/config \
  -p 8080:8080 \
  mkupermann/dataseter:latest
```

#### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  dataseter:
    image: mkupermann/dataseter:latest
    container_name: dataseter
    volumes:
      - ./data:/data
      - ./output:/output
      - ./config:/config
    ports:
      - "8080:8080"
    environment:
      - DATASETER_LOG_LEVEL=INFO
      - DATASETER_WORKERS=4
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

### Method 3: From Source (For Developers)

#### Clone and Install

```bash
# Clone repository
git clone https://github.com/mkupermann/dataseter.git
cd dataseter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

#### Building from Source

```bash
# Build package
python setup.py build

# Create distribution
python setup.py sdist bdist_wheel

# Install locally
pip install dist/dataseter-*.whl
```

### Method 4: Conda Installation

```bash
# Create conda environment
conda create -n dataseter python=3.10

# Activate environment
conda activate dataseter

# Install dependencies
conda install -c conda-forge \
  pandas numpy scipy \
  beautifulsoup4 requests \
  pdfplumber pytesseract

# Install Dataseter
pip install dataseter
```

## Dependency Installation

### Core Dependencies

Automatically installed with Dataseter:

```bash
# View installed dependencies
pip show dataseter
pip list | grep -E "(pandas|numpy|requests|beautifulsoup4)"
```

### OCR Dependencies

For OCR support (extracting text from images/scanned PDFs):

#### Ubuntu/Debian

```bash
# Install Tesseract OCR
sudo apt update
sudo apt install -y \
  tesseract-ocr \
  tesseract-ocr-eng \
  tesseract-ocr-deu \
  tesseract-ocr-fra \
  tesseract-ocr-spa \
  libtesseract-dev \
  poppler-utils

# Verify installation
tesseract --version
```

#### macOS

```bash
# Using Homebrew
brew install tesseract
brew install poppler

# Install language packs
brew install tesseract-lang

# Verify
tesseract --version
```

#### Windows

1. Download Tesseract installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer, note installation path
3. Add to PATH environment variable
4. Install poppler from [this site](http://blog.alivate.com.au/poppler-windows/)

```powershell
# Verify in PowerShell
tesseract --version
```

### Web Scraping Dependencies

For JavaScript-rendered pages:

#### Chromium/Chrome Driver

```bash
# Ubuntu/Debian
sudo apt install chromium-browser chromium-chromedriver

# macOS
brew install --cask chromium
brew install chromedriver

# Verify
chromedriver --version
```

#### Playwright (Alternative to Selenium)

```bash
# Install Playwright
pip install playwright

# Install browsers
playwright install

# Install with dependencies
playwright install --with-deps
```

### Language Processing Dependencies

```bash
# Install spaCy
pip install spacy

# Download language models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm

# Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Platform-Specific Setup

### Linux Setup

#### Ubuntu/Debian

```bash
# System dependencies
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-dev \
  build-essential \
  libxml2-dev \
  libxslt1-dev \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  zlib1g-dev \
  poppler-utils \
  tesseract-ocr \
  git \
  curl \
  wget

# Install Dataseter
pip install dataseter[all]
```

#### Red Hat/CentOS/Fedora

```bash
# System dependencies
sudo dnf install -y \
  python3-pip \
  python3-devel \
  gcc \
  gcc-c++ \
  make \
  libxml2-devel \
  libxslt-devel \
  libffi-devel \
  openssl-devel \
  libjpeg-turbo-devel \
  zlib-devel \
  poppler-utils \
  tesseract \
  git

# Install Dataseter
pip install dataseter[all]
```

### macOS Setup

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install \
  python@3.11 \
  tesseract \
  poppler \
  libxml2 \
  libxslt \
  jpeg \
  freetype \
  git

# Install Dataseter
pip3 install dataseter[all]
```

### Windows Setup

#### Using WSL2 (Recommended)

```powershell
# Install WSL2
wsl --install

# Update WSL
wsl --update

# Install Ubuntu
wsl --install -d Ubuntu-22.04

# Enter WSL
wsl

# Follow Linux installation steps
```

#### Native Windows

```powershell
# Install Chocolatey (Package Manager)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install dependencies
choco install -y python3 git tesseract

# Install Dataseter
pip install dataseter[all]
```

## Configuration

### Configuration File Structure

Create `config.yaml`:

```yaml
# General settings
general:
  log_level: INFO          # DEBUG, INFO, WARNING, ERROR
  max_workers: 4           # Parallel processing threads
  temp_directory: /tmp/dataseter
  cache_directory: ~/.dataseter/cache

# Extraction settings
extraction:
  pdf:
    ocr_enabled: true      # Enable OCR for scanned PDFs
    ocr_language: eng      # Tesseract language code
    extract_images: false  # Extract embedded images
    extract_tables: true   # Extract table data
    max_pages: null        # Limit pages (null = all)

  web:
    user_agent: "Dataseter/1.0"
    timeout: 30            # Request timeout in seconds
    rate_limit: 1.0        # Requests per second
    respect_robots: true   # Respect robots.txt
    javascript_rendering: false
    max_depth: 3          # Maximum crawl depth
    max_pages: 1000       # Maximum pages to crawl

  office:
    preserve_formatting: false
    extract_comments: true
    extract_headers_footers: true

  ebook:
    extract_metadata: true
    preserve_chapters: true

# Processing settings
processing:
  chunking:
    method: sliding_window  # sliding_window, sentence, paragraph
    chunk_size: 512        # Size in tokens
    chunk_overlap: 50      # Overlap between chunks

  cleaning:
    lowercase: false
    remove_html: true
    remove_urls: false
    normalize_whitespace: true
    remove_special_chars: false

  quality:
    min_length: 100        # Minimum text length
    max_length: 10000      # Maximum text length
    min_word_count: 20     # Minimum words
    language: en           # Expected language

  privacy:
    remove_emails: true
    remove_phone_numbers: true
    remove_credit_cards: true
    remove_ssn: true
    remove_ip_addresses: true
    custom_patterns: []    # Custom regex patterns

# Output settings
output:
  format: jsonl            # jsonl, parquet, csv, huggingface
  compression: gzip        # none, gzip, bz2, xz
  include_metadata: true
  include_statistics: true
  batch_size: 1000         # Records per file
```

### Environment Variables

Set environment variables for configuration:

```bash
# Set configuration file path
export DATASETER_CONFIG=/path/to/config.yaml

# Set log level
export DATASETER_LOG_LEVEL=DEBUG

# Set working directory
export DATASETER_WORK_DIR=/data/dataseter

# Set cache directory
export DATASETER_CACHE_DIR=/tmp/dataseter_cache

# Set maximum workers
export DATASETER_MAX_WORKERS=8
```

### Configuration Priority

Configuration is loaded in this order (later overrides earlier):
1. Default configuration
2. Configuration file (`config.yaml`)
3. Environment variables
4. Command-line arguments

## Verification

### Basic Verification

```bash
# Check installation
dataseter --version

# View help
dataseter --help

# List available commands
dataseter list-commands

# Run diagnostic
dataseter diagnose
```

### Component Testing

```bash
# Test PDF extraction
echo "Test content" > test.txt
dataseter test pdf test.txt

# Test web extraction
dataseter test web https://example.com

# Test OCR
dataseter test ocr sample.png

# Test all components
dataseter test all
```

### Creating Test Dataset

```bash
# Create test directory
mkdir test_dataseter
cd test_dataseter

# Create sample files
echo "This is a test document" > test.txt
echo "<html><body>Test HTML</body></html>" > test.html

# Run extraction
dataseter create \
  --text test.txt \
  --text test.html \
  -o test_output.jsonl \
  --verbose

# Verify output
cat test_output.jsonl | jq '.'
```

## Troubleshooting

### Common Installation Issues

#### Issue: pip command not found

```bash
# Solution: Install pip
python -m ensurepip --upgrade
# or
curl https://bootstrap.pypa.io/get-pip.py | python
```

#### Issue: Permission denied during installation

```bash
# Solution: Use user installation
pip install --user dataseter

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install dataseter
```

#### Issue: Building wheel failed

```bash
# Solution: Install build dependencies
# Ubuntu/Debian:
sudo apt install python3-dev build-essential

# macOS:
xcode-select --install

# Then retry installation
pip install --no-cache-dir dataseter
```

#### Issue: OCR not working

```bash
# Verify Tesseract installation
tesseract --version

# Check Tesseract data path
echo $TESSDATA_PREFIX

# Set if missing
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
```

#### Issue: Out of memory

```bash
# Reduce parallel workers
export DATASETER_MAX_WORKERS=2

# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Diagnostic Commands

```bash
# System information
dataseter diagnose --system

# Dependency check
dataseter diagnose --dependencies

# Configuration check
dataseter diagnose --config

# Full diagnostic
dataseter diagnose --full > diagnostic.log
```

## Upgrading

### Upgrading pip Installation

```bash
# Upgrade to latest version
pip install --upgrade dataseter

# Upgrade to specific version
pip install dataseter==1.2.0

# Upgrade with all dependencies
pip install --upgrade dataseter[all]
```

### Upgrading Docker Installation

```bash
# Pull latest image
docker pull mkupermann/dataseter:latest

# Stop old container
docker stop dataseter

# Remove old container
docker rm dataseter

# Run new version
docker run -d --name dataseter mkupermann/dataseter:latest
```

### Upgrading from Source

```bash
cd dataseter
git pull origin main
pip install --upgrade -e .[dev]
```

### Migration Guides

#### Migrating from 0.x to 1.x

```bash
# Backup old configuration
cp ~/.dataseter/config.yaml ~/.dataseter/config.yaml.backup

# Update configuration format
dataseter migrate-config ~/.dataseter/config.yaml

# Update scripts
dataseter migrate-scripts ./scripts/
```

## Uninstallation

### Removing pip Installation

```bash
# Uninstall package
pip uninstall dataseter

# Remove configuration
rm -rf ~/.dataseter

# Remove cache
rm -rf ~/.cache/dataseter
```

### Removing Docker Installation

```bash
# Stop and remove container
docker stop dataseter
docker rm dataseter

# Remove image
docker rmi mkupermann/dataseter:latest

# Remove volumes
docker volume rm dataseter_data
```

### Complete Cleanup

```bash
# Remove all Dataseter files
pip uninstall dataseter
rm -rf ~/.dataseter
rm -rf ~/.cache/dataseter
rm -rf /tmp/dataseter*

# Remove virtual environment
deactivate
rm -rf dataseter_env

# Remove Docker components
docker system prune -a --filter "label=app=dataseter"
```

## Next Steps

After successful installation:

1. Read the [Beginner's Guide](BEGINNER_GUIDE.md) to get started
2. Explore [CLI Commands](CLI_REFERENCE.md) for command-line usage
3. Try the [Quick Start Examples](../examples/README.md)
4. Configure for your needs using [Configuration Guide](CONFIGURATION.md)
5. Set up the [Web GUI](WEB_GUI_TUTORIAL.md) for visual interface

## Support

If you encounter installation issues:

1. Check the [FAQ](FAQ.md) for common problems
2. Search [GitHub Issues](https://github.com/mkupermann/dataseter/issues)
3. Run diagnostics: `dataseter diagnose --full`
4. Create a new issue with diagnostic output
5. Contact support with your diagnostic.log file