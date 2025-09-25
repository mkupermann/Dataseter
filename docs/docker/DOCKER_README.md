# Dataseter - AI Training Dataset Creator

Advanced tool for creating high-quality datasets for AI/ML training from diverse sources.

## Quick Start

```bash
docker run -p 8080:8080 mkupermann/dataseter:latest
```

Access the web UI at http://localhost:8080

## Features

- **Multi-Source Extraction**: PDF, Web, Office docs, eBooks
- **Web Scraping**: Recursive crawling with JavaScript support
- **Text Processing**: Chunking, cleaning, PII removal
- **Quality Control**: Scoring, deduplication, language detection
- **Output Formats**: JSONL, CSV, Parquet, HuggingFace
- **International**: Supports German (n-tv.de tested) and 50+ languages

## Usage Examples

### Web Interface
```bash
docker run -d \
  --name dataseter \
  -p 8080:8080 \
  -p 8000:8000 \
  mkupermann/dataseter:latest
```

### CLI Mode
```bash
docker run --rm \
  -v $(pwd)/data:/data \
  mkupermann/dataseter:latest \
  dataseter create \
    --website https://example.com \
    --output /data/output/dataset.jsonl
```

### With Configuration
```bash
docker run -d \
  --name dataseter \
  -p 8080:8080 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/data \
  -e DATASETER_CONFIG=/app/config/config.yaml \
  mkupermann/dataseter:latest
```

## Docker Compose

```yaml
version: '3.8'

services:
  dataseter:
    image: mkupermann/dataseter:latest
    ports:
      - "8080:8080"
      - "8000:8000"
    volumes:
      - ./data:/data
      - ./config:/app/config
    environment:
      - DATASETER_CONFIG=/app/config/config.yaml
```

## Environment Variables

- `DATASETER_CONFIG`: Path to configuration file
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)
- `DATASETER_CACHE_DIR`: Cache directory path

## Volumes

- `/data/input`: Input files
- `/data/output`: Output datasets
- `/data/cache`: Cache directory
- `/app/config`: Configuration files

## Ports

- `8080`: Web UI
- `8000`: REST API

## Tags

- `latest`: Latest stable version
- `1.0.0`: Specific version
- `main`: Development branch

## API Endpoints

- `GET /`: API status
- `POST /upload`: Upload file
- `POST /extract`: Start extraction
- `GET /job/{id}`: Get job status
- `GET /download/{id}`: Download results

## Supported Sources

- **Web**: Any website (tested with Wikipedia, n-tv.de, GitHub docs)
- **PDF**: With OCR support
- **Office**: Word, Excel, PowerPoint
- **eBooks**: EPUB, MOBI, AZW
- **Text**: TXT, Markdown, RTF

## License

MIT

## Source

https://github.com/mkupermann/dataseter