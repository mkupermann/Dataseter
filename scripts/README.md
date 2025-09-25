# Dataseter Scripts

This directory contains utility scripts for testing and deployment.

## Scripts

### Testing Scripts
- `run_tests.py` - Master test runner
- `test_live.py` - Live website extraction tests
- `test_ntv.py` - n-tv.de specific tests
- `demo.py` - Interactive demonstration

### Docker Scripts
- `docker-build.sh` - Build and push Docker image
- `docker-build-multiplatform.sh` - Multi-platform Docker build

## Usage

### Run Tests
```bash
python scripts/run_tests.py --all
python scripts/test_live.py
python scripts/test_ntv.py
```

### Build Docker
```bash
./scripts/docker-build.sh
./scripts/docker-build-multiplatform.sh
```

### Run Demo
```bash
python scripts/demo.py
```