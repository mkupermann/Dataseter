# Docker Commands for Dataseter

## Simple Build (Single Platform)

```bash
# Build for current platform only
docker build -t mkupermann/dataseter:latest .

# Or with version tag
docker build -t mkupermann/dataseter:1.0.0 -t mkupermann/dataseter:latest .
```

## Multi-Platform Build with Buildx

### Setup Buildx (First Time Only)

```bash
# Create a new builder instance
docker buildx create --name mybuilder --use

# Bootstrap the builder
docker buildx inspect --bootstrap
```

### Build and Push (Multi-Platform)

```bash
# Login to Docker Hub first
docker login -u mkupermann

# Build and push for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t mkupermann/dataseter:latest \
  -t mkupermann/dataseter:1.0.0 \
  --push \
  .
```

### Build Locally (Single Platform)

```bash
# Build only for local platform and load into Docker
docker buildx build \
  --platform linux/amd64 \
  -t mkupermann/dataseter:latest \
  --load \
  .
```

## Regular Docker Commands

### Push to Docker Hub

```bash
# Login first
docker login -u mkupermann

# Push the image
docker push mkupermann/dataseter:latest
docker push mkupermann/dataseter:1.0.0
```

### Run the Container

```bash
# Basic run
docker run -p 8080:8080 mkupermann/dataseter:latest

# Run with data persistence
docker run -d \
  --name dataseter \
  -p 8080:8080 \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  mkupermann/dataseter:latest

# Run with custom config
docker run -d \
  --name dataseter \
  -p 8080:8080 \
  -v $(pwd)/config/config.yaml:/app/config/config.yaml \
  -v $(pwd)/data:/data \
  mkupermann/dataseter:latest
```

## Docker Compose

```bash
# Build and start
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f dataseter

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Troubleshooting

### If buildx fails with "requires 1 argument"

The error occurs when the build context (.) is missing. Always include the dot at the end:

```bash
# WRONG
docker buildx build --platform linux/amd64

# CORRECT
docker buildx build --platform linux/amd64 .
#                                           ^ Don't forget this!
```

### Check Builder Status

```bash
# List builders
docker buildx ls

# Inspect current builder
docker buildx inspect

# Use default builder
docker buildx use default

# Use custom builder
docker buildx use mybuilder
```

### Remove Old Builders

```bash
# List all builders
docker buildx ls

# Remove a specific builder
docker buildx rm mybuilder

# Remove unused builders
docker buildx prune
```

## Quick Commands

### Build and Push in One Command

```bash
# Ensure you're logged in
docker login -u mkupermann

# Build and push
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t mkupermann/dataseter:latest \
  --push \
  .
```

### Test Locally Before Push

```bash
# Build locally
docker build -t mkupermann/dataseter:test .

# Run test
docker run --rm -p 8080:8080 mkupermann/dataseter:test

# If satisfied, tag and push
docker tag mkupermann/dataseter:test mkupermann/dataseter:latest
docker push mkupermann/dataseter:latest
```

## Verification

### Check Image Locally

```bash
# List local images
docker images | grep dataseter

# Inspect image
docker inspect mkupermann/dataseter:latest

# Check image size
docker images mkupermann/dataseter --format "table {{.Tag}}\t{{.Size}}"
```

### Check on Docker Hub

```bash
# Pull from Docker Hub to verify
docker pull mkupermann/dataseter:latest

# Or visit
# https://hub.docker.com/r/mkupermann/dataseter
```