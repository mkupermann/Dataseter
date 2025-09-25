# Docker Hub Deployment Instructions

## Prerequisites

1. **Docker Hub Account**: Ensure you're logged in to your mkupermann account
2. **Docker Installed**: Docker Desktop or Docker Engine
3. **Docker Hub Access Token** (recommended over password)

## Manual Build and Push

### 1. Login to Docker Hub
```bash
docker login -u mkupermann
# Enter your Docker Hub password or access token
```

### 2. Build the Image
```bash
# Build with version tag and latest
docker build -t mkupermann/dataseter:1.0.0 -t mkupermann/dataseter:latest .
```

### 3. Push to Docker Hub
```bash
# Push specific version
docker push mkupermann/dataseter:1.0.0

# Push latest tag
docker push mkupermann/dataseter:latest
```

## Automated Build Script

Use the provided script for easier deployment:

```bash
# Make script executable (first time only)
chmod +x docker-build.sh

# Run the build and push script
./docker-build.sh
```

The script will:
1. Build the image with proper tags
2. Ask if you want to push to Docker Hub
3. Optionally run the container locally for testing

## GitHub Actions (Automated CI/CD)

### Setup
1. Go to Docker Hub and generate an access token:
   - Account Settings → Security → New Access Token
   - Name: `GITHUB_ACTIONS`
   - Access permissions: Read, Write, Delete

2. Add to GitHub repository secrets:
   - Go to: https://github.com/mkupermann/dataseter/settings/secrets/actions
   - Add new secret:
     - Name: `DOCKER_PASSWORD`
     - Value: Your Docker Hub access token

3. Push to main branch or create a tag:
```bash
# Push to main branch (triggers build)
git push origin main

# Or create and push a version tag
git tag v1.0.0
git push origin v1.0.0
```

## Docker Compose Deployment

For local development with all services:

```bash
# Start all services
docker-compose up -d

# Or just Dataseter
docker-compose up dataseter

# View logs
docker-compose logs -f dataseter

# Stop services
docker-compose down
```

## Verify Deployment

### Check Docker Hub
Visit: https://hub.docker.com/r/mkupermann/dataseter

### Pull and Test
```bash
# Pull from Docker Hub
docker pull mkupermann/dataseter:latest

# Run container
docker run -p 8080:8080 mkupermann/dataseter:latest

# Test
curl http://localhost:8080/health
```

## Multi-Platform Build

For ARM64 (M1/M2 Macs) and AMD64:

```bash
# Setup buildx
docker buildx create --use

# Build and push multi-platform
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t mkupermann/dataseter:latest \
  --push .
```

## Container Management

### Run with Data Persistence
```bash
docker run -d \
  --name dataseter \
  -p 8080:8080 \
  -p 8000:8000 \
  -v dataseter-data:/data \
  --restart unless-stopped \
  mkupermann/dataseter:latest
```

### View Logs
```bash
docker logs dataseter
docker logs -f dataseter  # Follow logs
```

### Stop and Remove
```bash
docker stop dataseter
docker rm dataseter
```

### Update to Latest Version
```bash
docker pull mkupermann/dataseter:latest
docker stop dataseter
docker rm dataseter
# Run new container with same command as before
```

## Troubleshooting

### Build Issues
- Ensure all Python dependencies are in requirements.txt
- Check Dockerfile syntax
- Verify base image availability

### Push Issues
- Verify Docker Hub login: `docker login`
- Check image size (Docker Hub has limits)
- Ensure proper tagging

### Runtime Issues
- Check logs: `docker logs dataseter`
- Verify port availability
- Check volume permissions

## Security Notes

1. **Never commit secrets** to the repository
2. **Use access tokens** instead of passwords
3. **Run as non-root user** (already configured in Dockerfile)
4. **Keep base images updated** regularly

## Support

- GitHub Issues: https://github.com/mkupermann/dataseter/issues
- Docker Hub: https://hub.docker.com/r/mkupermann/dataseter