#!/bin/bash

# Docker build and push script for Dataseter
# Pushes to mkupermann Docker Hub account

set -e

# Configuration
DOCKER_USER="mkupermann"
IMAGE_NAME="dataseter"
VERSION="1.0.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Dataseter Docker image...${NC}"

# Build the image with multiple tags
echo -e "${YELLOW}Building ${DOCKER_USER}/${IMAGE_NAME}:${VERSION}${NC}"
docker build -t ${DOCKER_USER}/${IMAGE_NAME}:${VERSION} \
             -t ${DOCKER_USER}/${IMAGE_NAME}:latest \
             .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Ask if user wants to push to Docker Hub
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Logging in to Docker Hub...${NC}"
    docker login -u ${DOCKER_USER}

    echo -e "${YELLOW}Pushing ${DOCKER_USER}/${IMAGE_NAME}:${VERSION}...${NC}"
    docker push ${DOCKER_USER}/${IMAGE_NAME}:${VERSION}

    echo -e "${YELLOW}Pushing ${DOCKER_USER}/${IMAGE_NAME}:latest...${NC}"
    docker push ${DOCKER_USER}/${IMAGE_NAME}:latest

    echo -e "${GREEN}Push complete!${NC}"
    echo -e "${GREEN}Image available at: https://hub.docker.com/r/${DOCKER_USER}/${IMAGE_NAME}${NC}"
else
    echo -e "${YELLOW}Skipping push to Docker Hub${NC}"
fi

# Optional: Run the container locally for testing
read -p "Run container locally for testing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Running container...${NC}"
    docker run -d \
        --name dataseter-test \
        -p 8080:8080 \
        -p 8000:8000 \
        -v $(pwd)/data:/data \
        ${DOCKER_USER}/${IMAGE_NAME}:latest

    echo -e "${GREEN}Container running!${NC}"
    echo "Web UI: http://localhost:8080"
    echo "API: http://localhost:8000"
    echo ""
    echo "To stop: docker stop dataseter-test"
    echo "To remove: docker rm dataseter-test"
    echo "To view logs: docker logs dataseter-test"
fi