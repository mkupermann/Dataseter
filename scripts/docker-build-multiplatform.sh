#!/bin/bash

# Multi-platform Docker build and push script for Dataseter
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

echo -e "${GREEN}Building Dataseter Docker image (Multi-platform)...${NC}"

# Check if buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo -e "${RED}Docker buildx is not available. Please update Docker.${NC}"
    exit 1
fi

# Create and use a new builder instance
echo -e "${YELLOW}Setting up buildx builder...${NC}"
docker buildx create --name dataseter-builder --use 2>/dev/null || docker buildx use dataseter-builder

# Inspect the builder
docker buildx inspect --bootstrap

# Ask if user wants to push to Docker Hub
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Logging in to Docker Hub...${NC}"
    docker login -u ${DOCKER_USER}

    echo -e "${YELLOW}Building and pushing multi-platform image...${NC}"
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -t ${DOCKER_USER}/${IMAGE_NAME}:${VERSION} \
        -t ${DOCKER_USER}/${IMAGE_NAME}:latest \
        --push \
        .

    echo -e "${GREEN}Push complete!${NC}"
    echo -e "${GREEN}Image available at: https://hub.docker.com/r/${DOCKER_USER}/${IMAGE_NAME}${NC}"
else
    echo -e "${YELLOW}Building locally without push...${NC}"
    docker buildx build \
        --platform linux/amd64 \
        -t ${DOCKER_USER}/${IMAGE_NAME}:${VERSION} \
        -t ${DOCKER_USER}/${IMAGE_NAME}:latest \
        --load \
        .

    echo -e "${GREEN}Local build complete!${NC}"
fi

# Clean up builder (optional)
read -p "Remove buildx builder? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker buildx rm dataseter-builder
    echo -e "${GREEN}Builder removed${NC}"
fi