#!/bin/bash
# Script to deploy the application using Docker Compose

echo "Preparing for deployment..."

# Create necessary directories
mkdir -p data/vector_store

echo "Building and deploying with Docker Compose..."
docker-compose down
docker-compose build
docker-compose up -d

echo "Application deployed successfully!"
echo "Check logs with: docker-compose logs -f app"
