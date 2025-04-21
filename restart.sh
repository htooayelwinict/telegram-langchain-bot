#!/bin/bash
# Script to safely restart the Docker containers without losing data

echo "Stopping containers without removing volumes..."
docker-compose down

echo "Starting containers..."
docker-compose up -d

echo "Waiting for services to be ready..."
sleep 5

echo "Checking logs..."
docker-compose logs --tail=20 app

echo "Containers restarted successfully!"
echo "To view logs, run: docker-compose logs -f app"
