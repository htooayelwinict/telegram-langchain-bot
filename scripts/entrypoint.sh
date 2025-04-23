#!/bin/bash

# Docker entrypoint script
# This script will:
# 1. Run the webhook setup script if in webhook mode with ngrok
# 2. Start the application

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to log warnings
warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Function to log errors
error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Load environment variables
if [ -f .env ]; then
    log "Loading environment variables from .env file"
    source .env
fi

# Ensure data directories exist
log "Ensuring data directories exist"
mkdir -p /app/data/vector_store

# Check if we're in webhook mode and using ngrok
if [ -n "$WEBHOOK_URL" ] && [[ "$WEBHOOK_URL" == *"ngrok"* ]]; then
    log "Detected ngrok webhook URL, setting up webhook"
    
    # Make the webhook setup script executable
    chmod +x /app/scripts/setup_webhook.sh
    
    # Run the webhook setup script
    /app/scripts/setup_webhook.sh
    
    # Check if the webhook setup was successful
    if [ $? -ne 0 ]; then
        warn "Webhook setup failed, continuing anyway"
    fi
elif [ -n "$WEBHOOK_URL" ]; then
    log "Using custom webhook URL: $WEBHOOK_URL"
else
    log "Running in polling mode (no webhook URL specified)"
fi

# Start the application
log "Starting the application"
python -m app.main
