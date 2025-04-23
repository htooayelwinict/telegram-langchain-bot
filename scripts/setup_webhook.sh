#!/bin/bash

# Script to set up Telegram webhook with ngrok URL
# This script will:
# 1. Wait for ngrok to be ready
# 2. Get the ngrok public URL
# 3. Generate a webhook secret if not already present
# 4. Update the .env file with the ngrok URL and webhook secret
# 5. Set up the Telegram webhook

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

# Function to generate a random string for webhook secret
generate_secret() {
    local length=$1
    if [ -z "$length" ]; then
        length=32
    fi
    
    # Generate a random string using /dev/urandom
    local random_string=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w $length | head -n 1)
    echo $random_string
}

# Function to wait for ngrok to be ready
wait_for_ngrok() {
    log "Waiting for ngrok to be ready..."
    
    local max_attempts=30
    local attempt=0
    local ngrok_api="http://ngrok:4040/api/tunnels"
    
    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt+1))
        
        # Try to get tunnels from ngrok API
        local response=$(curl -s $ngrok_api)
        
        # Check if we got a valid response with tunnels
        if [[ $response == *"tunnels"* && $response == *"public_url"* ]]; then
            log "ngrok is ready!"
            return 0
        fi
        
        log "Waiting for ngrok... Attempt $attempt/$max_attempts"
        sleep 2
    done
    
    error "ngrok failed to start after $max_attempts attempts"
    return 1
}

# Function to get the ngrok public URL
get_ngrok_url() {
    log "Getting ngrok public URL..."
    
    local ngrok_api="http://ngrok:4040/api/tunnels"
    local response=$(curl -s $ngrok_api)
    
    # Extract the public URL from the response
    local public_url=$(echo $response | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*' | head -n 1)
    
    if [ -z "$public_url" ]; then
        error "Failed to get ngrok public URL"
        return 1
    fi
    
    log "ngrok public URL: $public_url"
    echo $public_url
}

# Function to update the .env file
update_env_file() {
    local env_file="/app/.env"
    local ngrok_url=$1
    local webhook_secret=$2
    
    log "Updating .env file with ngrok URL and webhook secret..."
    
    # Check if WEBHOOK_URL is already in the .env file
    if grep -q "WEBHOOK_URL=" $env_file; then
        # Update existing WEBHOOK_URL
        sed -i "s|WEBHOOK_URL=.*|WEBHOOK_URL=${ngrok_url}/webhook|g" $env_file
    else
        # Add WEBHOOK_URL
        echo "WEBHOOK_URL=${ngrok_url}/webhook" >> $env_file
    fi
    
    # Check if WEBHOOK_SECRET is already in the .env file
    if grep -q "WEBHOOK_SECRET=" $env_file; then
        # Check if WEBHOOK_SECRET is empty
        if grep -q "WEBHOOK_SECRET=$" $env_file || grep -q "WEBHOOK_SECRET=\"\"" $env_file || grep -q "WEBHOOK_SECRET=''" $env_file; then
            # Update empty WEBHOOK_SECRET
            sed -i "s|WEBHOOK_SECRET=.*|WEBHOOK_SECRET=${webhook_secret}|g" $env_file
        else
            warn "WEBHOOK_SECRET already exists in .env file and is not empty. Not updating."
        fi
    else
        # Add WEBHOOK_SECRET
        echo "WEBHOOK_SECRET=${webhook_secret}" >> $env_file
    fi
    
    log ".env file updated successfully"
}

# Function to set up the Telegram webhook
setup_telegram_webhook() {
    local webhook_url=$1
    local webhook_secret=$2
    local telegram_token=$3
    
    log "Setting up Telegram webhook..."
    
    # Construct the webhook URL with secret token
    local full_webhook_url="${webhook_url}/webhook"
    
    # Set up the webhook
    local response=$(curl -s -X POST "https://api.telegram.org/bot${telegram_token}/setWebhook" \
        -H "Content-Type: application/json" \
        -d "{\"url\": \"${full_webhook_url}\", \"secret_token\": \"${webhook_secret}\"}")
    
    # Check if the webhook was set up successfully
    if [[ $response == *"\"ok\":true"* ]]; then
        log "Telegram webhook set up successfully!"
    else
        error "Failed to set up Telegram webhook: $response"
        return 1
    fi
}

# Main function
main() {
    log "Starting webhook setup..."
    
    # Load environment variables
    source /app/.env
    
    # Check if TELEGRAM_TOKEN is set
    if [ -z "$TELEGRAM_TOKEN" ]; then
        error "TELEGRAM_TOKEN is not set in .env file"
        return 1
    fi
    
    # Wait for ngrok to be ready
    wait_for_ngrok
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Get the ngrok public URL
    local ngrok_url=$(get_ngrok_url)
    if [ -z "$ngrok_url" ]; then
        return 1
    fi
    
    # Generate a webhook secret if not already present
    local webhook_secret=$WEBHOOK_SECRET
    if [ -z "$webhook_secret" ]; then
        webhook_secret=$(generate_secret 32)
        log "Generated webhook secret: $webhook_secret"
    else
        log "Using existing webhook secret"
    fi
    
    # Update the .env file
    update_env_file "$ngrok_url" "$webhook_secret"
    
    # Set up the Telegram webhook
    setup_telegram_webhook "$ngrok_url" "$webhook_secret" "$TELEGRAM_TOKEN"
    
    log "Webhook setup completed!"
}

# Run the main function
main
