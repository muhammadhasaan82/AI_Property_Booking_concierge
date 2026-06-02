#!/bin/bash
set -e

echo "Deploying self-hosted Langfuse..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "Error: docker compose is not installed."
    exit 1
fi

# Define target directory
TARGET_DIR="${1:-/root/langfuse-selfhosted}"

if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR already exists. Pulling latest changes..."
    cd "$TARGET_DIR"
    git pull origin main
else
    echo "Cloning Langfuse repository to $TARGET_DIR..."
    git clone https://github.com/langfuse/langfuse.git "$TARGET_DIR"
    cd "$TARGET_DIR"
fi

# Setup environment
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "Please edit $TARGET_DIR/.env and replace all #CHANGEME values before proceeding."
    echo "Press Enter to continue once you have updated the .env file..."
    read -r
fi

# Deploy
echo "Starting Langfuse services..."
docker compose up -d

echo "Deployment complete! Access Langfuse at http://<server-ip>:3000"
echo "Remember to change the default admin credentials immediately."
