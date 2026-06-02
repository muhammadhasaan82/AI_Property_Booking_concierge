#!/bin/bash
set -e

echo "Checking Langfuse deployment status..."

TARGET_DIR="${1:-/root/langfuse-selfhosted}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist. Has Langfuse been deployed?"
    exit 1
fi

cd "$TARGET_DIR"

echo "Checking Docker Compose services..."
docker compose ps

echo ""
echo "Checking container logs for errors (last 20 lines)..."
docker compose logs --tail=20 web

echo ""
echo "To view full logs, run: cd $TARGET_DIR && docker compose logs -f"
echo "To check service health, run: curl -I http://localhost:3000"
