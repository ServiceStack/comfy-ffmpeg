#!/bin/bash

# Initialize verbose flag
VERBOSE=false

# Process command line arguments
while getopts "v" opt; do
    case $opt in
        v) VERBOSE=true ;;
        *) echo "Usage: $0 [-v]" >&2
           exit 1 ;;
    esac
done

# Helper function for verbose logging
log() {
    if [ "$VERBOSE" = true ]; then
        echo "$1"
    fi
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Docker is not installed. Please install Docker before running this script."
        echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker compose &> /dev/null; then
        echo "Docker Compose is not installed or not in PATH."
        echo "Recent Docker Desktop versions include Compose. If you're using Docker Desktop, please make sure it's up to date."
        echo "Otherwise, visit https://docs.docker.com/compose/install/ for installation instructions."
        exit 1
    fi

    log "Prerequisites check passed. Docker, Docker Compose, and jq are installed."
}

# Reusable function to write to .env
write_env() {
    echo "$1=$2" >> .env
}

setup() {

    # Check for REQUIRE_CUSTOM_NODES Environment Variable
    if [ -z "$REQUIRE_CUSTOM_NODES" ]; then
        log "REQUIRE_CUSTOM_NODES environment variable is not set. Using default value."
        write_env "REQUIRE_CUSTOM_NODES" "ltdrdata/ComfyUI-Manager,MoonHugo/ComfyUI-FFmpeg,ServiceStack/comfy-asset-downloader"
    fi

    # Create comfy-network network if it doesn't exist
    docker network create comfy-network &> /dev/null || true

    # Ensure latest version of comfy-ffmpeg docker image
    docker compose pull

    # Start comfy-ffmpeg container
    docker compose up -d
}

# Run the prerequisites check function
check_prerequisites

# Run comfy-ffmpeg server with docker compose
setup
