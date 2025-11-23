#!/bin/bash

# PETR4.SA Prediction System - Quick Start Script
# This script provides convenient commands to manage the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    print_color $BLUE "🚀 PETR4.SA Prediction System"
    print_color $BLUE "================================"
    echo
}

print_usage() {
    print_header
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  local       - Run locally with Poetry"
    echo "  docker      - Build and run with Docker"
    echo "  compose     - Run with Docker Compose"
    echo "  build       - Build Docker image only"
    echo "  test        - Run tests"
    echo "  clean       - Clean Docker resources"
    echo "  logs        - Show application logs"
    echo "  health      - Check API health"
    echo "  help        - Show this help message"
    echo
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Local development
run_local() {
    print_header
    print_color $YELLOW "🏃 Running locally with Poetry..."
    
    if ! command_exists poetry; then
        print_color $RED "❌ Poetry not found. Please install Poetry first:"
        echo "curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi
    
    print_color $BLUE "📦 Installing dependencies..."
    poetry install
    
    print_color $BLUE "🚀 Starting server..."
    poetry run python main.py
}

# Docker build and run
run_docker() {
    print_header
    print_color $YELLOW "🐳 Building and running with Docker..."
    
    if ! command_exists docker; then
        print_color $RED "❌ Docker not found. Please install Docker first."
        exit 1
    fi
    
    print_color $BLUE "🔨 Building Docker image..."
    docker build -t petr4-prediction .
    
    print_color $BLUE "🚀 Starting container..."
    docker run --rm -it -p 5000:5000 --name petr4-api petr4-prediction
}

# Docker Compose
run_compose() {
    print_header
    print_color $YELLOW "🐳 Running with Docker Compose..."
    
    if ! command_exists docker-compose; then
        print_color $RED "❌ Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi
    
    print_color $BLUE "🚀 Starting services..."
    docker-compose up --build
}

# Build only
build_image() {
    print_header
    print_color $YELLOW "🔨 Building Docker image..."
    
    docker build -t petr4-prediction .
    print_color $GREEN "✅ Image built successfully!"
    docker images | grep petr4-prediction
}

# Run tests
run_tests() {
    print_header
    print_color $YELLOW "🧪 Running tests..."
    
    if command_exists poetry; then
        poetry run pytest
    else
        python -m pytest
    fi
}

# Clean Docker resources
clean_docker() {
    print_header
    print_color $YELLOW "🧹 Cleaning Docker resources..."
    
    print_color $BLUE "Stopping containers..."
    docker-compose down 2>/dev/null || true
    
    print_color $BLUE "Removing image..."
    docker rmi petr4-prediction 2>/dev/null || true
    
    print_color $BLUE "Pruning unused resources..."
    docker system prune -f
    
    print_color $GREEN "✅ Cleanup completed!"
}

# Show logs
show_logs() {
    print_header
    print_color $YELLOW "📋 Showing application logs..."
    
    if docker ps | grep -q petr4-api; then
        docker logs -f petr4-api
    else
        print_color $RED "❌ Container not running. Start the application first."
        exit 1
    fi
}

# Health check
health_check() {
    print_header
    print_color $YELLOW "🏥 Checking API health..."
    
    if command_exists curl; then
        response=$(curl -s -w "%{http_code}" http://localhost:5000/api/petr4/health -o /tmp/health.json)
        
        if [ "$response" = "200" ]; then
            print_color $GREEN "✅ API is healthy!"
            cat /tmp/health.json | python -m json.tool
        else
            print_color $RED "❌ API health check failed (HTTP $response)"
            exit 1
        fi
    else
        print_color $RED "❌ curl not found. Please install curl to run health checks."
        exit 1
    fi
}

# Main script logic
case "$1" in
    local)
        run_local
        ;;
    docker)
        run_docker
        ;;
    compose)
        run_compose
        ;;
    build)
        build_image
        ;;
    test)
        run_tests
        ;;
    clean)
        clean_docker
        ;;
    logs)
        show_logs
        ;;
    health)
        health_check
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        print_color $RED "❌ Unknown command: $1"
        print_usage
        exit 1
        ;;
esac