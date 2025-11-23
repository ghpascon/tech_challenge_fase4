# Makefile for PETR4.SA Prediction System
# Provides convenient commands for development and deployment

.PHONY: help install run docker compose build test clean logs health dev

# Default target
.DEFAULT_GOAL := help

# Variables
IMAGE_NAME := petr4-prediction
CONTAINER_NAME := petr4-api
PORT := 5000

# Colors
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m

# Help target
help: ## Show this help message
	@echo "$(BLUE)🚀 PETR4.SA Prediction System$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

install: ## Install dependencies with Poetry
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	poetry install

run: ## Run the application locally
	@echo "$(BLUE)🚀 Starting PETR4 Prediction API...$(NC)"
	poetry run python main.py

dev: ## Run in development mode with auto-reload
	@echo "$(BLUE)🔧 Starting in development mode...$(NC)"
	poetry run uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

docker: ## Build and run with Docker
	@echo "$(BLUE)🐳 Building and running with Docker...$(NC)"
	docker build -t $(IMAGE_NAME) .
	docker run --rm -it -p $(PORT):$(PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

compose: ## Run with Docker Compose
	@echo "$(BLUE)🐳 Starting with Docker Compose...$(NC)"
	docker-compose up --build

compose-d: ## Run with Docker Compose in detached mode
	@echo "$(BLUE)🐳 Starting with Docker Compose (detached)...$(NC)"
	docker-compose up --build -d

build: ## Build Docker image
	@echo "$(BLUE)🔨 Building Docker image...$(NC)"
	docker build -t $(IMAGE_NAME) .
	@echo "$(GREEN)✅ Image built successfully!$(NC)"

test: ## Run tests
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	poetry run pytest -v

lint: ## Run code linting
	@echo "$(BLUE)🔍 Running linting...$(NC)"
	poetry run flake8 app/
	poetry run black --check app/
	poetry run isort --check-only app/

format: ## Format code
	@echo "$(BLUE)✨ Formatting code...$(NC)"
	poetry run black app/
	poetry run isort app/

clean: ## Clean Docker resources
	@echo "$(BLUE)🧹 Cleaning Docker resources...$(NC)"
	-docker-compose down
	-docker rmi $(IMAGE_NAME)
	-docker system prune -f
	@echo "$(GREEN)✅ Cleanup completed!$(NC)"

logs: ## Show application logs
	@echo "$(BLUE)📋 Showing logs...$(NC)"
	docker logs -f $(CONTAINER_NAME)

logs-compose: ## Show Docker Compose logs
	@echo "$(BLUE)📋 Showing Docker Compose logs...$(NC)"
	docker-compose logs -f

health: ## Check API health
	@echo "$(BLUE)🏥 Checking API health...$(NC)"
	@curl -s http://localhost:$(PORT)/api/petr4/health | python -m json.tool || echo "$(RED)❌ API not responding$(NC)"

stop: ## Stop Docker containers
	@echo "$(BLUE)🛑 Stopping containers...$(NC)"
	-docker stop $(CONTAINER_NAME)
	-docker-compose down

restart: ## Restart the application
	@echo "$(BLUE)🔄 Restarting application...$(NC)"
	$(MAKE) stop
	$(MAKE) compose-d

ps: ## Show running containers
	@echo "$(BLUE)📊 Container status:$(NC)"
	@docker ps | grep petr4 || echo "$(YELLOW)No containers running$(NC)"

shell: ## Open shell in running container
	@echo "$(BLUE)🐚 Opening shell in container...$(NC)"
	docker exec -it $(CONTAINER_NAME) /bin/bash

predict: ## Test prediction endpoint
	@echo "$(BLUE)🎯 Testing prediction endpoint...$(NC)"
	@curl -X POST http://localhost:$(PORT)/api/petr4/predict \
		-H "Content-Type: application/json" \
		-d '{"days_ahead": 1}' | python -m json.tool

setup: ## Complete setup (install + build)
	@echo "$(BLUE)⚙️ Complete setup...$(NC)"
	$(MAKE) install
	$(MAKE) build
	@echo "$(GREEN)✅ Setup completed!$(NC)"

deploy: ## Deploy application (build + compose)
	@echo "$(BLUE)🚀 Deploying application...$(NC)"
	$(MAKE) build
	$(MAKE) compose-d
	@echo "$(GREEN)✅ Application deployed!$(NC)"

info: ## Show system information
	@echo "$(BLUE)ℹ️ System Information:$(NC)"
	@echo "Image: $(IMAGE_NAME)"
	@echo "Container: $(CONTAINER_NAME)"
	@echo "Port: $(PORT)"
	@echo "API URL: http://localhost:$(PORT)"
	@echo "Docs URL: http://localhost:$(PORT)/docs"