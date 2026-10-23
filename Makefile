PYTHON := .venv/bin/python
PIP := .venv/bin/pip
COMPOSE := docker compose
COMPOSE_FILES ?= -f compose.yaml

.PHONY: install run stop purge

install:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

run:
	@mkdir -p .run
	@if [ -f .run/api.pid ] && kill -0 $$(cat .run/api.pid) 2>/dev/null; then \
		echo "API уже запущен"; \
	else \
		$(PYTHON) -m uvicorn api:app --host 127.0.0.1 --port 8000 > .run/api.log 2>&1 & echo $$! > .run/api.pid; \
		echo "API запускается"; \
	fi
	@if [ -f .run/camera.pid ] && kill -0 $$(cat .run/camera.pid) 2>/dev/null; then \
		echo "Камера уже запущена"; \
	else \
		$(PYTHON) camera.py > .run/camera.log 2>&1 & echo $$! > .run/camera.pid; \
		echo "Камера запускается"; \
	fi

stop:
	@for name in camera api; do \
		if [ -f .run/$$name.pid ]; then \
			kill $$(cat .run/$$name.pid) 2>/dev/null || true; \
			rm -f .run/$$name.pid; \
		fi; \
	done
	@echo "Сервисы остановлены"

purge: stop
	@rm -rf .run
	@echo "Временные файлы удалены"

.PHONY: docker-install docker-run docker-stop docker-purge

docker-install:
	$(COMPOSE) $(COMPOSE_FILES) build

docker-run:
	$(COMPOSE) $(COMPOSE_FILES) up -d

docker-stop:
	$(COMPOSE) $(COMPOSE_FILES) down

docker-purge:
	$(COMPOSE) $(COMPOSE_FILES) down --rmi local --remove-orphans

# Install quality tools separately: .venv/bin/pip install -r requirements-dev.txt
.PHONY: lint format-check test check

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest

check: lint format-check test

# Quiet validation using public values; never load private .env.
.PHONY: docker-config
docker-config:
	ANPR_ENV_FILE=.env.example $(COMPOSE) --env-file .env.example $(COMPOSE_FILES) config --quiet
