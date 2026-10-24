# The four primary commands manage only the rover-anpr Compose project.
COMPOSE ?= docker compose
COMPOSE_FILES ?= -f compose.yaml -f compose.cuda.yaml
STACK = $(COMPOSE) --project-name rover-anpr $(COMPOSE_FILES)
PYTHON ?= python3

.PHONY: install run stop purge check test lint format-check check-rtsp docker-config

install:
	@test -f .env || (umask 077; cp .env.example .env)
	@mkdir -p checkpoints
	$(STACK) build
	$(STACK) run --rm --no-deps --user "$$(id -u):$$(id -g)" \
		-v "$(CURDIR)/checkpoints:/app/checkpoints:rw" \
		camera python download_models.py

run:
	@test -f .env || (echo "Create .env from .env.example first"; exit 1)
	@test -f checkpoints/plate_detector.pt -a -f checkpoints/plate_recognizer.pt || (echo "Place both checkpoints in checkpoints/"; exit 1)
	$(STACK) up -d --no-build

stop:
	$(STACK) stop

# Never use --volumes, global prune, or delete files here. Persistent data stays.
purge:
	$(STACK) down --rmi local --remove-orphans

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest

check: lint format-check test

# One capture probe, no neural models. Stop Camera first to avoid a second RTSP session.
RTSP_DURATION ?= 15
check-rtsp:
	$(STACK) run --rm --no-deps camera python check_rtsp.py --duration $(RTSP_DURATION)

docker-config:
	ANPR_ENV_FILE=.env.example $(STACK) --env-file .env.example config --quiet

# Compatibility with existing usage; primary commands above are preferred.
.PHONY: docker-install docker-run docker-stop docker-purge
docker-install: install
docker-run: run
docker-stop: stop
docker-purge: purge
