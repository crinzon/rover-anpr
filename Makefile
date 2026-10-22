PYTHON := .venv/bin/python
PIP := .venv/bin/pip

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
	docker compose build

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

docker-purge:
	docker compose down --rmi local --remove-orphans
