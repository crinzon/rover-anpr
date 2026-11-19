# Standalone Ubuntu profile. Root/Windows Makefile and .env are unchanged.
PROJECT_ROOT := $(abspath ../..)
ENV_FILE := $(CURDIR)/.env
COMPOSE ?= docker compose
STACK = ANPR_ENV_FILE="$(ENV_FILE)" $(COMPOSE) --project-directory "$(PROJECT_ROOT)" --project-name rover-anpr -f "$(PROJECT_ROOT)/compose.yaml" -f "$(PROJECT_ROOT)/compose.cuda.yaml"
.PHONY: install run stop purge config preflight

# Never download/replace user weights. Copy the existing verified checkpoints.
install:
	@test -f .env || (umask 077; cp .env.example .env)
	@test -f "$(PROJECT_ROOT)/checkpoints/plate_detector.pt" -a -f "$(PROJECT_ROOT)/checkpoints/plate_recognizer.pt" || (echo 'Copy both original checkpoints first'; exit 1)
	$(STACK) build

run:
	@test -f .env || (echo 'Run make install and edit deploy/rover/.env'; exit 1)
	$(STACK) up -d --no-build

stop:
	$(STACK) stop

# Stops/removes only this project's containers/network. NO volumes/images/data deletion.
# Models, DB volumes, captures, results and user files are deliberately retained.
purge:
	$(STACK) down --remove-orphans

config:
	ANPR_ENV_FILE="$(CURDIR)/.env.example" $(COMPOSE) --project-directory "$(PROJECT_ROOT)" --project-name rover-anpr -f "$(PROJECT_ROOT)/compose.yaml" -f "$(PROJECT_ROOT)/compose.cuda.yaml" config --quiet

preflight:
	$(STACK) run --rm --no-deps camera python -c 'import torch; from full_stand.config import StandConfig; from full_stand.pipeline import PlatePipeline; assert torch.cuda.is_available(); p=PlatePipeline(StandConfig(device="cuda",confidence=.5)); p.warmup(); torch.cuda.synchronize(); print(torch.__version__,torch.cuda.get_device_name(0),p.device)'
