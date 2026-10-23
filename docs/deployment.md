# Запуск ровера и Docker

## Ровер: локальный запуск и Docker

Локально (macOS/Linux): `make install` создаёт `.venv` и ставит
`requirements.txt`. Подготовьте собственный `.env` по `.env.example`, затем
`make run`; `make stop` останавливает процессы. На Mac оставьте `DEVICE=mps`,
для CPU задайте `DEVICE=cpu`. Веса должны находиться в `checkpoints/`.
`make purge` останавливает процессы и **удаляет `.run` с PID и логами**;
база, снимки и веса сохраняются. Не запускайте очистку без необходимости.
Инструменты качества устанавливаются отдельно:
`.venv/bin/pip install -r requirements-dev.txt`; проверка — `make check`.

Compose по умолчанию использует CPU и не требует GPU. Команды:

```sh
make docker-config
make docker-install
make docker-run
make docker-stop
```

Для Ubuntu x86_64/NVIDIA нужен драйвер, совместимый с CUDA 12.6,
NVIDIA Container Toolkit и Compose с поддержкой `gpus: all`:

```sh
make docker-config COMPOSE_FILES="-f compose.yaml -f compose.cuda.yaml"
make docker-install COMPOSE_FILES="-f compose.yaml -f compose.cuda.yaml"
make docker-run COMPOSE_FILES="-f compose.yaml -f compose.cuda.yaml"
make docker-stop COMPOSE_FILES="-f compose.yaml -f compose.cuda.yaml"
```

В Docker MPS недоступен; Docker Desktop на Mac использует CPU. Доступ USB
в контейнер автоматически не предоставляется: на Linux нужен отдельный
локальный override с `devices` и группой устройства. На Mac USB-камеру
используйте через локальный запуск либо выберите доступный RTSP-поток.

Камера читает `.env` через Compose; этот файл не включается в образ.
Для другого файла задайте `ANPR_ENV_FILE`. Для Hikvision укажите собственный
RTSP URL в `VIDEO_SOURCE`; реальные адреса и пароли в README не размещаются.
`VIDEO_RETRY_INITIAL`, `VIDEO_RETRY_MAX`, `VIDEO_OPERATION_TIMEOUT` управляют
переподключением; `DELIVERY_*` — доставкой. В контейнере принудительно заданы
внутренний адрес API, CPU/CUDA и отсутствие графического окна. Не публикуйте
вывод обычного `docker compose config`: он может раскрыть секреты.
`make docker-config` использует только `.env.example` и тихую проверку.

API доступен на `127.0.0.1:8000`; `/health` подтверждает запуск HTTP-приложения
после инициализации SQLite, но не проверяет камеру или качество OCR.
Камера стартует после успешного healthcheck API; последующие сбои API
обрабатывает очередь доставки. Оба сервиса перезапускаются `unless-stopped`.

Контейнеры работают под UID/GID 10001. SQLite и снимки хранятся в именованных
volumes `anpr-data` и `anpr-captures`, начальные права задаются образом.
Существующие локальные `plates.db`, `data/` и `captures/` автоматически
не переносятся. Для bind mounts заранее подготовьте права UID/GID 10001;
не меняйте владельцев пользовательских данных вслепую.
`make docker-stop` сохраняет volumes. `make docker-purge` удаляет контейнеры
и локальные образы, но не volumes; `docker compose down -v` удаляет данные.

Образ камеры явно копирует runtime-модули, `full_stand` и `ocr_model`.
Модели подключаются через read-only volume. Генератор и датасеты не нужны.
Основные Docker-зависимости
закреплены по версиям локального окружения; транзитивные зависимости,
базовые образы и apt-пакеты пока не зафиксированы по хешам. Это ещё не полный
lock для воспроизводимой сборки. Образы не проверены сборкой в текущей среде;
совместимость CUDA и производительность требуют проверки на целевой машине.
