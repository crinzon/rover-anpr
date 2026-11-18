# rover-anpr

Распознавание российских автомобильных номеров: YOLO26n + OCR,
подтверждение по кадрам, Camera и FastAPI, SQLite и живой просмотр.
[Архитектура системы](docs/architecture.puml).

## Требования

WSL Ubuntu 22.04 или Linux, Docker Compose и GNU Make. Для NVIDIA GPU —
драйвер и NVIDIA Container Toolkit с поддержкой CUDA 12.8.
Нужен RTSP-поток или видеофайл; при установке требуется интернет.

## Установка и запуск

```bash
git clone https://github.com/crinzon/rover-anpr.git
cd rover-anpr
make install
# Настройте VIDEO_SOURCE в .env; DEVICE=cuda, SHOW_WINDOW=false.
make run
make stop
make purge
```

- `install` — собирает контейнеры и автоматически загружает обе модели,
  проверяя размер, SHA256 и загрузку весов. Существующие веса не заменяет;
  при несовпадении контрольной суммы останавливается.
- `run` — запускает Camera и API без пересборки.
- `stop` — останавливает систему.
- `purge` — удаляет ресурсы Compose проекта, сохраняя volumes и файлы.

Настройки берутся из `.env.example`. RTSP-адрес с паролем храните только
в `.env`; не публикуйте этот файл. Для CPU задавайте
`COMPOSE_FILES="-f compose.yaml"` при `install` и `run`.

Веб-интерфейс: **http://localhost:8000/live**.

`GET /plates` возвращает актуальные треки с индивидуальным сроком жизни.
Пиксельные смещения, время кадра и совместимость старых отправителей
описаны в [API-контракте](docs/api-contract.md).

## Обработка записей робота

В локальном Python-окружении с зависимостями и двумя моделями:

```bash
python process_video.py --input test_videos --output results --device cpu
```

CLI рекурсивно обрабатывает MP4/MOV без API, сохраняет структуру каталогов,
размеченные MP4 и JSON/CSV. Для NVIDIA используйте `--device cuda`.
Существующие результаты не перезаписываются.
[Временные метки, отчёты, ограничения и проверки](docs/video-processing.md).

[Проверенные форматы, PARTIAL/CONFIRMED и ограничения](docs/plate-formats.md).
