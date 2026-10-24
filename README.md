# Rover ANPR

Распознавание российских автомобильных номеров для ровера: **YOLO26n → OCR →
проверка формата → подтверждение по наблюдениям → относительное положение**.
OpenCV получает RTSP/видео, PyTorch выполняет инференс, FastAPI отдаёт результаты,
SQLite хранит историю. Веб-просмотр — <http://localhost:8000/live>.

## Требования и настройка

Основной стенд: Linux/WSL2, Docker Compose с поддержкой `gpus: all`, NVIDIA
Container Toolkit / GPU-интеграция Docker Desktop и совместимый драйвер.
Проверено на RTX 5070 Laptop GPU, PyTorch **2.9.1+cu128**, OpenCV **5.0.0**.
В Docker используется Python 3.11; системный Python для запуска не нужен.
Нужен GNU Make (Ubuntu: `sudo apt-get install make`).

1. Получите веса из разрешённого источника и разместите локально:
   `checkpoints/plate_detector.pt`, `checkpoints/plate_recognizer.pt`.
   Они не публикуются и подключаются read-only; автоматического скачивания нет.
2. Создайте `.env` из `.env.example`, только если своего файла ещё нет.
   Укажите `VIDEO_SOURCE` с RTSP URL. Типовой путь Hikvision:
   `/Streaming/Channels/101` (основной) или `102` (дополнительный).
   Спецсимволы логина/пароля percent-encode; выполните `chmod 600 .env`.
   Не публикуйте URL, `.env`, `docker inspect` и полный `docker compose config`.
3. Основные значения: `DEVICE=cuda`, `SHOW_WINDOW=false`,
   `VIDEO_BACKEND=auto`, `RTSP_TRANSPORT=tcp`. Compose принудительно задаёт
   CUDA через override, отсутствие окна и внутренний адрес API.

## Четыре команды

```sh
make install  # подготовить .env при отсутствии и собрать образы с CUDA
make run      # запустить API и Camera из готовых образов
make stop     # штатно остановить сервисы
make purge    # удалить контейнеры, сети и локальные образы только rover-anpr
```

**`purge` не удаляет volumes, SQLite, снимки, веса, видео и результаты.**
Не используйте `down -v` или глобальный `docker system prune` для этого проекта.
CPU-вариант: добавьте `COMPOSE_FILES="-f compose.yaml"` к команде.
Первичная сборка тяжёлая; изменение `.env` требует пересоздания контейнера,
а не пересборки. После изменения исходников выполните `make install && make run`.
SQLite и снимки находятся в volumes `rover-anpr_anpr-data` и
`rover-anpr_anpr-captures`; существующие локальные файлы туда не переносятся.

RTSP без моделей: остановите Camera (`docker compose --project-name rover-anpr
-f compose.yaml -f compose.cuda.yaml stop camera`), затем
`make check-rtsp RTSP_DURATION=20`. После проверки — `make run`.
Не запускайте второй экземпляр распознавания параллельно Camera.

## API и просмотр

API опубликован только на `127.0.0.1:8000`, без отдельной авторизации.

| Endpoint | Назначение |
|---|---|
| `GET /health` | Готовность HTTP/SQLite, не камеры или моделей |
| `POST /plates` | Приём подтверждённых событий Camera с session/event/track ID |
| `GET /plates` | Последний принятый пакет с относительной позицией; срок 2 с |
| `GET /history` | История: номер, confidence, время приёма, путь снимка |
| `GET /live`, `/live/frame`, `/live/status` | Свежий JPEG, FPS, задержки, состояние ANPR |
| `GET /live/stream` | MJPEG для совместимости |
| `GET /live/diagnostics` | Последний диагностический стоп-кадр, если включён |
| `GET /captures/...` | Сохранённые снимки |

Превью использует тот же захват; отдельные модели не загружаются.
`LIVE_FPS=20`, `LIVE_MAX_WIDTH=1920`, `LIVE_JPEG_QUALITY=90` управляют только
превью, не входом моделей. При отсутствии свежих кадров показывается «Нет
сигнала». Рамки последних результатов скрываются через `LIVE_OVERLAY_MAX_AGE`.
Для временной диагностики: `ANPR_DIAGNOSTICS=true`, перезапуск Camera,
страница `/live?debug=1`. Там доступны все выходные proposals YOLO, OCR-кропы,
проверки формата и причины отказа. После проверки выключите этот режим.

## Тестирование и результаты

```sh
python3 -m venv .venv       # Ubuntu: пакет python3-venv
.venv/bin/pip install -r requirements-dev.txt -r docker/api/requirements.txt
make check PYTHON=.venv/bin/python  # Ruff, форматирование и pytest
make docker-config         # тихая проверка Compose только на публичном шаблоне
```

Тесты моделей/OpenCV автоматически пропускаются без runtime-зависимостей.
В окружении Camera дополнительно выполняются проверки реального OpenCV и
постобработки на искусственных данных, без загрузки весов и подключения RTSP.
Фотостенд сохранён: после установки `requirements.txt` и весов запустите
`python -m full_stand.app --port 8080` (отдельная модель, не вместе с Camera).

Для пяти файлов `test_videos/` используется **тот же `camera.main`**: режим
`evaluate_videos.py` ждёт обработки каждого кадра и считает таймауты по PTS.
Запуск в существующем образе, после остановки Camera:

```sh
docker compose --project-name rover-anpr -f compose.yaml -f compose.cuda.yaml \
  run --rm --no-deps -v "$PWD:/workspace:ro" -w /workspace camera \
  python evaluate_videos.py --videos /workspace/test_videos \
  --output /app/captures/evaluation_results/new_run --api-base http://api:8000
```

Выходной каталог должен быть новым. Артефакты доступны в volume снимков;
`docker cp rover-anpr-api-1:/app/captures/evaluation_results/new_run evaluation_results/`
копирует их локально. `python evaluate_videos.py --report-only <каталог>`
пересобирает отчёт **только в указанном каталоге**; исходные эксперименты не менять.

Эксперимент 24.10.2026: **2185/2185 кадров, 37.01 FPS, 17 итоговых треков**;
пользователь вручную проверил все 17 итоговых результатов. Были промежуточное
переключение Y/X и обработка IMG_2901 ниже 30 FPS. Это не измерение общей
precision/recall. Исходные результаты сохранены локально в
`evaluation_results/20261024_five_videos/`, вместе с видео исключены из Git.

## Архитектура и ограничения

[DFD в PlantUML](docs/architecture.puml) отражает текущие потоки данных.
GitHub показывает исходник `.puml`; для изображения откройте его расширением
PlantUML или отрендерите локально. Дополнительных архитектурных документов нет.

- Рабочие пороги сохранены: YOLO 0.5, OCR ≥ 0.85, подтверждение 3 из последних
  5 принятых наблюдений, смена 7 из истории 10. Поддержан гражданский формат РФ,
  не полная юридическая проверка ГОСТ и не все 29 подтипов OCR в основном API.
- `GET /plates` — последний пакет, а не гарантированный список всех видимых
  машин. Координаты передаются через API, но **не сохраняются в SQLite**.
- Очередь доставки в памяти допускает потери при переполнении/остановке.
  Confidence — оценка модели, не измеренная точность.
- ONVIF Profile G **не реализован**. Производитель заявляет его для
  DS-2CD2047G2H-LIU/SL, но архив установленной камеры не проверен. Сначала нужны
  точная модель/прошивка, сервисы Recording/Search/Replay, microSD с записями,
  проверка UTC/NTP и получение короткого архивного фрагмента для того же pipeline.
  Модель в спецификации — 4 MP; наблюдавшийся 4K-поток требует сверки устройства.
  [Profile G](https://www.onvif.org/profiles/profile-g/),
  [клиент ONVIF](https://github.com/nirsimetri/onvif-python).
- ROS2 и QR-стыковка — будущий этап. Обучение/генератор сохранены локально вне
  публикуемого runtime; веса и права их распространения не включены в релиз.

## Hooks и релизы

Подключение один раз в клоне: `git config core.hooksPath .githooks`.
Hook повышает PATCH в `VERSION`, добавляет его в индекс и не вызывает commit.
Повторная попытка коммита с уже подготовленным VERSION не повышает версию снова.
GitHub Actions запускает быстрые проверки; тег `v<содержимое VERSION>` запускает
проверки и выпуск исходного `tar.gz` с SHA-256. В архиве нет видео, весов,
SQLite, секретов, пользовательских результатов и исследовательских материалов.
**Commit, push и отправка релизного тега выполняются только владельцем.**
