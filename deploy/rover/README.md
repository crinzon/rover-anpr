# Ubuntu 22.04 / RTX A2000: переносимый профиль

Это профиль существующего ANPR, а не готовность неизвестного протокола ровера.
FastAPI, модели и алгоритм подтверждения сохранены. Локально проверены CUDA на
RTX 5070, синтаксис Compose, Python-адаптер и архив шести видео. **Ubuntu 22.04,
RTX A2000 и реальный ROS/DDS на ровере не проверены.**

## Среда

- Ровер: Ubuntu 22.04 x86_64, i7-12700TE, RTX A2000 12 GB, RAM 32 GB.
- ANPR: существующие Linux-контейнеры Python 3.11; PyTorch 2.9.1 /
  torchvision 0.24.1 из cu128, текущие зависимости docker/*/requirements.txt.
- ROS 2 Humble остаётся в системном Ubuntu Python 3.10. Не устанавливать
  rclpy через pip в PyTorch-venv, не заменять системный Python ради ANPR.
- CUDA-wheel проверен на RTX 5070 и содержит sm_86 для Ampere. Это архитектурная
  совместимость с A2000/3050, а не измеренная производительность этих устройств.
- На целевой машине нужны совместимый NVIDIA Linux driver, Docker Engine,
  Docker Compose с поддержкой `gpus: all`, NVIDIA Container Toolkit и GNU Make.
  Системный CUDA Toolkit для готового torch-wheel обычно не нужен. Совместимость
  имеющегося драйвера проверяется до изменения пакетов.
- Windows 11/RTX 3050 сервера не тестировались. Возможен существующий Windows→WSL
  режим, но профиль ниже предназначен для Linux-вычислителя ровера.

Официальные источники совместимости:
[PyTorch 2.9.1/cu128](https://pytorch.org/get-started/previous-versions/),
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[GPU compute capability](https://developer.nvidia.com/cuda/gpus),
[ROS Humble Ubuntu packages](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html).
Никакие драйверы/Docker/ROS этим профилем автоматически не устанавливаются.

## Перенос и установка

На Ubuntu распаковать подготовленный `rover-anpr-transfer.zip` в выбранную папку.
Архив включает исходники и текущие веса, но не .env, БД, captures, результаты,
test_videos или виртуальные окружения. Не распаковывать поверх рабочего проекта
без резервной копии. После распаковки:

```bash
cd /opt/rover-anpr
sha256sum -c transfer_manifest.sha256
nvidia-smi
docker version
docker compose version
cd deploy/rover
make install
# Отредактировать .env: реальный VIDEO_SOURCE и пароль только локально.
make preflight
make run
```

`/opt/rover-anpr` — пример пути; отдельные файлы не содержат Windows/WSL-путей.
`make install` собирает существующие контейнеры и требует оба имеющихся
checkpoint. Не скачивает и не заменяет веса. Сборка требует сети/времени;
она здесь не выполнялась, поскольку Docker daemon недоступен. Локальные .env
не перезаписываются повторной установкой. Не запускать root Makefile вместо
этого профиля, если нужна установка без скачивания checkpoint.

Для проверки только конфигурации без демона и секретов: `make config`.
Если VIDEO_SOURCE — локальный видеофайл, его нужно явно смонтировать в camera
отдельным Compose override; путь хоста без bind mount внутри контейнера не виден.
USB требует передачи /dev/videoN; профиль по умолчанию настроен для RTSP.
Не утверждается, что USB/RTSP автоматически проверены этим файлом.

## Браузер, остановка, очистка

На ровере: `http://127.0.0.1:8000/live`. API сохранён на localhost, наружу
без авторизации не открыт. С сервера Windows через согласованный SSH-доступ:

```powershell
ssh -N -L 8000:127.0.0.1:8000 USER@ROVER_IP
```

Затем `http://127.0.0.1:8000/live` на сервере. Это туннель браузера, не протокол
управления ровером. Одновременно профиль рассчитан на одну camera-службу;
multi-camera Live/калибровка и выбор нескольких источников требуют отдельного
согласования. Общий анализ двух камер сейчас офлайн.

```bash
cd /opt/rover-anpr/deploy/rover
make stop
make run
make purge
```

`purge` останавливает и удаляет только контейнеры/сеть проекта. **Не удаляет**
volumes, образы, checkpoints, БД, результаты, captures или пользовательские файлы.
Не применяет prune/rm -rf. Общая очистка временных файлов умышленно отсутствует:
временные результаты диагностики могут быть доказательствами. Каждый новый
тест запускать в новую results-папку; хранение/ротацию согласовать отдельно.
Volume БД `rover-anpr_anpr-data` отличается от старой Windows plates.db;
автоматического переноса/обнуления рабочей БД не выполняется.

## Отдельный экспериментальный ROS 2 Humble адаптер

```bash
cd /opt/rover-anpr
source /opt/ros/humble/setup.bash
ros2 topic echo /anpr/experimental/events std_msgs/msg/String
```

В другом терминале, в той же ROS/DDS-среде:

```bash
source /opt/ros/humble/setup.bash
cd /opt/rover-anpr
/usr/bin/python3 -B -m adapters.ros2_humble --self-test
# Далее реальные доступные snapshot-события существующего API:
/usr/bin/python3 -B -m adapters.ros2_humble --api-url 'http://127.0.0.1:8000/plates?include_partial=true' --camera-id Front
```

`--self-test` помечает событие TEST_ONLY, PARTIAL, test_event=true; оно не является
распознаванием или подтверждением машины. Настоящий DDS publish/receive необходимо
проверить указанными командами на Ubuntu. Локально ROS отсутствует, установка
не выполнялась. Проверка без ROS:

```bash
python3 -B -m adapters.ros2_humble --self-test --dry-run
python3 -B -m adapters.ros2_humble --input /PATH/receiver/events.jsonl --dry-run
```

Формат — **экспериментальный std_msgs/String с JSON, не официальный ROS-интерфейс
ровера**. Внутри сохранён исходный event: plate, status, track_id, captured_at,
confidence, position.x/y и прочие имеющиеся поля. Если поля нет, оно не создаётся.
camera_id берётся из event либо явно заданной конфигурации; если нет — null.
`--camera-id Front` допустим только для известного источника Front; на произвольном
общем API это не способ определить камеру. track_id глобально уникален только
вместе с session_id. X/Y — нормализованные смещения центра номера в изображении,
не мировые координаты/метры. Время — timestamp имеющегося события, не время
экспозиции сенсора и не автоматически синхронизированный ROS clock.

Адаптер не трекает, не голосует и не меняет статус. В режиме API polling могут
теряться промежуточные события (TTL API 2 с), особенно при разрыве связи; это
не lossless-шина. Дедупликация ограничена памятью и сроком жизни процесса.
Для долговечной доставки нужен согласованный push/outbox-протокол; без запроса
команды ровера он не добавлялся. Для полной записи текущего потока уже существует
rover_event_receiver.py, но у producer один URL назначения: переключение на него
вместо API прекращает обновление API-истории. Новый скрытый fan-out не добавлен.

## Известные блокеры

При физической проверке телефона один номер создал 9 записей из-за фрагментации;
частные P0 и PARTIAL→CONFIRMED тесты проходят, общая проблема не устранена.
Ложные устойчивые объекты могут проходить временный registry gate. Иностранные
номера не объявлены поддержанными. До полевого запуска нужны размеченные
проезды, отрицательные примеры, RTSP/reconnect, проверка задержек и тепловой
нагрузки на A2000, согласованный транспорт/ACK/безопасность/система координат.
