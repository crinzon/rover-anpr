# Данные, генерация и обучение

## Какие данные используются

**Детектор** обучается на фотографиях с размеченными рамками номеров. **OCR** — на вырезанных реальных номерах с текстовой разметкой и синтетике собственного генератора.

| Задача | Источник | Изображений после отбора |
|---|---|---:|
| Детекция | [AUTO.RIA / Nomeroff Net](https://nomeroff.net.ua/), архив 2026-06-04 | 19 888 |
| Детекция | [Car_plate_detecting_dataset](https://huggingface.co/datasets/AY000554/Car_plate_detecting_dataset) | 24 822 |
| Детекция | [Car plate object detection](https://www.kaggle.com/datasets/andrewteplov/car-plate-object-detetcion) | 482 |
| Детекция | [Number plates, 50% Russian / 50% others](https://www.kaggle.com/datasets/kirillpribludenko/number-plates-50-russain-50-others) | 369 |
| OCR | [AUTO.RIA OCR RU, 2021-09-01](https://web.archive.org/web/20220514070933id_/https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip), гражданские номера | 56 611 |
| OCR | [AUTO.RIA OCR RU Military, 2022-03-25](https://web.archive.org/web/20220514075608id_/https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-Military-2022-03-25.zip), военные номера | 27 513 |
| OCR | [MADE CV / Car plates OCR](https://www.kaggle.com/datasets/xairete/car-plates-ocr), кропы по полигонам разметки | 26 799 |
| OCR | [Car_plate_OCR_dataset](https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset), уникальная часть | 4 747 |

## Как сгенерировать данные

Используйте окружение из [инструкции веб-стенда](web-stand.md). Основной вариант генерации работает через **Blender Cycles на NVIDIA GPU**. Подготовьте Blender, затем создайте выборку:

```powershell
python -m generator.setup --blender
python -m generator generate outputs/synthetic --count 5000 --seed 42 --backend cycles --device CUDA --profile gost --preview
python -m generator validate outputs/synthetic
```

Для небольшой пробы уменьшите `--count`, например до 32. Генерация через `--backend cpu` доступна без Blender и GPU — в этом случае первый шаг можно пропустить.

Генератор выбирает тип и текст номера, строит знак по геометрии ГОСТ, затем моделирует съёмку: меняет ракурс, освещение, загрязнение, блики и смаз. Вместе с изображением автоматически создаётся разметка текста, типа и положения знака. Так можно дополнить реальные данные примерами редких типов и разных условий съёмки.

Для текущего OCR используется профиль `gost`. Количество задаётся через `--count`, seed — через `--seed`; только целевые типы можно выбрать через `--types type1 type1a type1b`.

В `outputs/synthetic/` появятся изображения, разметка `annotations.jsonl`, конфигурация и отчёт о генерации. Посмотреть примеры можно в `preview.jpg` или `gallery.html`. Этот же каталог передаётся в обучение OCR.

## Как обучить модели

Примеры ниже рассчитаны на **одну NVIDIA GPU** и окружение из [инструкции веб-стенда](web-stand.md). Для OCR сначала сгенерируйте синтетику командой выше.

### Детектор

```powershell
yolo detect train model=yolo26n.pt data=real_plates2/yolo_plates_ru2/data.yaml `
    epochs=400 patience=40 imgsz=640 batch=16 device=0 project=outputs name=detector
```

Начальные веса `yolo26n.pt` скачиваются автоматически. Лучшие веса сохраняются в `outputs/detector/weights/best.pt`; при повторных запусках имя каталога может получить числовой суффикс.

### OCR

Обучение на реальных номерах и сгенерированных изображениях, валидация — на отдельной реальной выборке:

```powershell
python ocr_model/run_training.py `
    --data real_plates2/ocr_plates_ru/train.tsv outputs/synthetic `
    --val-data real_plates2/ocr_plates_ru/val.tsv `
    --output outputs/ocr --epochs 25 --gpus 0 --batch-size 32 --workers 4
```

Лучшие веса сохраняются в `outputs/ocr/best.pt`, последние — в `outputs/ocr/last.pt`. Для проверки одного шага добавьте `--dry-run`. При нехватке видеопамяти уменьшите батч; если GPU не поддерживает BF16, добавьте `--amp none`.

Для обучения только на реальных данных уберите `outputs/synthetic` из команды. Передавайте `train.tsv` и `val.tsv` отдельно, чтобы сохранить готовое разделение данных.

По сведениям прежней документации локальные веса `checkpoints/plate_recognizer.pt` обучены 25 эпох на **2 386 685 изображениях**: 109 886 реальных кропов из `real_plates2/ocr_plates_ru/train.tsv` и 2 276 799 синтетических. Синтетика закрывает все 29 подтипов, с упором на основные: `type1` — 283 660 изображений, `type1a` — 313 639, `type1b` — 287 707, у остальных типов по 52–54 тысячи. Валидация смешанная: 5 784 реальных кропа и 43 413 синтетических, доля полностью верных ответов (текст и подтип одновременно) на ней — 0.976.

### Проверить свои модели на стенде

```powershell
python full_stand/app.py `
    --detector outputs/detector/weights/best.pt `
    --recognizer outputs/ocr/best.pt --port 8080
```

Откройте тот же адрес [http://127.0.0.1:8080](http://127.0.0.1:8080) и загрузите изображения для проверки.

Полный список параметров доступен через `--help` у соответствующей команды.


Эти сведения об обучении и метрике перенесены из прежней документации;
они не были независимо воспроизведены при инженерной проверке проекта.
Датасеты и checkpoint не распространяются вместе с репозиторием.
До генерации подготовьте [локальные глифы](local-resources.md).
