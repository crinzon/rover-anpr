from pathlib import Path

from full_stand.config import StandConfig
from full_stand.pipeline import PlatePipeline


IMAGE_PATH = Path("test.jpg")


config = StandConfig(device="cpu")
config.validate()

pipeline = PlatePipeline(config)

image_bytes = IMAGE_PATH.read_bytes()

result = pipeline(image_bytes)

if not result["plates"]:
    print("Номера не найдены")
else:
    for plate in result["plates"]:
        print("Номер:", plate["text"])
        print("Тип:", plate["subtype"])
        print("ГОСТ:", plate["valid"])
        print("Уверенность OCR:", plate["confidence"])
        print("Уверенность детектора:", plate["detection"])