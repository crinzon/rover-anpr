
import cv2

from pathlib import Path
from full_stand.config import StandConfig
from full_stand.pipeline import PlatePipeline

config = StandConfig(device="mps", confidence=0.5)
config.validate()
pipeline = PlatePipeline(config)

IMAGE_PATH = "test.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError("Image not found")

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

l = clahe.apply(l)

enhanced_lab = cv2.merge((l, a, b))
enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

cv2.imwrite("test_clahe.jpg", enhanced)


def recognize(image):
    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        raise RuntimeError("Encoding failed")

    output = pipeline(buffer.tobytes())

    for plate in output["plates"]:
        print(
            plate["text"],
            "| Уверенность:",
            round(plate["confidence"], 3)
        )


print("ОРИГИНАЛ")
recognize(image)

print("CLAHE")
recognize(enhanced)
