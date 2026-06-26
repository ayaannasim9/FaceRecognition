from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people


PROJECT_DIR = Path(__file__).resolve().parent
DATA_HOME = PROJECT_DIR / ".sklearn_data"
FACE_DB = PROJECT_DIR / "face_db"


def clean_name(name):
    return name.replace(" ", "_").replace("/", "_")


lfw_people = fetch_lfw_people(
    data_home=DATA_HOME,
    min_faces_per_person=3,
    color=True,
    resize=1.0,
)

FACE_DB.mkdir(exist_ok=True)

for index, image in enumerate(lfw_people.images):
    person = clean_name(lfw_people.target_names[lfw_people.target[index]])
    person_dir = FACE_DB / person
    person_dir.mkdir(exist_ok=True)

    # LFW images arrive as numpy arrays. Convert safely to normal JPG pixels.
    if image.max() <= 1:
        image = image * 255

    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(person_dir / f"{index:04d}.jpg")

print(f"Saved {len(lfw_people.images)} images into {FACE_DB}")
