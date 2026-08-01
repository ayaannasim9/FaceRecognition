from collections import defaultdict
from pathlib import Path
import re
import unicodedata

from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parent
PARQUET_FILE = (
    PROJECT_DIR
    / "datasets"
    / "celebrity_1000"
    / "train-00000-of-00001.parquet"
)
FACE_DB = PROJECT_DIR / "face_db"
CACHE_DIR = PROJECT_DIR / "datasets" / ".cache"


def folder_name(celebrity_name):
    ascii_name = unicodedata.normalize("NFKD", celebrity_name)
    ascii_name = ascii_name.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")


def export_selected_faces():
    dataset = load_dataset(
        "parquet",
        data_files=str(PARQUET_FILE),
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    label_names = dataset.features["label"].names
    image_counts = defaultdict(int)
    labels_by_folder = defaultdict(set)

    for row in dataset:
        celebrity_name = label_names[row["label"]]
        output_folder = folder_name(celebrity_name)
        labels_by_folder[output_folder].add(celebrity_name)

        image_counts[output_folder] += 1
        output_directory = FACE_DB / output_folder
        output_directory.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_directory
            / f"dataset_{image_counts[output_folder]:04d}.jpg"
        )
        row["image"].convert("RGB").save(output_path, quality=95)

    merged_labels = {
        output_folder: labels
        for output_folder, labels in labels_by_folder.items()
        if len(labels) > 1
    }
    print(f"Exported {sum(image_counts.values())} images")
    print(f"Created or updated {len(image_counts)} celebrity folders")
    for output_folder, labels in sorted(merged_labels.items()):
        print(f"Merged {sorted(labels)} into {output_folder}")


if __name__ == "__main__":
    export_selected_faces()
