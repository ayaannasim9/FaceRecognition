import gc
import pickle
from pathlib import Path

from deepface import DeepFace
from deepface.modules import verification
import numpy as np
import pandas as pd

QUERY_IMAGE = "face_db/Adam_Sandler/0379.jpg"
DATABASE = "face_db"
CACHE_FILENAME = (
    "ds_model_vggface_detector_skip_aligned_"
    "normalization_base_expand_0.pkl"
)

_face_indexes = {}


class FaceIndex:
    def __init__(self, database):
        cache_path = Path(database) / CACHE_FILENAME
        if not cache_path.exists():
            raise RuntimeError(
                f"Embedding cache not found: {cache_path}. Run face_recog.py first."
            )

        print(f"Loading face index from {cache_path}...")
        with cache_path.open("rb") as cache_file:
            representations = pickle.load(cache_file)

        valid_representations = [
            representation
            for representation in representations
            if representation["embedding"] is not None
        ]
        embeddings = np.asarray(
            [representation["embedding"] for representation in valid_representations],
            dtype=np.float32,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.embeddings = embeddings / np.maximum(norms, 1e-10)
        self.identities = np.asarray(
            [representation["identity"] for representation in valid_representations]
        )
        self.threshold = verification.find_threshold("VGG-Face", "cosine")

        del representations
        gc.collect()
        print(f"Loaded {len(self.identities)} face embeddings")

    def find_matches(self, query_image):
        query = DeepFace.represent(
            img_path=query_image,
            model_name="VGG-Face",
            detector_backend="skip",
            normalization="base",
        )[0]["embedding"]
        query = np.asarray(query, dtype=np.float32)
        query /= max(np.linalg.norm(query), 1e-10)

        distances = 1 - self.embeddings @ query
        matching_indices = np.flatnonzero(distances <= self.threshold)
        matching_indices = matching_indices[
            np.argsort(distances[matching_indices])
        ]

        return pd.DataFrame(
            {
                "identity": self.identities[matching_indices],
                "distance": distances[matching_indices],
                "threshold": self.threshold,
            }
        )


def prepare_face_recognition(database):
    database_path = str(Path(database).resolve())
    if database_path not in _face_indexes:
        _face_indexes[database_path] = FaceIndex(database)
        DeepFace.build_model("VGG-Face")

    return _face_indexes[database_path]


def find_matches(query_image, database):
    face_index = prepare_face_recognition(database)
    return face_index.find_matches(query_image)


def identify_face(matches, number_of_matches=5):
    if matches.empty:
        return None

    top_matches = matches.head(number_of_matches).copy()
    top_matches["person"] = top_matches["identity"].apply(
        lambda path: Path(path).parent.name
    )

    return top_matches["person"].mode().iloc[0]


def main():
    matches = find_matches(QUERY_IMAGE, DATABASE)
    predicted_person = identify_face(matches)
    
    print(f"Predicted person: {predicted_person}")


if __name__ == "__main__":
    main()
