from deepface import DeepFace
from pathlib import Path

QUERY_IMAGE = "face_db/Adam_Sandler/0379.jpg"
DATABASE = "face_db"


def find_matches(query_image, database):
    results = DeepFace.find(
        img_path=query_image,
        db_path=database,
        model_name="VGG-Face",
        detector_backend="skip",
        silent=True
    )

    return results[0]


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
