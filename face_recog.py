from deepface import DeepFace
from pathlib import Path

query_image="/Users/ayaannasim/Documents/face_recognition/face_db/Adam_Sandler/0379.jpg"
database="face_db"
results=DeepFace.find(img_path=query_image,
                      db_path=database,
                      model_name="VGG-Face",
                      detector_backend="skip")
matches=results[0]
# print(matches[["identity", "distance"]].head())

def identify_face():
    top_matches=matches[["identity"]].head().copy()
    top_matches["person"]=top_matches["identity"].apply(lambda path: Path(path).parent.name)

    top_match=top_matches["person"].mode()
    predicted_person=top_match.iloc[0]
    print(f"predicted person : {predicted_person}")
    
    
identify_face()