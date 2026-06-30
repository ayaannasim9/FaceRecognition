from deepface import DeepFace

query_image="/Users/ayaannasim/Documents/face_recognition/face_db/Jake_Gyllenhaal/3227.jpg"
database="face_db"
results=DeepFace.find(img_path=query_image,
                      db_path=database,
                      model_name="VGG-Face",
                      detector_backend="skip")
matches=results[0]
print(matches[["identity", "distance"]].head())