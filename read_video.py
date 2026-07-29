import cv2 as cv
from face_recog import find_matches, identify_face
from collections import Counter, deque

VIDEO_PATH = "IMG_6113.MOV"
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
DATABASE = "face_db"


def detect_faces(frame, face_cascade):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_gray = cv.equalizeHist(frame_gray)

    return face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(70, 70),
    )


def draw_faces(frame, faces):
    for x, y, width, height in faces:
        cv.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            4,
        )

def clean(name):
    return name.translate(str.maketrans("_", " "))
    
def play_video(video_path):
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    face_cascade = cv.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Unable to load cascade: {CASCADE_PATH}")

    frame_number=0
    predicted_person="Unkown"
    prediction_history=deque(maxlen=3)
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_number+=1
        faces = detect_faces(frame, face_cascade)
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, width, height = largest_face

            if frame_number % 30 == 0:
                face_crop = frame[y:y + height, x:x + width]
                matches = find_matches(face_crop, database=DATABASE)
                new_prediction = identify_face(matches) or "Unknown"
                prediction_history.append(new_prediction)

                most_common_person, votes=Counter(prediction_history).most_common(1)[0]

                if votes>=2:
                    predicted_person=most_common_person

            cv.putText(
                frame,
                clean(predicted_person),
                (x, max(y - 10, 20)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

        draw_faces(frame, faces)
        cv.imshow("Capture - Face detection", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()


def main():
    play_video(VIDEO_PATH)


if __name__ == "__main__":
    main()
