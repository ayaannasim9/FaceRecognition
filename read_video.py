import cv2 as cv
from face_recog import find_matches, identify_face
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

VIDEO_PATH = "IMG_6113.MOV"
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
DATABASE = "face_db"
DETECTION_SCALE=0.5
MIN_FACE_SIZE=70


def detect_faces(frame, face_cascade):
    small_frame=cv.resize(
        frame,
        None,
        fx=DETECTION_SCALE,
        fy=DETECTION_SCALE,
        interpolation=cv.INTER_AREA
    )

    frame_gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    minimum_size=int(MIN_FACE_SIZE*DETECTION_SCALE)

    small_faces=face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(minimum_size,minimum_size)
    )
    # frame_gray = cv.equalizeHist(frame_gray)

    return [
        (
            int(x/DETECTION_SCALE),
        int(y/DETECTION_SCALE),
        int(width/DETECTION_SCALE),
        int(height/DETECTION_SCALE),
        )
        for x,y,width, height in small_faces
    ]


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

def voting(new_prediction, prediction_history, predicted_person):
    prediction_history.append(new_prediction)
    most_common_person, votes=Counter(prediction_history).most_common(1)[0]

    if votes>=2:
        predicted_person=most_common_person
    return predicted_person

def recognize_face(cropped_face):
    matches=find_matches(cropped_face,database=DATABASE)
    new_prediction=identify_face(matches) or "Unknown"

    return new_prediction

def play_video(video_path):
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    face_cascade = cv.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Unable to load cascade: {CASCADE_PATH}")

    frame_number=0
    predicted_person="Unknown"
    prediction_history=deque(maxlen=3)

    executor=ThreadPoolExecutor(max_workers=1)
    recognition_job=None
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_number+=1
        if recognition_job is not None and recognition_job.done():
            new_prediction=recognition_job.result()

            predicted_person=voting(new_prediction,prediction_history,predicted_person)

            recognition_job=None
    
        faces = detect_faces(frame, face_cascade)
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, width, height = largest_face

            if recognition_job is None:
                face_crop = frame[y:y + height, x:x + width]
                recognition_job=executor.submit(recognize_face,face_crop.copy())


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

    executor.shutdown()
    capture.release()
    cv.destroyAllWindows()


def main():
    play_video(VIDEO_PATH)


if __name__ == "__main__":
    main()
