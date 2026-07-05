import cv2 as cv

VIDEO_PATH = "face_vid.mp4"
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"


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


def play_video(video_path):
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    face_cascade = cv.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Unable to load cascade: {CASCADE_PATH}")

    while True:
        success, frame = capture.read()
        if not success:
            break

        faces = detect_faces(frame, face_cascade)
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
