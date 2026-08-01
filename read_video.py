from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from face_track import FaceTrack

import cv2 as cv

from face_recog import find_matches, identify_face


VIDEO_PATH = "IMG_6132.MOV"
OUTPUT_PATH = "annotated_video.mp4"
DATABASE = "face_db"
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

DETECTION_SCALE = 0.5
MIN_FACE_SIZE = 70
VOTE_HISTORY_SIZE = 3
VOTES_REQUIRED = 2

BOX_COLOR = (255, 0, 0)
BOX_THICKNESS = 4

IOU_THRESHOLD=0.3
MAX_MISSED_FRAMES=8


def load_face_cascade():
    face_cascade = cv.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Unable to load cascade: {CASCADE_PATH}")

    return face_cascade


def create_video_writer(capture, output_path):
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Unable to create video: {output_path}")

    return writer


def detect_faces(frame, face_cascade):
    small_frame = cv.resize(
        frame,
        None,
        fx=DETECTION_SCALE,
        fy=DETECTION_SCALE,
        interpolation=cv.INTER_AREA,
    )
    frame_gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
    minimum_size = int(MIN_FACE_SIZE * DETECTION_SCALE)

    small_faces = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=1.05,
        minNeighbors=10,
        minSize=(minimum_size, minimum_size),
    )

    return [
        (
            int(x / DETECTION_SCALE),
            int(y / DETECTION_SCALE),
            int(width / DETECTION_SCALE),
            int(height / DETECTION_SCALE),
        )
        for x, y, width, height in small_faces
    ]


def largest_face(faces):
    return max(faces, key=lambda face: face[2] * face[3])


def draw_faces(frame, faces):
    for x, y, width, height in faces:
        cv.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            BOX_COLOR,
            BOX_THICKNESS,
        )


def draw_name(frame, face, name):
    x, y, _, _ = face
    display_name = name.replace("_", " ")

    cv.putText(
        frame,
        display_name,
        (x, max(y - 10, 20)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        BOX_COLOR,
        2,
    )


def vote_for_prediction(new_prediction, prediction_history, current_prediction):
    prediction_history.append(new_prediction)
    most_common_person, votes = Counter(prediction_history).most_common(1)[0]

    if votes >= VOTES_REQUIRED:
        return most_common_person

    return current_prediction


def recognize_face(cropped_face):
    matches = find_matches(cropped_face, database=DATABASE)
    return identify_face(matches) or "Unknown"


def play_video(video_path, output_path):
    tracks={}
    next_track_id=1
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    writer = None

    try:
        face_cascade = load_face_cascade()
        writer = create_video_writer(capture, output_path)

        predicted_person = "Unknown"
        prediction_history = deque(maxlen=VOTE_HISTORY_SIZE)
        recognition_job = None

        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if recognition_job is not None and recognition_job.done():
                    new_prediction = recognition_job.result()
                    predicted_person = vote_for_prediction(
                        new_prediction,
                        prediction_history,
                        predicted_person,
                    )
                    recognition_job = None

                faces = detect_faces(frame, face_cascade)
                matched_track_ids=set()

                for face in faces:
                    best_iou=0
                    best_track=None
                    for track_id, track in tracks.items():
                        if track_id in matched_track_ids:
                            continue
                        iou=track.calculate_iou(face)
                        if iou>best_iou:
                            best_iou=iou
                            best_track=track
                    if best_iou>=IOU_THRESHOLD:
                        best_track.box=face
                        best_track.missed_frames=0
                        best_track.consecutive_hits+=1
                        if best_track.consecutive_hits>=3:
                            best_track.confirmed=True
                        matched_track_ids.add(best_track.track_id)
                    else:
                        tracks[next_track_id]=FaceTrack(track_id=next_track_id, box=face)
                        matched_track_ids.add(next_track_id)
                        next_track_id+=1
                for track_id, track in list(tracks.items()):
                    if track_id not in matched_track_ids:
                        track.missed_frames+=1
                        track.consecutive_hits=0
                    if track.missed_frames>=MAX_MISSED_FRAMES:
                        del tracks[track_id]
                    

                # if faces:
                #     primary_face = largest_face(faces)
                #     x, y, width, height = primary_face

                    # if recognition_job is None:
                    #     face_crop = frame[y : y + height, x : x + width]
                    #     recognition_job = executor.submit(
                    #         recognize_face,
                    #         face_crop.copy(),
                    #     )

                    # draw_name(frame, primary_face, predicted_person)

                for track in tracks.values():
                    if not track.confirmed:
                        continue
                    draw_faces(frame,[track.box])
                    draw_name(frame,track.box, f"{track.track_id}")
                writer.write(frame)
                cv.imshow("Capture - Face detection", frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv.destroyAllWindows()


def main():
    play_video(VIDEO_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
