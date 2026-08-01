from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from face_track import FaceTrack

import cv2 as cv

from face_recog import find_matches, identify_face, prepare_face_recognition


VIDEO_PATH = "IMG_6133.MOV"
OUTPUT_PATH = "annotated_video.mp4"
DATABASE = "face_db"
CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

DETECTION_SCALE = 0.5
MIN_FACE_SIZE = 120
VOTE_HISTORY_SIZE = 3
VOTES_REQUIRED = 2
RECOGNITION_INTERVAL_FRAMES = 30
FACE_CROP_PADDING = 0.2
SCALE_FACTOR=1.1
MIN_NEIGHBORS=11

BOX_COLOR = (255, 0, 0)
BOX_THICKNESS = 4

IOU_THRESHOLD=0.3
MAX_MISSED_FRAMES=80
CONFIRMATION_HITS=5


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
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
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
    if new_prediction == "Unknown":
        return current_prediction

    prediction_history.append(new_prediction)
    most_common_person, votes = Counter(prediction_history).most_common(1)[0]

    if votes >= VOTES_REQUIRED:
        return most_common_person

    return current_prediction


def recognize_face(cropped_face):
    matches = find_matches(cropped_face, database=DATABASE)
    return identify_face(matches) or "Unknown"


def crop_face(frame, face):
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = face
    padding_x = int(width * FACE_CROP_PADDING)
    padding_y = int(height * FACE_CROP_PADDING)

    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(frame_width, x + width + padding_x)
    bottom = min(frame_height, y + height + padding_y)

    return frame[top:bottom, left:right]


def play_video(video_path, output_path):
    tracks={}
    next_track_id=1
    capture = cv.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    writer = None

    try:
        face_cascade = load_face_cascade()
        prepare_face_recognition(DATABASE)
        writer = create_video_writer(capture, output_path)

        # predicted_person = "Unknown"
        # prediction_history = deque(maxlen=VOTE_HISTORY_SIZE)
        recognition_job = None
        recognition_track_id=None
        frame_number = 0

        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_number += 1

                # jobs completed,getting result
                if recognition_job is not None and recognition_job.done():
                    new_prediction = recognition_job.result()
                    track=tracks.get(recognition_track_id)

                    if track is not None:
                        previous_name = track.name
                        track.name = vote_for_prediction(
                            new_prediction,
                            track.prediction_history,
                            track.name
                        )
                        if track.name != previous_name:
                            print(
                                f"Track {track.track_id}: "
                                f"{previous_name} -> {track.name}"
                            )
                    else:
                        print(
                            f"Discarded result for expired track "
                            f"{recognition_track_id}"
                        )
                    recognition_job = None
                    recognition_track_id=None

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
                        if best_track.consecutive_hits>=CONFIRMATION_HITS:
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

                # starting jobs
                if recognition_job is None:
                    eligible_tracks = [
                        track
                        for track in tracks.values()
                        if track.confirmed
                        and track.missed_frames == 0
                        and frame_number - track.last_recognition_frame
                        >= RECOGNITION_INTERVAL_FRAMES
                    ]

                    if eligible_tracks:
                        track = min(
                            eligible_tracks,
                            key=lambda candidate: (
                                candidate.recognition_attempts,
                                candidate.track_id,
                            ),
                        )

                        face_crop = crop_face(frame, track.box)
                        recognition_job = executor.submit(
                            recognize_face,
                            face_crop.copy(),
                        )
                        recognition_track_id = track.track_id
                        track.recognition_attempts += 1
                        track.last_recognition_frame = frame_number

                for track in tracks.values():
                    if not track.confirmed or track.missed_frames>0:
                        continue
                    draw_faces(frame,[track.box])
                    draw_name(frame,track.box, f"{track.name}")
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
