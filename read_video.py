import cv2 as cv
capture=cv.VideoCapture("video.MOV")
if not capture.isOpened():
    print("Unable to open video")
    exit(0)

while True:
    res,frame=capture.read()

    if not res:
        break

    cv.imshow("Celebrity recognition", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()