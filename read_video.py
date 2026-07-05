import cv2 as cv
capture=cv.VideoCapture("face_vid.mp4")
if not capture.isOpened():
    print("Unable to open video")
    exit(0)

def detect(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_gray = cv.equalizeHist(frame_gray)
 
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.05, minNeighbors=5, minSize=(70,70))
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),4)       
 
    cv.imshow('Capture - Face detection', frame)

face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
while True:
    res,frame=capture.read()

    if not res:
        break

    detect(frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()