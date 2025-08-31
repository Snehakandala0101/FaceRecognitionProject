import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 60:
            label = f"User {id} ({round(100 - confidence)}%)"
        else:
            label = "Unknown"

        cv2.putText(img, label, (x, y-10), font, 0.9, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
