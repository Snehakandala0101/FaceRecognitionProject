import cv2
import os

# Ask for user ID
user_id = input("Enter user ID (e.g., 1): ")
user_name = input("Enter user name (optional): ")

# Create dataset folder if not exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start video capture
cam = cv2.VideoCapture(0)

print("\n[INFO] Initializing face capture. Look at the camera...")

count = 0
while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]

        # Save the captured face in dataset folder
        file_path = os.path.join(dataset_path, f"User.{user_id}.{count}.jpg")
        cv2.imwrite(file_path, face_img)

        # Show image with rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f"Sample {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Collecting Faces", img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # Press 'ESC' to quit
        break
    elif count >= 30:  # Take 30 face samples then stop
        break

# Cleanup
print(f"\n[INFO] Collected {count} face samples for User {user_id} ({user_name})")
cam.release()
cv2.destroyAllWindows()
