# FACE RECOGNITION PROJECT
**Real-Time Face Detection & Recognition using OpenCV**

---

##Overview
This project implements a **real-time face recognition system** using Python and OpenCV. It uses **Haar Cascade Classifier** for face detection and **LBPH (Local Binary Patterns Histograms)** Face Recognizer for training and recognition. The system can detect faces from webcam input, collect datasets, train a model, and perform recognition in real-time.

---

##Features
- Capture dataset of face images using webcam
- Train an LBPH model on the collected dataset
- Real-time face recognition via webcam
- Modular, simple, and beginner-friendly

---

##Algorithms & Technologies Used
- **Haar Cascade Classifier** -> For detecting faces in frames
- **LBPH (Local Binary Patterns Histograms)** -> For recognizing and matching faces
- **Python Libraries**:
  - OpenCV
  - NumPy
  - Pillow

---
##Workflow
1.**Create Dataset** – Capture 30 samples of the user’s face via webcam
2.**Train Model** – Train LBPH recognizer using the collected dataset
3.**Recognize Faces** – Detect and recognize faces in real-time

---

##Project Structure
FaceRecognitionProject/
- dataset/                                # Folder to store captured face images (currently empty)
- .gitignore
- 0_create_dataset.py          # Script to collect face dataset using webcam
- 1_train_model.py             # Script to train the recognizer and save trainer.yml
- 2_face_recognition.py        # Real-time face recognition script
- 3_haarcascade_frontalface_default.xml
- 4_trainer.yml                 # Trained model file (generated after training)
- 5_requirements.txt            # Dependencies
- README.md

>**Note:** The 'dataset/' folder is empty and is not uploaded to GitHub. Users need to create their own dataset using the provided script.

---


##Installation
1. Clone this repository:
git clone https://github.com/YourUsername/FaceRecognitionProject.git
cd FaceRecognitionProject

2. Install required dependencies:
pip install -r requirements.txt

---

##Usage
1. Collect Dataset
python 0_create_dataset.py
- Enter a numeric user ID and name
- Look at the camera until 30 samples are captured in the dataset/ folder

2. Train Model
python 1_train_model.py
- Trains LBPH model and generates trainer.yml

3. Run Real-Time Recognition
python 2_face_recognition.py
- Detects and recognizes known faces from webcam input
- Shows User ID / Name and confidence score

Sample Output
- Faces detected are highlighted with rectangles
- Recognized faces show User ID / Name and confidence score

Notes for GitHub Users
- The dataset/ folder is empty to begin with. Users must collect their own face images.
- The .gitignore file ensures temporary files, __pycache__, and other unnecessary files are not uploaded.

Future Improvements
- Add deep learning-based recognition (FaceNet, Dlib)
- Multi-face recognition and tracking
- GUI for easier use
- Database integration for storing face data

Author
Kandala Sneha
Email: kandalasneha2411@gmail.com
