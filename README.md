# Driving Eye – Drowsiness Detection

Driving Eye is a simple Python-based project that monitors a person’s eyes through the webcam and alerts them if they start getting drowsy. It uses MediaPipe Face Mesh to track eye landmarks, calculates the Eye Aspect Ratio (EAR), and plays a sound whenever the eyes stay closed for too long. The goal is to provide a basic, real-time safety check for drivers or anyone working long hours.

---

## What the project does

- Opens the webcam and detects your face  
- Tracks the key points around your eyes  
- Calculates EAR to understand if your eyes are open or closed  
- Shows your EAR value live on the screen  
- Displays a warning and plays a beep sound if you're getting sleepy  
- Generates the beep sound automatically if it's missing  

It runs smoothly on a normal computer and doesn’t require any special hardware.

---

## Project Files

Driving-Eye/
│
├── eye drive/
│ └── drive.py # Main detection script
│
├── alert.wav # Beep sound (created automatically if not found)
└── README.md

## How to install

Make sure Python is installed, then install the required libraries:

```bash
pip install opencv-python mediapipe numpy pygame

How to run

Go to the folder and run:

python "eye drive/drive.py"

Technologies used

MediaPipe Face Mesh

OpenCV

NumPy

Pygame (for the alert sound)
