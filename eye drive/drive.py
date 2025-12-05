import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import wave
import struct

alert_sound = r"d:\DL_final\alert.wav"

if not os.path.exists(alert_sound):
    sample_rate = 44100
    duration = 1.0
    frequency = 1000
    samples = int(sample_rate * duration)
    wav = wave.open(alert_sound, "w")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    for i in range(samples):
        value = int(32767 * np.sin(2 * np.pi * frequency * i / sample_rate))
        wav.writeframes(struct.pack("<h", value))
    wav.close()

mp_face_mesh = mp.solutions.face_mesh
mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT = [362, 385, 387, 263, 373, 380]
RIGHT = [33, 160, 158, 133, 153, 144]

pygame.mixer.init()
beep = pygame.mixer.Sound(alert_sound)

def ear(pts):
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2 * C)

cap = cv2.VideoCapture(0)

THRESH = 0.25
LIMIT = 20
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if result.multi_face_landmarks:
        for lm in result.multi_face_landmarks:
            left = [(lm.landmark[i].x * frame.shape[1], lm.landmark[i].y * frame.shape[0]) for i in LEFT]
            right = [(lm.landmark[i].x * frame.shape[1], lm.landmark[i].y * frame.shape[0]) for i in RIGHT]

            l_ear = ear(left)
            r_ear = ear(right)
            avg = (l_ear + r_ear) / 2

            for p in left + right:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

            if avg < THRESH:
                counter += 1
                if counter >= LIMIT:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    beep.play()
                    time.sleep(1)
            else:
                counter = 0

            cv2.putText(frame, f"EAR: {avg:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
