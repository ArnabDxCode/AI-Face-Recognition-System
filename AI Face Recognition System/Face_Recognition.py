import cv2
import pickle
import numpy as np
import time
import pyttsx3
import csv
import os
from datetime import datetime

print("🚀 Starting Cyber Face Recognition System...")

# ===============================
# TEXT TO SPEECH
# ===============================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===============================
# LOAD MODEL
# ===============================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("labels.pickle", "rb") as f:
    original_labels = pickle.load(f)
    labels = {v: k for k, v in original_labels.items()}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

# ===============================
# VARIABLES
# ===============================
start_time = time.time()
scanning_duration = 3
recognized_names = set()

prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
smooth_factor = 0.7  # Higher = smoother

# Attendance file
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# ===============================
# MAIN LOOP
# ===============================
while True:
    frame_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    elapsed_time = time.time() - start_time

    # ===============================
    # DARK CYBER HUD OVERLAY
    # ===============================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    alpha_overlay = 0.35
    frame = cv2.addWeighted(overlay, alpha_overlay, frame, 1 - alpha_overlay, 0)

    # ===============================
    # SMOOTH FADE-IN SCANNING
    # ===============================
    if elapsed_time < scanning_duration:
        fade_alpha = elapsed_time / scanning_duration

        cv2.putText(frame, "SCANNING...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, int(255 * fade_alpha), 255), 3)

        height = frame.shape[0]
        line_y = int((elapsed_time / scanning_duration) * height)
        cv2.line(frame, (0, line_y),
                 (frame.shape[1], line_y),
                 (0, int(255 * fade_alpha), 255), 2)

    else:
        for (x, y, w, h) in faces:

            # ===============================
            # BOX STABILIZATION (SMOOTHING)
            # ===============================
            x = int(prev_x * smooth_factor + x * (1 - smooth_factor))
            y = int(prev_y * smooth_factor + y * (1 - smooth_factor))
            w = int(prev_w * smooth_factor + w * (1 - smooth_factor))
            h = int(prev_h * smooth_factor + h * (1 - smooth_factor))

            prev_x, prev_y, prev_w, prev_h = x, y, w, h

            roi_gray = gray[y:y+h, x:x+w]
            id_, confidence = recognizer.predict(roi_gray)
            confidence_percent = round(100 - confidence)

            if confidence < 80:
                name = labels[id_]
                color = (0, 255, 0)
                label_text = f"{name} ({confidence_percent}%)"

                if name not in recognized_names:
                    speak(f"Welcome {name}")
                    recognized_names.add(name)

                    with open("attendance.csv", "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
            else:
                color = (0, 0, 255)
                label_text = f"UNKNOWN ({confidence_percent}%)"

            # ===============================
            # CYBER CORNER BOX
            # ===============================
            thickness = 2
            line_length = 30

            # Top-left
            cv2.line(frame, (x, y), (x+line_length, y), color, thickness)
            cv2.line(frame, (x, y), (x, y+line_length), color, thickness)

            # Top-right
            cv2.line(frame, (x+w, y), (x+w-line_length, y), color, thickness)
            cv2.line(frame, (x+w, y), (x+w, y+line_length), color, thickness)

            # Bottom-left
            cv2.line(frame, (x, y+h), (x+line_length, y+h), color, thickness)
            cv2.line(frame, (x, y+h), (x, y+h-line_length), color, thickness)

            # Bottom-right
            cv2.line(frame, (x+w, y+h), (x+w-line_length, y+h), color, thickness)
            cv2.line(frame, (x+w, y+h), (x+w, y+h-line_length), color, thickness)

            # Label background
            cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 0, 0), -1)

            # Label text
            cv2.putText(frame, label_text, (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ===============================
    # FPS COUNTER
    # ===============================
    fps = 1 / (time.time() - frame_start)
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    cv2.imshow("Cyber Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()