import cv2
import os
import time
import winsound
from tkinter import *
from PIL import Image, ImageTk

# ---------------- GUI SETUP ----------------

root = Tk()
root.title("ArnabDxCode - Ultimate Face Capture")
root.geometry("950x750")
root.configure(bg="#0D1B2A")

# Load Logo
logo_img = Image.open("logo.png")
logo_img = logo_img.resize((220, 130))
logo_photo = ImageTk.PhotoImage(logo_img)

Label(root, image=logo_photo, bg="#0D1B2A").pack(pady=10)

Label(root, text="Ultimate Face Capture System",
      font=("Arial", 22, "bold"),
      fg="#FFD60A", bg="#0D1B2A").pack()

# ---------------- DATASET ----------------

if not os.path.exists("dataset"):
    os.makedirs("dataset")

count = 0
current_name = ""
cap = None
auto_mode = False
stable_start = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- VIDEO FRAME ----------------

video_label = Label(root, bg="#1B263B")
video_label.pack(pady=20)

# ---------------- COUNTER ----------------

counter_label = Label(root,
                      text="Images Captured: 0",
                      font=("Arial", 14, "bold"),
                      fg="white", bg="#0D1B2A")
counter_label.pack()

# ---------------- NAME ENTRY ----------------

name_entry = Entry(root, font=("Arial", 14),
                   bg="#1B263B", fg="white",
                   insertbackground="white",
                   justify="center")
name_entry.pack(pady=5)
name_entry.insert(0, "")

# ---------------- CAMERA ----------------

def start_camera():
    global cap, current_name, count

    current_name = name_entry.get().strip()
    if current_name == "" or current_name == "Enter Name":
        return

    person_path = os.path.join("dataset", current_name)
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    count = len(os.listdir(person_path))
    counter_label.config(text=f"Images Captured: {count}")

    cap = cv2.VideoCapture(0)
    root.focus()
    show_frame()

def show_frame():
    global cap, auto_mode, stable_start

    if cap is not None:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

                # AUTO CAPTURE LOGIC
                if auto_mode:
                    if stable_start is None:
                        stable_start = time.time()
                    elif time.time() - stable_start > 2:
                        capture_face()
                        stable_start = None
                else:
                    stable_start = None
            else:
                stable_start = None

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            video_label.imgtk = img
            video_label.configure(image=img)

        video_label.after(10, show_frame)

# ---------------- CAPTURE ----------------

def capture_face():
    global cap, count

    if cap is None:
        return

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]

        count += 1

        # Add watermark
        cv2.putText(face, "ArnabDxCode",
                    (5, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

        file_path = f"dataset/{current_name}/{count}.jpg"
        cv2.imwrite(file_path, face)

        counter_label.config(text=f"Images Captured: {count}")

        # Capture sound
        winsound.Beep(1000, 150)

# ---------------- KEY BIND ----------------

def capture_key(event):
    capture_face()
    return "break"

# ---------------- AUTO MODE ----------------

def toggle_auto():
    global auto_mode
    auto_mode = not auto_mode
    if auto_mode:
        auto_btn.config(text="Auto Mode: ON")
    else:
        auto_btn.config(text="Auto Mode: OFF")

# ---------------- STOP ----------------

def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        video_label.config(image='')

# ---------------- BUTTONS ----------------

btn_frame = Frame(root, bg="#0D1B2A")
btn_frame.pack(pady=15)

Button(btn_frame, text="Start Camera",
       command=start_camera,
       font=("Arial", 12, "bold"),
       bg="#FFD60A", fg="black",
       width=15).grid(row=0, column=0, padx=10)

Button(btn_frame, text="Capture",
       command=capture_face,
       font=("Arial", 12, "bold"),
       bg="#FFD60A", fg="black",
       width=15).grid(row=0, column=1, padx=10)

auto_btn = Button(btn_frame, text="Auto Mode: OFF",
                  command=toggle_auto,
                  font=("Arial", 12, "bold"),
                  bg="#FFD60A", fg="black",
                  width=15)
auto_btn.grid(row=0, column=2, padx=10)

Button(btn_frame, text="Stop Camera",
       command=stop_camera,
       font=("Arial", 12, "bold"),
       bg="#FFD60A", fg="black",
       width=15).grid(row=0, column=3, padx=10)

# Key binding
root.bind('<c>', capture_key)
root.bind('<C>', capture_key)

root.mainloop()