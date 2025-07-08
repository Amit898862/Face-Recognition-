import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
import os
import datetime

# ---------- Helper Functions ----------

def save_user_info(user_id, user_name):
    if not os.path.exists("user_data.csv"):
        with open("user_data.csv", "w") as f:
            f.write("ID,Name\n")
    with open("user_data.csv", "a") as f:
        f.write(f"{user_id},{user_name}\n")

def get_user_name(user_id):
    if not os.path.exists("user_data.csv"):
        return "Unknown"
    with open("user_data.csv", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            uid, name = line.strip().split(',')
            if int(uid) == user_id:
                return name
    return "Unknown"

def mark_attendance(user_id):
    name = get_user_name(user_id)
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("ID,Name,Date,Time\n")
    already_marked = False
    with open("attendance.csv", "r") as f:
        for line in f.readlines():
            if line.startswith(f"{user_id},{name},{date}"):
                already_marked = True
                break
    if not already_marked:
        with open("attendance.csv", "a") as f:
            f.write(f"{user_id},{name},{date},{time}\n")

# ---------- GUI Setup ----------

window = Tk()
window.title("Face Recognition System")
window.geometry("400x450")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

Label(window, text="Enter User ID:", font=("Arial", 12)).pack(pady=5)
entry_id = Entry(window, font=("Arial", 12))
entry_id.pack(pady=5)

Label(window, text="Enter Name:", font=("Arial", 12)).pack(pady=5)
entry_name = Entry(window, font=("Arial", 12))
entry_name.pack(pady=5)

status_label = Label(window, text="", font=("Arial", 10), fg="red")
status_label.pack(pady=10)

# ---------- Face Capture ----------

def capture_face():
    user_id = entry_id.get()
    user_name = entry_name.get()
    if not user_id or not user_name:
        status_label.config(text="Enter both ID and Name")
        return
    save_user_info(user_id, user_name)
    cam = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow('Capturing Face - Press Q to Exit', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break
    cam.release()
    cv2.destroyAllWindows()
    status_label.config(text=f"Captured {count} images for {user_name}")

# ---------- Train Model ----------

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    path = 'dataset'
    for image_name in os.listdir(path):
        img_path = os.path.join(path, image_name)
        gray_img = Image.open(img_path).convert('L')
        np_img = np.array(gray_img, 'uint8')
        id = int(image_name.split('.')[1])
        faces.append(np_img)
        ids.append(id)
    recognizer.train(faces, np.array(ids))
    if not os.path.exists('trainer'):
        os.makedirs('trainer')
    recognizer.save('trainer/trainer.yml')
    status_label.config(text="Training Completed")

# ---------- Face Recognition ----------

def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 100:
                name = get_user_name(id)
                label = f"{name} ({round(100 - confidence)}%)"
                mark_attendance(id)
            else:
                label = "Unknown"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow('Face Recognition - Press Q to Exit', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

# ---------- GUI Buttons ----------

Button(window, text="Capture Face", command=capture_face, bg="blue", fg="white", font=("Arial", 12)).pack(pady=10)
Button(window, text="Train Model", command=train_model, bg="green", fg="white", font=("Arial", 12)).pack(pady=10)
Button(window, text="Recognize Face", command=recognize_face, bg="purple", fg="white", font=("Arial", 12)).pack(pady=10)

window.mainloop()
