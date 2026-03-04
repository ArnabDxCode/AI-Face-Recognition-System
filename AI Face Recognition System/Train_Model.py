import cv2
import os
import numpy as np
from PIL import Image
import pickle

print("🔄 Starting Training Process...\n")

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    print("❌ Dataset folder not found!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_ids = {}
current_id = 0
total_images = 0

# Walk through dataset folder
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"📂 Processing: {person_name}")

    if person_name not in label_ids:
        label_ids[person_name] = current_id
        current_id += 1

    id_ = label_ids[person_name]

    for image_name in os.listdir(person_path):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(person_path, image_name)

            try:
                img = Image.open(image_path).convert("L")
                img = img.resize((200, 200))  # Standard size
                img_np = np.array(img, "uint8")

                faces.append(img_np)
                labels.append(id_)
                total_images += 1

            except Exception as e:
                print(f"⚠ Skipping {image_name} due to error")

# Safety check
if len(faces) == 0:
    print("\n❌ No images found for training!")
    exit()

print("\n🧠 Training Model...")
recognizer.train(faces, np.array(labels))

recognizer.save("trainer.yml")

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

print("\n✅ Training Completed Successfully!")
print(f"👤 Total Persons: {len(label_ids)}")
print(f"🖼 Total Images Used: {total_images}")
print("💾 Model saved as trainer.yml")
print("📌 Labels saved as labels.pickle")