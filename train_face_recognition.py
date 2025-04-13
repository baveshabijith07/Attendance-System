import os
import face_recognition
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

KNOWN_FACES_DIR = "known_faces"
MODEL = "hog"  # 'cnn' for accuracy, 'hog' for speed
OUTPUT_MODEL = "face_recognition_svm.pkl"
OUTPUT_ENCODINGS = "known_face_encodings.pkl"
OUTPUT_LABELS = "known_face_labels.pkl"

face_encodings = []
face_labels = []

print("🔍 Scanning known faces...")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        roll_number = os.path.splitext(filename)[0]  # Extract roll number

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image, model=MODEL)

        if len(encodings) > 1:
            print(f"⚠️ Multiple faces detected in {filename}. Use a clearer image.")
        elif encodings:
            face_encodings.append(encodings[0])
            face_labels.append(roll_number)
            print(f"✅ Face encoding added for {filename}")
        else:
            print(f"⚠️ No face encoding found for {filename}! Check image quality.")

if len(face_encodings) == 0:
    print("❌ No valid face encodings found! Please check your images.")
    exit()

# Convert labels to numerical values
label_encoder = LabelEncoder()
face_labels_encoded = label_encoder.fit_transform(face_labels)

# Save encodings for faster access
with open(OUTPUT_ENCODINGS, "wb") as f:
    pickle.dump(face_encodings, f)
with open(OUTPUT_LABELS, "wb") as f:
    pickle.dump(face_labels, f)

# Train SVM Model with balanced class weights
X_train, X_test, y_train, y_test = train_test_split(face_encodings, face_labels_encoded, test_size=0.2, random_state=42)

model = SVC(kernel="linear", probability=True, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print(f"🎯 Training complete! Model accuracy: {accuracy * 100:.2f}%")
if accuracy < 30:
    print("⚠️ WARNING: Accuracy is low! Improve dataset quality.")

# Save model
with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved as {OUTPUT_MODEL}")
print(f"✅ Encodings saved as {OUTPUT_ENCODINGS}")
print(f"✅ Label encoder saved as {OUTPUT_LABELS}")
