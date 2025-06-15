# 📸 Attendance System with Live Location & Anti-Spoofing

A secure and intelligent attendance system built using Python and Streamlit that ensures proxy-free attendance through facial recognition, live location validation, and anti-spoofing technology. This project verifies student presence only during scheduled classes and logs attendance with subject-wise tracking.

---

## 🚀 Features

- 🔒 **Anti-Spoofing Detection** using deep learning to prevent photo/video-based proxy.
- 🌍 **Live Location Validation** to ensure student is within college premises.
- 📅 **Subject-wise Attendance Logging** based on actual class schedule.
- 🧠 **Face Recognition** using SVM with Dlib & FaceNet encodings.
- 📸 **Real-time Camera Capture** for face verification.
- 🕒 **Time-based Control** — only marks attendance during scheduled classes.
- 💾 **MySQL Integration** for storing attendance logs securely.

---

## 📂 Project Structure

```
📦 Attendance-System
│── anti_spoofing_model_xception.h5
│── known_face_encodings.pkl
│── known_face_labels.pkl
│── face_recognition_svm.pkl
│── schedule.csv
│── shape_predictor_68_face_landmarks.dat
├── 📄 app.py
├── 📄 README.md
```

---

## 🛠️ Installation Guide

> ⚠️ Make sure Python 3.8+ is installed on your system.

1. **Clone the repository**

```bash
git clone https://github.com/your-username/attendance-system.git
cd attendance-system
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

3. **Setup MySQL Database**

- Create a database named `attendance_system`.
- Import the following schema:

```sql
CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    roll_number VARCHAR(20),
    subject VARCHAR(50),
    status VARCHAR(10),
    timestamp DATETIME
);
```

4. **Environment Variables**

Create a `.env` file and add:

```env
DB_HOST=localhost
DB_USER= your use name
DB_PASSWORD= your password
DB_NAME=attendance_system
```

---

## ▶️ Running the App
```bash
python train_anti_spoofing.py
python train_face_recognition.py
```
```bash
streamlit run app.py
```

> Make sure you allow **camera** and **location access** in your browser.

---

## ✅ Setup Checklist

- [x] All model files placed in `/models`
- [x] Face data trained and encodings stored
- [x] Schedule CSV formatted with `Day`, `Start_Time`, `End_Time`, `Subject`
- [x] Database connected and verified
- [x] `shape_predictor_68_face_landmarks.dat` downloaded in `utils/`

---

## 📌 Technologies Used

- Streamlit
- OpenCV
- Dlib
- TensorFlow / Keras
- Face Recognition
- MySQL
- Pandas
- dotenv
- Geolocation API via `streamlit_js_eval`

---

## Done By

- [A BAVESHABIJITH](https://github.com/baveshabijith07)
- [Thirupathi S](https://github.com/thirupathi-01)
---
