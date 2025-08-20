import streamlit as st
import face_recognition
import cv2
import numpy as np
import pickle
import mysql.connector
import time
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import dlib
from imutils import face_utils
import os
from dotenv import load_dotenv
from streamlit_js_eval import get_geolocation

# Load environment variables
load_dotenv()

# College Location (Latitude, Longitude)
COLLEGE_LAT = 11.0772746
COLLEGE_LON = 76.9897629
LOCATION_TOLERANCE = 0.001

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root1234",
    "database": "attendance_system",
}

# Load class schedule
def load_schedule():
    return pd.read_csv("schedule.csv")

schedule_df = load_schedule()

# Load trained models
try:
    with open("face_recognition_svm.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("known_face_encodings.pkl", "rb") as f:
        known_encodings = pickle.load(f)
    with open("known_face_labels.pkl", "rb") as f:
        known_labels = pickle.load(f)
    
    spoof_model = tf.keras.models.load_model("anti_spoofing_model_xception.h5")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Load dlib face detector and landmark predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    st.error("❌ Missing 'shape_predictor_68_face_landmarks.dat'. Please download it.")
    st.stop()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

TOLERANCE = 0.35

def min_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_real_face(frame):
    frame_resized = cv2.resize(frame, (128, 128))
    frame_normalized = frame_resized.astype("float32") / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    prediction = spoof_model.predict(frame_expanded)[0][0]
    
    blur_value = min_blur(frame)
    return prediction > 0.7 and blur_value > 30

def is_within_college(lat, lon):
    return abs(lat - COLLEGE_LAT) <= LOCATION_TOLERANCE and abs(lon - COLLEGE_LON) <= LOCATION_TOLERANCE

def get_current_subject():
    now = datetime.now()
    day = now.strftime("%A")
    current_time = now.strftime("%H:%M")
    
    for _, row in schedule_df.iterrows():
        if row["Day"].strip().lower() == day.lower():
            start_time = datetime.strptime(row["Start_Time"], "%H:%M").time()
            end_time = datetime.strptime(row["End_Time"], "%H:%M").time()
            current_dt = datetime.strptime(current_time, "%H:%M").time()
            
            if start_time <= current_dt <= end_time:
                return row["Subject"].strip() if row["Subject"].strip().lower() != "free" else None
    return None

def insert_attendance(roll_number, subject, status):
    if subject is None:
        st.error("❌ Attendance cannot be marked during free periods or outside scheduled class hours!")
        return False
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if attendance was marked within the last hour
        query = """
        SELECT timestamp FROM attendance 
        WHERE roll_number = %s AND subject = %s 
        ORDER BY timestamp DESC LIMIT 1
        """
        cursor.execute(query, (roll_number, subject))
        last_record = cursor.fetchone()
        
        if last_record:
            last_attendance_time = last_record[0]
            if datetime.now() - last_attendance_time < timedelta(hours=1):
                st.error("❌ Attendance can only be marked once per hour!")
                cursor.close()
                conn.close()
                return False
        
        # Insert new attendance record
        query = "INSERT INTO attendance (roll_number, subject, status, timestamp) VALUES (%s, %s, %s, NOW())"
        cursor.execute(query, (roll_number, subject, status))
        conn.commit()
        
        cursor.close()
        conn.close()
        return True
    
    except mysql.connector.Error as err:
        st.error(f"❌ Database Error: {err}")
        return False

st.title("Attendance System with Live Location & Anti-Spoofing")

location = get_geolocation()
if location:
    lat, lon = location['coords']['latitude'], location['coords']['longitude']
    st.write(f"🌍 Your Location: {lat}, {lon}")
    
    if st.button("Capture & Verify Attendance"):
        cap = cv2.VideoCapture(0)
        time.sleep(0.5)

        if not cap.isOpened():
            st.error("Error: Could not access webcam.")
        else:
            ret, frame = cap.read()
            if ret:
                st.image(frame, channels="BGR", caption="Captured Image")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)

                if face_encodings:
                    recognized_roll = known_labels[np.argmax(cosine_similarity([face_encodings[0]], known_encodings))]
                    subject = get_current_subject()
                    
                    if recognized_roll:
                        if not is_real_face(frame):
                            st.error(f"❌ Spoofing detected! Marking {recognized_roll} as ABSENT.")
                            insert_attendance(recognized_roll, subject, "ABSENT")
                        elif not is_within_college(lat, lon):
                            st.error(f"❌ You are outside the college! Marking {recognized_roll} as ABSENT.")
                            insert_attendance(recognized_roll, subject, "ABSENT")
                        else:
                            if insert_attendance(recognized_roll, subject, "PRESENT"):
                                st.success(f"✅ Attendance marked for {recognized_roll} in {subject} as PRESENT.")
                    else:
                        st.error("❌ No matching face found.")
                else:
                    st.error("❌ No face detected in the image.")

            cap.release()
            cv2.destroyAllWindows()
else:
    st.warning("⚠️ Please allow location access in your browser.")
