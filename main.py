import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import os
import pyttsx3
import time

# ---------------------------
# Initialize OpenAI Client
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# Initialize Text-to-Speech
# ---------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 175)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ---------------------------
# Mediapipe Initializations
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# ---------------------------
# AI Emotion Analysis Function
# ---------------------------
def analyze_emotion(face_features):
    try:
        prompt = (
            "You are an expert emotion & stress analyzer. "
            "Based on numeric facial landmark features, estimate:\n"
            "- Emotion (happy, sad, neutral, angry, fear, surprise)\n"
            "- Stress level (low, medium, high)\n\n"
            f"Facial feature data: {face_features}\n"
            "Respond in a short sentence."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You analyze emotions & stress from face geometry."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print("AI Error:", e)
        return "I could not analyze emotion."

# ---------------------------
# Extract Custom Facial Metrics
# ---------------------------
def extract_features(landmarks):
    # Example features (distances between key landmark points)
    def dist(p1, p2):
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

    left_eye = dist(landmarks[33], landmarks[160])
    right_eye = dist(landmarks[263], landmarks[387])
    mouth_open = dist(landmarks[13], landmarks[14])
    eyebrow_raise = dist(landmarks[70], landmarks[107])

    return {
        "left_eye": float(left_eye),
        "right_eye": float(right_eye),
        "mouth_open": float(mouth_open),
        "eyebrow_raise": float(eyebrow_raise)
    }

# ---------------------------
# Main Loop
# ---------------------------
def main_loop():
    cap = cv2.VideoCapture(0)

    last_ai_call_time = 0
    ai_interval = 4   # AI runs every 4 seconds

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera not detected.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    lm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    drawing_spec,
                    drawing_spec
                )

                # AI Call (once every 4 seconds)
                if time.time() - last_ai_call_time > ai_interval:
                    last_ai_call_time = time.time()

                    features = extract_features(lm.landmark)
                    print("Extracted features:", features)

                    result = analyze_emotion(features)
                    print("AI Response:", result)

                    speak(result)

            cv2.imshow("MirrorMind AI - Emotion & Stress Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------
# Run program
# ---------------------------
if __name__ == "__main__":
    print("Starting MirrorMind AI...")
    main_loop()
