import os
import cv2
import numpy as np
import mediapipe as mp
import requests
import json
import threading
import time

# ============================================================
#                WATSONX CONFIG
# ============================================================

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_REGION = "us-south"

if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
    raise RuntimeError("❌ ERROR: Set WATSONX_API_KEY and WATSONX_PROJECT_ID before running.")

WATSONX_URL = (
    f"https://{WATSONX_REGION}.ml.cloud.ibm.com/"
    "ml/v1/text/generation?version=2023-05-29"
)

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {WATSONX_API_KEY}"
}

# ============================================================
#                     WATSONX FUNCTION
# ============================================================

def call_watsonx(prompt):
    data = {
        "model_id": "ibm/granite-20b-multilingual",
        "input": prompt,
        "project_id": WATSONX_PROJECT_ID,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 60
        }
    }
    try:
        r = requests.post(WATSONX_URL, headers=HEADERS, json=data)
        r.raise_for_status()
        return r.json()["results"][0]["generated_text"]
    except Exception as e:
        print("Watsonx error:", e)
        return "Error analyzing emotion."

# ============================================================
#              MEDIAPIPE FACE DETECTOR + MESH
# ============================================================

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

KEY_LANDMARKS = list(range(0, 468, 12))  # clean mesh

# Watsonx state
last_emotion = "neutral"
last_stress = "medium"
last_request_time = 0

# ============================================================
#               WATSONX THREAD HANDLER
# ============================================================

def watsonx_thread(features):
    global last_emotion, last_stress

    prompt = f"""
Analyze this facial landmark data:
{features}

Return:
Emotion: <emotion>
Stress: <low/medium/high>
"""

    result = call_watsonx(prompt)

    if "Emotion" in result:
        try:
            lines = result.split("\n")
            last_emotion = lines[0].split(":")[1].strip()
            last_stress = lines[1].split(":")[1].strip()
        except:
            pass

# ============================================================
#                     VIDEO LOOP
# ============================================================

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

last_face_seen = time.time()

print("🚀 MirrorMind V4 Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # ---------------------------
    # Face Detection (Primary)
    # ---------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = face_detector.process(rgb)

    if not det.detections:
        cv2.putText(display, "No face detected…", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # auto-exit after 3 seconds
        if time.time() - last_face_seen > 3:
            print("No face for 3 seconds → exiting.")
            break

        cv2.imshow("MirrorMind V4", display)
        if cv2.waitKey(1) == 27:
            break
        continue

    last_face_seen = time.time()

    # ---------------------------
    # FaceMesh (Only after detection)
    # ---------------------------
    mesh_results = face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks:
        lm = mesh_results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        features = []
        for idx in KEY_LANDMARKS:
            p = lm.landmark[idx]
            x, y = int(p.x * w), int(p.y * h)
            features.append((p.x, p.y))
            cv2.circle(display, (x, y), 2, (0, 255, 0), -1)

        # Watsonx call every 4 seconds (safe)
        if time.time() - last_request_time > 4:
            threading.Thread(target=watsonx_thread, args=(str(features),), daemon=True).start()
            last_request_time = time.time()

    # ---------------------------
    # Overlay emotion result
    # ---------------------------
    cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(display, f"Emotion: {last_emotion}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display, f"Stress: {last_stress}", (350, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)

    cv2.imshow("MirrorMind V4", display)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
