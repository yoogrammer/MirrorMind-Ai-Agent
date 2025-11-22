"""
main.py – Watsonx Version
AI-powered face-based emotion & stress detector with voice feedback.

Requirements:
    pip install opencv-python mediapipe requests pyttsx3 numpy pillow
"""

import os
import time
import threading
import json
from queue import Queue, Empty

import cv2
import mediapipe as mp
import numpy as np
import requests
import pyttsx3
from PIL import Image

# ============================================================
#  Watsonx Configuration
# ============================================================

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = "ibm/granite-13b-chat-v2"  # choose model

if not WATSONX_API_KEY or not PROJECT_ID:
    raise RuntimeError("Set WATSONX_API_KEY and WATSONX_PROJECT_ID before running.")

BASE_URL = "https://us-south.ml.cloud.ibm.com"  # update if your region differs

# ============================================================
#  Camera / Settings
# ============================================================

CAMERA_INDEX = 0
NO_FACE_TIMEOUT = 10.0
FEEDBACK_COOLDOWN = 6.0
LANDMARK_HISTORY_SECONDS = 2.0

# ============================================================
#  MediaPipe Setup
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ============================================================
#  TTS Setup
# ============================================================

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 165)
tts_engine.setProperty("volume", 1.0)

# ============================================================
#  Landmark Calculations
# ============================================================

def landmarks_to_np(landmarks, shape):
    h, w = shape[:2]
    return np.array([(int(l.x * w), int(l.y * h)) for l in landmarks], dtype=np.int32)

def mouth_aspect_ratio(pts):
    try:
        top, bottom = pts[13], pts[14]
        left, right = pts[61], pts[291]
        return float(np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6))
    except:
        return 0.0

def eye_aspect_ratio(pts, left=True):
    try:
        if left:
            t, b = pts[33], pts[159]
            l, r = pts[33], pts[133]
        else:
            t, b = pts[362], pts[386]
            l, r = pts[362], pts[263]
        return float(np.linalg.norm(t - b) / (np.linalg.norm(l - r) + 1e-6))
    except:
        return 0.0

def brow_raise_metric(pts):
    try:
        lb, le = pts[105], pts[159]
        rb, re = pts[334], pts[386]
        face_h = np.linalg.norm(pts[10] - pts[152]) + 1e-6
        dist = (np.linalg.norm(lb - le) + np.linalg.norm(rb - re)) / 2.0
        return float(dist / face_h)
    except:
        return 0.0

def head_tilt_metric(pts):
    try:
        left, right, nose = pts[234], pts[454], pts[1]
        return float((nose[0] - left[0]) / (np.linalg.norm(right - left) + 1e-6))
    except:
        return 0.5

def summarize_features(flist):
    arr = np.array(flist)
    m, s = np.mean(arr, axis=0), np.std(arr, axis=0)
    return {
        "mar_mean": float(m[0]), "mar_std": float(s[0]),
        "earL_mean": float(m[1]), "earL_std": float(s[1]),
        "earR_mean": float(m[2]), "earR_std": float(s[2]),
        "brow_mean": float(m[3]), "brow_std": float(s[3]),
        "head_rel_mean": float(m[4]), "head_rel_std": float(s[4]),
        "samples": len(arr),
    }

# ============================================================
#  Watsonx AI Function
# ============================================================

def ask_ai_for_emotion_and_feedback(feature_summary):

    url = f"{BASE_URL}/ml/v1/text/generate"

    headers = {
        "Authorization": f"Bearer {WATSONX_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are an empathetic assistant that analyzes facial-feature summary statistics "
        "and returns ONLY JSON with keys: emotion, stress_level, confidence, feedback."
    )

    payload = {
        "model_id": WATSONX_MODEL_ID,
        "project_id": PROJECT_ID,
        "input": f"""
System: {system_prompt}

User: Here are facial-feature statistics:
{json.dumps(feature_summary, indent=2)}

Return clean JSON only, example:
{{"emotion":"happy","stress_level":"low","confidence":0.92,"feedback":"You look relaxed!"}}
""",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.2
        }
    }

    try:
        r = requests.post(url, json=payload, headers=headers)
        r.raise_for_status()

        text = r.json()["results"][0]["generated_text"]

        # extract JSON
        import re
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    except Exception as e:
        print("Watsonx error:", e)
        return {
            "emotion": "neutral",
            "stress_level": "low",
            "confidence": 0.5,
            "feedback": "Unable to analyze clearly, try adjusting lighting."
        }

# ============================================================
#  Worker Thread
# ============================================================

ai_request_queue = Queue()
ai_response_queue = Queue()

def ai_worker():
    while True:
        task = ai_request_queue.get()
        if task is None:
            break
        result = ask_ai_for_emotion_and_feedback(task)
        ai_response_queue.put(result)

threading.Thread(target=ai_worker, daemon=True).start()

# ============================================================
#  Main Video Loop
# ============================================================

def main_loop():

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera not found!")
        return

    landmark_buffer = []
    last_face_time = time.time()
    last_feedback_time = 0
    last_ai_result = None

    window = "Watsonx Emotion & Stress Detector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 900, 700)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        h, w = frame.shape[:2]
        face_found = False

        if results.multi_face_landmarks:
            face_found = True
            last_face_time = time.time()

            pts = landmarks_to_np(results.multi_face_landmarks[0].landmark, frame.shape)

            mar = mouth_aspect_ratio(pts)
            earL = eye_aspect_ratio(pts, left=True)
            earR = eye_aspect_ratio(pts, left=False)
            brow = brow_raise_metric(pts)
            head_rel = head_tilt_metric(pts)

            ts = time.time()
            landmark_buffer.append((ts, (mar, earL, earR, brow, head_rel)))

            cutoff = ts - LANDMARK_HISTORY_SECONDS
            landmark_buffer = [(t, v) for (t, v) in landmark_buffer if t >= cutoff]

            mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0],
                                      mp_face_mesh.FACEMESH_TESSELATION)

            # bounding box
            x1, y1 = np.min(pts[:,0]), np.min(pts[:,1])
            x2, y2 = np.max(pts[:,0]), np.max(pts[:,1])
            cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (255,200,0), 2)

            # request AI feedback
            if (ts - last_feedback_time) > FEEDBACK_COOLDOWN and len(landmark_buffer) >= 5:
                summary = summarize_features([v for (_, v) in landmark_buffer])
                ai_request_queue.put(summary)
                last_feedback_time = ts

        # receive AI output
        try:
            ai_result = ai_response_queue.get_nowait()
            last_ai_result = ai_result

            threading.Thread(
                target=lambda t: (tts_engine.say(t), tts_engine.runAndWait()),
                args=(ai_result.get("feedback", ""),),
                daemon=True
            ).start()
        except Empty:
            pass

        # overlay AI result
        if last_ai_result:
            box = f"Emotion: {last_ai_result['emotion']} | Stress: {last_ai_result['stress_level']}"
            fb = last_ai_result['feedback']

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, h-110), (w-10, h-10), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

            cv2.putText(frame, box, (20, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, fb, (20, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        cv2.imshow(window, frame)

        if time.time() - last_face_time > NO_FACE_TIMEOUT:
            print("No face detected. Exiting.")
            break

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ai_request_queue.put(None)
    print("Program Ended.")

# ============================================================
#  Run
# ============================================================

if __name__ == "__main__":
    main_loop()
