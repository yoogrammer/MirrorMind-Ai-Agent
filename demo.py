
"""
main.py
Camera-based face mesh + facial metrics visualizer.
AI is fully disabled for testing.
"""

import os
import time
import threading
import json
from queue import Queue, Empty

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from PIL import Image

# --- AI DISABLED ---
def ask_ai_for_emotion_and_feedback(feature_summary):
    return {
        "emotion": "neutral",
        "stress_level": "low",
        "confidence": 0.0,
        "feedback": "AI disabled — showing only camera & face detection."
    }

# No API key needed
OPENAI_KEY = None

# --- Camera settings ---
CAMERA_INDEX = 0
NO_FACE_TIMEOUT = 10.0
FEEDBACK_COOLDOWN = 6.0
LANDMARK_HISTORY_SECONDS = 2.0

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# --- TTS (still works but not used by AI now) ---
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 165)
tts_engine.setProperty("volume", 1.0)

# --- Helper functions ---
def landmarks_to_np(landmarks, shape):
    h, w = shape[:2]
    pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks], dtype=np.int32)
    return pts

def mouth_aspect_ratio(pts):
    try:
        top = pts[13]
        bottom = pts[14]
        left = pts[61]
        right = pts[291]
        vert = np.linalg.norm(top - bottom)
        hor = np.linalg.norm(left - right) + 1e-6
        return float(vert / hor)
    except Exception:
        return 0.0

def eye_aspect_ratio(pts, left=True):
    try:
        if left:
            idx = [33, 159]
            corners = [33, 133]
        else:
            idx = [362, 386]
            corners = [362, 263]

        top = pts[idx[0]]
        bottom = pts[idx[1]]
        leftc = pts[corners[0]]
        rightc = pts[corners[1]]

        vert = np.linalg.norm(top - bottom)
        hor = np.linalg.norm(leftc - rightc) + 1e-6
        return float(vert / hor)
    except Exception:
        return 0.0

def brow_raise_metric(pts):
    try:
        left_brow = pts[105]
        left_eye_top = pts[159]
        right_brow = pts[334]
        right_eye_top = pts[386]
        h = np.linalg.norm(pts[10] - pts[152]) + 1e-6
        dist = (np.linalg.norm(left_brow - left_eye_top) + np.linalg.norm(right_brow - right_eye_top)) / 2.0
        return float(dist / h)
    except Exception:
        return 0.0

def head_tilt_metric(pts):
    try:
        left = pts[234]
        right = pts[454]
        nose = pts[1]
        denom = np.linalg.norm(right - left) + 1e-6
        rel = (nose[0] - left[0]) / denom
        return float(rel)
    except Exception:
        return 0.5

def summarize_features(flist):
    arr = np.array(flist)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return {
        "mar_mean": float(mean[0]), "mar_std": float(std[0]),
        "earL_mean": float(mean[1]), "earL_std": float(std[1]),
        "earR_mean": float(mean[2]), "earR_std": float(std[2]),
        "brow_mean": float(mean[3]), "brow_std": float(std[3]),
        "head_rel_mean": float(mean[4]), "head_rel_std": float(std[4]),
        "samples": int(arr.shape[0])
    }

# --- Worker thread (AI disabled but keep structure) ---
ai_request_queue = Queue()
ai_response_queue = Queue()

def ai_worker():
    while True:
        task = ai_request_queue.get()
        if task is None:
            break
        result = ask_ai_for_emotion_and_feedback(task)
        ai_response_queue.put(result)

ai_thread = threading.Thread(target=ai_worker, daemon=True)
ai_thread.start()

# --- Main camera loop ---
def main_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    last_face_time = time.time()
    last_feedback_time = 0.0
    landmark_buffer = []
    last_ai_result = None

    if not cap.isOpened():
        print("Unable to open camera.")
        return

    window_name = "Emotion Detector (AI disabled)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not received.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        h, w = frame.shape[:2]
        face_present = False

        if results.multi_face_landmarks:
            face_present = True
            last_face_time = time.time()

            face_landmarks = results.multi_face_landmarks[0].landmark
            pts = landmarks_to_np(face_landmarks, frame.shape)

            mar = mouth_aspect_ratio(pts)
            earL = eye_aspect_ratio(pts, left=True)
            earR = eye_aspect_ratio(pts, left=False)
            brow = brow_raise_metric(pts)
            head_rel = head_tilt_metric(pts)

            ts = time.time()
            landmark_buffer.append((ts, (mar, earL, earR, brow, head_rel)))

            cutoff = ts - LANDMARK_HISTORY_SECONDS
            landmark_buffer = [(t, v) for (t, v) in landmark_buffer if t >= cutoff]

            mp_drawing.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=1),
            )

            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            y_min = np.min(pts[:, 1])
            y_max = np.max(pts[:, 1])
            cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (255, 200, 0), 2)

            now = time.time()
            if (now - last_feedback_time) > FEEDBACK_COOLDOWN and len(landmark_buffer) >= 5:
                flist = [v for (t, v) in landmark_buffer]
                summary = summarize_features(flist)
                ai_request_queue.put(summary)
                last_feedback_time = now

            cv2.putText(
                frame,
                f"MAR:{mar:.2f} EAR_L:{earL:.2f} EAR_R:{earR:.2f} BROW:{brow:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # AI result (dummy)
        try:
            ai_result = ai_response_queue.get_nowait()
            last_ai_result = ai_result
        except Empty:
            pass

        if last_ai_result:
            text_block = (
                f"Emotion: {last_ai_result['emotion']} | Stress: {last_ai_result['stress_level']}\n"
                f"{last_ai_result['feedback']}"
            )
            overlay_img = frame.copy()
            cv2.rectangle(overlay_img, (10, h - 110), (w - 10, h - 10), (0, 0, 0), -1)
            alpha = 0.45
            cv2.addWeighted(overlay_img, alpha, frame, 1 - alpha, 0, frame)

            y0 = h - 90
            for i, line in enumerate(text_block.split("\n")):
                cv2.putText(frame, line, (20, y0 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        if (time.time() - last_face_time) > NO_FACE_TIMEOUT:
            print("No face detected for 10 seconds. Exiting.")
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ai_request_queue.put(None)
    print("Program ended.")

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        print("Fatal error:", e)
