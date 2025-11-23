# """
# mirrormind_voice_c_continuous.py

# Offline MirrorMind with continuous voice feedback:
#  - MediaPipe FaceMesh
#  - Rule-based emotion detection (voting)
#  - Non-blocking pyttsx3 TTS
#  - ALWAYS speaks full feedback continuously while face present
#  - Prints and displays full feedback on screen
#  - Auto camera detect, auto-close after NO_FACE_TIMEOUT, close by window X
# """

# import cv2
# import time
# import random
# import threading
# import queue
# import numpy as np

# # -------- CONFIG --------
# MAX_CAMERA_INDEX = 4
# CAMERA_OPEN_TIMEOUT = 3.0
# NO_FACE_TIMEOUT = 5.0          # seconds without face -> exit
# SMOOTHING_WINDOW = 5
# USE_DSHOW = True               # on Windows prefer CAP_DSHOW
# VERBOSE = False                # set True to print feature details

# # Test image (uploaded by you). Use by setting TEST_ON_IMAGE = True
# TEST_ON_IMAGE = False
# TEST_IMAGE_PATH = "/mnt/data/Screenshot 2025-11-22 202149.png"

# # -------- FEEDBACK PHRASES --------
# FEEDBACK = {
#     "happy": [
#         "You look happy. Keep smiling.",
#         "Nice smile! That suits you.",
#         "Good to see you smiling."
#     ],
#     "sad": [
#         "You seem sad. It's okay, take a deep breath.",
#         "You look a bit down. Consider a short break.",
#         "If you need, pause and take care of yourself."
#     ],
#     "surprised": [
#         "You look surprised! Something unexpected happened?",
#         "That expression shows surprise.",
#         "Wow — that looks surprising."
#     ],
#     "angry": [
#         "You seem angry. Try a slow, deep breath to calm down.",
#         "Take a moment to relax your shoulders and breathe.",
#         "Consider a short pause to cool down."
#     ],
#     "tired": [
#         "You look tired. A short rest might help.",
#         "Maybe blink and take a quick break.",
#         "Your eyes look weary — consider resting them."
#     ],
#     "confused": [
#         "You might be confused. Take your time to think it through.",
#         "You seem uncertain. Asking a question can help.",
#         "If unsure, pause and reflect for a moment."
#     ],
#     "neutral": [
#         "You look calm and neutral.",
#         "A peaceful expression — all looks fine.",
#         "You appear composed."
#     ],
# }

# # -------- Non-blocking TTS worker --------
# try:
#     import pyttsx3
#     TTS_AVAILABLE = True
# except Exception:
#     pyttsx3 = None
#     TTS_AVAILABLE = False

# class TTSWorker:
#     def __init__(self):
#         self.enabled = TTS_AVAILABLE
#         self.q = queue.Queue()
#         self.thread = None
#         self.running = False
#         if self.enabled:
#             self.engine = pyttsx3.init()
#             self.engine.setProperty("rate", 165)
#             self.engine.setProperty("volume", 1.0)

#     def start(self):
#         if not self.enabled or (self.thread and self.thread.is_alive()):
#             return
#         self.running = True
#         self.thread = threading.Thread(target=self._loop, daemon=True)
#         self.thread.start()

#     def _loop(self):
#         while self.running:
#             try:
#                 text = self.q.get(timeout=0.5)
#             except queue.Empty:
#                 continue
#             try:
#                 self.engine.say(text)
#                 self.engine.runAndWait()
#             except Exception as e:
#                 print("TTS error:", e)

#     def speak(self, text):
#         if not self.enabled:
#             print("[TTS unavailable] " + text)
#             return
#         try:
#             self.q.put_nowait(text)
#         except queue.Full:
#             pass

#     def stop(self):
#         self.running = False

# # -------- MediaPipe availability --------
# try:
#     import mediapipe as mp
#     MP_AVAILABLE = True
# except Exception as e:
#     mp = None
#     MP_AVAILABLE = False
#     print("ERROR: mediapipe not available. Install with: pip install mediapipe")

# # -------- Landmark indices --------
# LIP_TOP = 13
# LIP_BOTTOM = 14
# MOUTH_LEFT = 61
# MOUTH_RIGHT = 291
# EYE_L_TOP = 159
# EYE_L_BOTTOM = 145
# EYE_R_TOP = 386
# EYE_R_BOTTOM = 374
# BROW_L = 105
# BROW_R = 334
# CHEEK_L = 234
# CHEEK_R = 454
# NOSE_TOP = 1
# CHIN = 152

# def landmarks_to_pixel_array(landmarks, shape):
#     h, w = shape[:2]
#     pts = []
#     for lm in landmarks:
#         pts.append((int(lm.x * w), int(lm.y * h)))
#     return np.array(pts)

# # -------- Features --------
# def face_width(pts):
#     try:
#         return np.linalg.norm(pts[CHEEK_R] - pts[CHEEK_L])
#     except:
#         return float(np.max(pts[:,0]) - np.min(pts[:,0]) + 1e-6)

# def face_height(pts):
#     try:
#         return np.linalg.norm(pts[CHIN] - pts[NOSE_TOP])
#     except:
#         return float(np.max(pts[:,1]) - np.min(pts[:,1]) + 1e-6)

# def compute_features(pts):
#     fw = face_width(pts) + 1e-6
#     fh = face_height(pts) + 1e-6

#     mouth_w = np.linalg.norm(pts[MOUTH_RIGHT] - pts[MOUTH_LEFT])
#     mouth_open = np.linalg.norm(pts[LIP_TOP] - pts[LIP_BOTTOM])

#     lip_center_y = (pts[LIP_TOP][1] + pts[LIP_BOTTOM][1]) / 2.0
#     lift_left = lip_center_y - pts[MOUTH_LEFT][1]
#     lift_right = lip_center_y - pts[MOUTH_RIGHT][1]
#     lip_lift = (lift_left + lift_right) / 2.0

#     cheek_left = np.linalg.norm(pts[CHEEK_L] - pts[EYE_L_TOP])
#     cheek_right = np.linalg.norm(pts[CHEEK_R] - pts[EYE_R_TOP])
#     cheek_avg = (cheek_left + cheek_right) / 2.0

#     eye_l = np.linalg.norm(pts[EYE_L_TOP] - pts[EYE_L_BOTTOM])
#     eye_r = np.linalg.norm(pts[EYE_R_TOP] - pts[EYE_R_BOTTOM])
#     eye_avg = (eye_l + eye_r) / 2.0

#     brow_left = np.linalg.norm(pts[BROW_L] - pts[EYE_L_TOP])
#     brow_right = np.linalg.norm(pts[BROW_R] - pts[EYE_R_TOP])
#     brow_avg = (brow_left + brow_right) / 2.0

#     return {
#         "mouth_width_ratio": mouth_w / fw,
#         "mouth_open_ratio": mouth_open / fh,
#         "lip_lift_norm": lip_lift / fh * 100.0,
#         "cheek_norm": cheek_avg / fh,
#         "eye_ratio": eye_avg / fw,
#         "brow_ratio": brow_avg / fh,
#         "raw_mouth_w": mouth_w,
#         "raw_mouth_open": mouth_open,
#         "face_w": fw,
#         "face_h": fh
#     }

# # -------- Emotion Voting --------
# def vote_emotion(feats):
#     votes = {}
#     def add(e, w=1):
#         votes[e] = votes.get(e, 0) + w

#     mw = feats["mouth_width_ratio"]
#     mo = feats["mouth_open_ratio"]
#     lift = feats["lip_lift_norm"]
#     cheek = feats["cheek_norm"]
#     eye = feats["eye_ratio"]
#     brow = feats["brow_ratio"]

#     if mw > 0.32 and lift > 2.5 and cheek < 0.50:
#         add("happy", 2)
#     elif mw > 0.28 and lift > 1.6:
#         add("happy", 1)

#     if mo > 0.12 and eye > 0.045 and brow > 0.085:
#         add("surprised", 2)
#     elif mo > 0.10 and eye > 0.05:
#         add("surprised", 1)

#     if brow < 0.055 and eye < 0.03:
#         add("angry", 2)
#     elif brow < 0.06 and eye < 0.035:
#         add("angry", 1)

#     if lift < 1.0 and mw < 0.28 and eye > 0.03:
#         add("sad", 1)

#     if eye < 0.018:
#         add("tired", 2)
#     elif eye < 0.025:
#         add("tired", 1)

#     if mo > 0.06 and abs(brow - 0.08) > 0.03:
#         add("confused", 1)

#     if not votes:
#         add("neutral", 1)

#     priority = ["surprised", "happy", "angry", "sad", "tired", "confused", "neutral"]
#     max_vote = max(votes.values())
#     candidates = [k for k, v in votes.items() if v == max_vote]
#     for p in priority:
#         if p in candidates:
#             return p, votes, feats
#     return candidates[0], votes, feats

# # -------- Camera --------
# def try_open_camera(index, use_dshow=False, timeout=CAMERA_OPEN_TIMEOUT):
#     try:
#         cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if use_dshow else 0)
#     except:
#         return None
#     if not cap or not cap.isOpened():
#         return None
#     start = time.time()
#     while time.time() - start < timeout:
#         ret, frame = cap.read()
#         if ret and frame is not None:
#             return cap
#         time.sleep(0.08)
#     cap.release()
#     return None

# def find_camera(max_index=MAX_CAMERA_INDEX, use_dshow=USE_DSHOW):
#     for i in range(max_index + 1):
#         cap = try_open_camera(i, use_dshow)
#         if cap:
#             return cap, i
#     return None, None

# # -------- Main --------
# def main():
#     if not MP_AVAILABLE:
#         print("ERROR: mediapipe not available. Install with: pip install mediapipe")
#         return

#     tts = TTSWorker()
#     tts.start()

#     mp_face_mesh = mp.solutions.face_mesh
#     mp_drawing = mp.solutions.drawing_utils
#     face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
#                                       refine_landmarks=True, min_detection_confidence=0.5,
#                                       min_tracking_confidence=0.5)

#     # TEST IMAGE MODE
#     if TEST_ON_IMAGE:
#         img = cv2.imread(TEST_IMAGE_PATH)
#         if img is None:
#             print("ERROR: Test image not found:", TEST_IMAGE_PATH)
#             return
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         res = face_mesh.process(rgb)
#         if not res.multi_face_landmarks:
#             print("No face in test image.")
#             cv2.imshow("Test", img)
#             cv2.waitKey(0)
#             return
#         pts = landmarks_to_pixel_array(res.multi_face_landmarks[0].landmark, img.shape)
#         emotion, votes, feats = vote_emotion(compute_features(pts))
#         phrase = random.choice(FEEDBACK.get(emotion, FEEDBACK["neutral"]))
#         text = f"Emotion detected: {emotion}. {phrase}"
#         tts.speak(text)
#         print("Spoken (test image):", text)
#         mp_drawing.draw_landmarks(img, res.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
#                                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
#                                    mp_drawing.DrawingSpec(color=(0,128,255), thickness=1))
#         cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
#         cv2.imshow("Test", img)
#         cv2.waitKey(0)
#         return

#     # FIND CAMERA
#     cap, idx = find_camera()
#     if cap is None:
#         print("ERROR: No camera found.")
#         return

#     win = "MirrorMind Continuous Voice"
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(win, 960, 720)

#     last_face_time = time.time()
#     history = []

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 if time.time() - last_face_time > 2.0:
#                     print("ERROR: Camera frame stopped.")
#                     break
#                 time.sleep(0.02)
#                 continue

#             frame = cv2.flip(frame, 1)
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = face_mesh.process(rgb)

#             if res and res.multi_face_landmarks:
#                 last_face_time = time.time()
#                 lm = res.multi_face_landmarks[0].landmark
#                 pts = landmarks_to_pixel_array(lm, frame.shape)

#                 feats = compute_features(pts)
#                 emotion, votes, feats = vote_emotion(feats)

#                 # smoothing
#                 history.append(emotion)
#                 if len(history) > SMOOTHING_WINDOW:
#                     history.pop(0)
#                 emotion = max(set(history), key=history.count)

#                 phrase = random.choice(FEEDBACK.get(emotion, FEEDBACK["neutral"]))
#                 text = f"Emotion detected: {emotion}. {phrase}"

#                 # overlay
#                 mp_drawing.draw_landmarks(frame, res.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
#                                           mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
#                                           mp_drawing.DrawingSpec(color=(0,128,255), thickness=1))
#                 cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

#                 # continuously speak
#                 tts.speak(text)

#             else:
#                 cv2.putText(frame, "No face detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

#             cv2.imshow(win, frame)

#             # auto-close if no face
#             if time.time() - last_face_time > NO_FACE_TIMEOUT:
#                 break

#             if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
#                 break

#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#     except KeyboardInterrupt:
#         pass
#     finally:
#         try:
#             face_mesh.close()
#         except:
#             pass
#         cap.release()
#         cv2.destroyAllWindows()
#         tts.stop()

# if __name__ == "__main__":
#     main()



"""
mirrormind_fer_voice.py

MirrorMind using FER library for emotion detection with continuous voice feedback
"""
import cv2
import time
import random
import threading
import queue
from fer import FER

# -------- CONFIG --------
MAX_CAMERA_INDEX = 4
CAMERA_OPEN_TIMEOUT = 3.0
NO_FACE_TIMEOUT = 5.0
SMOOTHING_WINDOW = 5
USE_DSHOW = True
VERBOSE = False

# -------- FEEDBACK MESSAGES --------
FEEDBACK = {
    "happy": [
        "You look happy. Keep smiling.",
        "Nice smile! That suits you.",
        "Good to see you smiling."
    ],
    "sad": [
        "You seem sad. It's okay, take a deep breath.",
        "You look a bit down. Consider a short break.",
        "If you need, pause and take care of yourself."
    ],
    "surprise": [
        "You look surprised! Something unexpected happened?",
        "That expression shows surprise.",
        "Wow — that looks surprising."
    ],
    "angry": [
        "You seem angry. Try a slow, deep breath to calm down.",
        "Take a moment to relax your shoulders and breathe.",
        "Consider a short pause to cool down."
    ],
    "fear": [
        "You seem afraid. Stay calm and breathe.",
        "Take a moment to relax — all will be fine.",
    ],
    "disgust": [
        "Hmm, that looks like disgust. Take a moment to relax.",
    ],
    "neutral": [
        "You look calm and neutral.",
        "A peaceful expression — all looks fine.",
        "You appear composed."
    ],
}

# -------- Non-blocking TTS worker --------
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    TTS_AVAILABLE = False

class TTSWorker:
    """Non-blocking TTS engine using pyttsx3 and a queue."""
    def __init__(self):
        self.enabled = TTS_AVAILABLE
        self.q = queue.Queue()
        self.thread = None
        self.running = False
        if self.enabled:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 165)
            self.engine.setProperty("volume", 1.0)

    def start(self):
        if not self.enabled or (self.thread and self.thread.is_alive()):
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                text = self.q.get(timeout=0.5)
                self.engine.say(text)
                self.engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print("TTS error:", e)

    def speak(self, text):
        if not self.enabled:
            print("[TTS unavailable] " + text)
            return
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def stop(self):
        self.running = False

# -------- CAMERA UTILITIES --------
def try_open_camera(index, use_dshow=False, timeout=CAMERA_OPEN_TIMEOUT):
    """Attempt to open a camera and read a frame within timeout."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if use_dshow else 0)
    if not cap.isOpened():
        return None
    start = time.time()
    while time.time() - start < timeout:
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        time.sleep(0.05)
    cap.release()
    return None

def find_camera(max_index=MAX_CAMERA_INDEX, use_dshow=USE_DSHOW):
    """Find the first available camera."""
    for i in range(max_index + 1):
        cap = try_open_camera(i, use_dshow)
        if cap:
            return cap, i
    return None, None

# -------- MAIN LOOP --------
def main():
    tts = TTSWorker()
    tts.start()

    detector = FER(mtcnn=True)

    cap, idx = find_camera()
    if cap is None:
        print("ERROR: No camera found.")
        return

    win = "MirrorMind FER Voice"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    last_face_time = time.time()
    history = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if time.time() - last_face_time > 2.0:
                    print("ERROR: Camera frame stopped.")
                    break
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            faces = detector.detect_emotions(frame)

            if faces:
                last_face_time = time.time()
                # Take the largest face
                top_face = max(faces, key=lambda f: f['box'][2]*f['box'][3])
                emo_scores = top_face['emotions']
                emotion = max(emo_scores, key=emo_scores.get)

                # Smooth history
                history.append(emotion)
                if len(history) > SMOOTHING_WINDOW:
                    history.pop(0)
                emotion = max(set(history), key=history.count)

                # Prepare feedback text
                phrase = random.choice(FEEDBACK.get(emotion, FEEDBACK["neutral"]))
                text = f"Emotion detected: {emotion}. {phrase}"

                # Draw rectangle & text
                x, y, w, h = top_face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,255), 2, cv2.LINE_AA)

                # Speak emotion
                tts.speak(text)
            else:
                cv2.putText(frame, "No face detected", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow(win, frame)

            if time.time() - last_face_time > NO_FACE_TIMEOUT:
                print("No face detected for timeout. Exiting...")
                break

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.stop()

if __name__ == "__main__":
    main()
