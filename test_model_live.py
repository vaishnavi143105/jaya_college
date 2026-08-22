import os
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = os.path.join('ai', 'models', 'sign_lstm_model.keras')
ACTIONS_PATH = os.path.join('ai', 'models', 'actions.npy')

if not os.path.exists(MODEL_PATH) or not os.path.exists(ACTIONS_PATH):
    print(f"[ERROR] Missing model files at: {MODEL_PATH} or {ACTIONS_PATH}")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
actions = np.load(ACTIONS_PATH)

@tf.function(input_signature=[tf.TensorSpec(shape=[1, 30, 126], dtype=tf.float32)])
def predict_fn(tensor):
    return model(tensor, training=False)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

# Non-blocking voice synthesizer
def speak_async(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 145)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")
    threading.Thread(target=_speak, daemon=True).start()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Ensure no browser tabs or apps are using it.")
    exit(1)

sequence = []
top_predictions = []
current_candidate = None
candidate_start_time = 0

# Configured to 20% confidence and 2.0s hold duration
HOLD_DURATION = 2.0
CONFIDENCE_THRESHOLD = 0.20

sentence_history = []
last_spoken_word = None

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        has_hands = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        progress = 0.0

        if has_hands:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                input_tensor = tf.convert_to_tensor([sequence], dtype=tf.float32)
                res = predict_fn(input_tensor).numpy()[0]
                
                top_indices = np.argsort(res)[::-1][:3]
                top_predictions = [(actions[i], float(res[i])) for i in top_indices]
                
                best_action, best_conf = top_predictions[0]

                # 20% and above
                if best_conf >= CONFIDENCE_THRESHOLD:
                    now = time.time()
                    if current_candidate == best_action:
                        elapsed = now - candidate_start_time
                        progress = min(elapsed / HOLD_DURATION, 1.0)

                        # Once held 2 seconds -> Store and speak
                        if elapsed >= HOLD_DURATION and current_candidate != last_spoken_word:
                            sentence_history.append(current_candidate)
                            last_spoken_word = current_candidate
                            print(f"[STORED & SPOKEN]: {current_candidate}")
                            speak_async(current_candidate)
                    else:
                        current_candidate = best_action
                        candidate_start_time = now
                        progress = 0.0
                else:
                    current_candidate = None
                    progress = 0.0
        else:
            current_candidate = None
            progress = 0.0
            top_predictions = []
            last_spoken_word = None

        # Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 65), (15, 23, 42), -1)
        active_display = current_candidate.upper() if current_candidate else "DETECTING..."
        cv2.putText(frame, f"SIGN: {active_display}", (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 189, 248), 2, cv2.LINE_AA)

        # 2-second Hold Progress Bar (Green line)
        if current_candidate:
            bar_width = int(progress * (w - 30))
            cv2.rectangle(frame, (15, 55), (15 + bar_width, 62), (16, 185, 129), -1)
            cv2.putText(frame, f"Hold: {progress*100:.0f}%", (w - 160, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 185, 129), 2, cv2.LINE_AA)

        # Bottom Sentence History Bar
        cv2.rectangle(frame, (0, h - 50), (w, h), (30, 41, 59), -1)
        sentence_str = "Sentence: " + " ".join(sentence_history[-6:])
        cv2.putText(frame, sentence_str, (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 250, 252), 2, cv2.LINE_AA)

        # Top-3 Predictions list
        y_offset = h - 140
        for i, (act, prob) in enumerate(top_predictions):
            bar_w = int(prob * 140)
            cv2.rectangle(frame, (15, y_offset + (i * 24)), (15 + bar_w, y_offset + 18 + (i * 24)), (56, 189, 248), -1)
            cv2.putText(frame, f"{act}: {prob*100:.1f}%", (165, y_offset + 15 + (i * 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('MissVoice - 20% Conf + 2s Voice', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence_history.clear()

cap.release()
cv2.destroyAllWindows()
