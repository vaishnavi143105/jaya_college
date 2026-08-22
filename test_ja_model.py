import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_PATH = os.path.join('ai', 'models', 'sign_lstm_model_ja.keras')
ACTIONS_PATH = os.path.join('ai', 'models', 'actions_ja.npy')

if not os.path.exists(MODEL_PATH) or not os.path.exists(ACTIONS_PATH):
    print("[ERROR] Model files missing.")
    exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
actions = np.load(ACTIONS_PATH)
print(f"\n[INFO] Loaded JSL Model with {len(actions)} classes: {list(actions)}")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

JSL_DISPLAY = {
    'hello_j': ('こんにちは', 'HELLO'),
    'thankyou_j': ('ありがとう', 'THANK YOU'),
    'please_j': ('お願いします', 'PLEASE'),
    'yes_j': ('はい', 'YES'),
    'name_j': ('名前', 'NAME')
}

sequence = []
threshold = 0.50
predicted_action = "Show sign..."
confidence_score = 0.0

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        has_hands = bool(results.left_hand_landmarks or results.right_hand_landmarks)

        sequence.append(keypoints)
        sequence = sequence[-30:]

        if has_hands:
            if len(sequence) < 30:
                input_seq = [keypoints] * (30 - len(sequence)) + sequence
            else:
                input_seq = sequence

            input_data = np.expand_dims(input_seq, axis=0)
            res = model.predict(input_data, verbose=0)[0]
            best_idx = int(np.argmax(res))
            confidence_score = float(res[best_idx])
            current_action = str(actions[best_idx])

            top_indices = np.argsort(res)[::-1][:2]
            debug_info = " | ".join([f"{actions[i]}: {res[i]:.2f}" for i in top_indices])
            print(f"[LIVE] Top: {debug_info}")

            if confidence_score >= threshold:
                predicted_action = current_action
        else:
            confidence_score = 0.0

        cv2.rectangle(frame, (0, 0), (640, 75), (20, 20, 20), -1)
        
        display_label = predicted_action
        sub_label = ""
        if predicted_action in JSL_DISPLAY:
            jp_txt, en_txt = JSL_DISPLAY[predicted_action]
            display_label = f"{en_txt}"
            sub_label = f"JP: {jp_txt}"

        cv2.putText(frame, f"SIGN: {display_label}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        
        conf_color = (0, 255, 0) if confidence_score >= threshold else (0, 165, 255)
        cv2.putText(frame, f"CONF: {confidence_score:.2f} | {sub_label}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 1, cv2.LINE_AA)

        cv2.imshow('JSL Real-Time Test', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()