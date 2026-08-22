import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

mp_holistic = mp.solutions.holistic

class SignDetector:
    def __init__(self):
        model_path = 'ai/models/sign_lstm_model.keras'
        actions_path = 'ai/models/actions.npy'
        
        self.model = tf.keras.models.load_model(model_path) if os.path.exists(model_path) else None
        self.actions = np.load(actions_path) if os.path.exists(actions_path) else np.array([])
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = []
        self.threshold = 0.70

    def extract_keypoints(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([lh, rh])

    def process_frame(self, frame):
        if self.model is None or len(self.actions) == 0:
            return None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        keypoints = self.extract_keypoints(results)
        
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            if res[np.argmax(res)] > self.threshold:
                return str(self.actions[np.argmax(res)])
        return None