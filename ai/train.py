import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mp_holistic = mp.solutions.holistic
DATA_PATH = os.path.join('ai', 'dataset', 'raw')
SEQUENCE_LENGTH = 30

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def train_models():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Path {DATA_PATH} does not exist!")
        return

    actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    actions.sort()
    ACTIONS = np.array(actions)
    print(f"[INFO] Found {len(ACTIONS)} classes: {ACTIONS}")

    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(ACTIONS)}

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            action_dir = os.path.join(DATA_PATH, action)
            video_files = [f for f in os.listdir(action_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"[INFO] Processing '{action}' ({len(video_files)} videos)...")

            for video_file in video_files:
                cap = cv2.VideoCapture(os.path.join(action_dir, video_file))
                window = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    window.append(extract_keypoints(results))
                    if len(window) == SEQUENCE_LENGTH:
                        sequences.append(window)
                        labels.append(label_map[action])
                        window = []
                cap.release()

    if len(sequences) == 0:
        print("[ERROR] No video sequence data extracted. Ensure video files are placed inside ai/dataset/raw/<class_name>/")
        return

    X = np.array(sequences)
    y = np.array(labels)
    y_cat = to_categorical(y, num_classes=len(ACTIONS)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=42)

    os.makedirs('ai/models', exist_ok=True)
    np.save('ai/models/actions.npy', ACTIONS)

    # 1. Train LSTM Model
    print("\n[INFO] Training LSTM Model...")
    lstm_model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 126)),
        Dropout(0.2),
        LSTM(128, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(len(ACTIONS), activation='softmax')
    ])
    lstm_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    lstm_model.fit(X_train, y_train, epochs=35, batch_size=16, validation_data=(X_test, y_test))
    lstm_model.save('ai/models/sign_lstm_model.keras')

    # 2. Train RandomForest Model
    print("\n[INFO] Training RandomForest Model...")
    X_rf = X.reshape(X.shape[0], -1)
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_rf, y)
    joblib.dump(rf_model, 'ai/models/sign_randomforest.pkl')

    print("\n============================================================")
    print(" [SUCCESS] Models trained and saved to ai/models/")
    print("============================================================")

if __name__ == '__main__':
    train_models()