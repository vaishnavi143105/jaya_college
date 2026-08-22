import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

DATASET_RAW = os.path.join("dataset", "raw")
OUTPUT_MODEL_DIR = os.path.join("ai", "models")
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# 19 actions matching your dataset folder structure
ACTIONS = np.array([
    "cold", "dizzy", "earache", "family", "fever",
    "goodbye", "headache", "hello", "house", "i",
    "i love you", "love", "no", "please", "sorry",
    "thank you", "unconscious", "yes", "you_are_welcome"
])

SEQUENCE_LENGTH = 30  # 30 frames per gesture sample
KEYPOINTS_DIM = 126    # 21 landmarks * 3 (x,y,z) * 2 hands

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

def process_video_or_frames(file_or_dir_path, holistic):
    frames_keypoints = []

    # If the entry is a folder of images or .npy files
    if os.path.isdir(file_or_dir_path):
        items = sorted(os.listdir(file_or_dir_path))
        if any(f.endswith(".npy") for f in items):
            for f in items:
                if f.endswith(".npy"):
                    frames_keypoints.append(np.load(os.path.join(file_or_dir_path, f)))
        else:
            for f in items:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(os.path.join(file_or_dir_path, f))
                    if img is not None:
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        res = holistic.process(rgb)
                        frames_keypoints.append(extract_keypoints(res))

    # If the entry is a video file
    elif os.path.isfile(file_or_dir_path) and file_or_dir_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(file_or_dir_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            frames_keypoints.append(extract_keypoints(res))
        cap.release()

    if len(frames_keypoints) == 0:
        return None

    # Resample or pad sequence to 30 frames
    if len(frames_keypoints) < SEQUENCE_LENGTH:
        while len(frames_keypoints) < SEQUENCE_LENGTH:
            frames_keypoints.append(frames_keypoints[-1])
    elif len(frames_keypoints) > SEQUENCE_LENGTH:
        indices = np.linspace(0, len(frames_keypoints) - 1, SEQUENCE_LENGTH, dtype=int)
        frames_keypoints = [frames_keypoints[i] for i in indices]

    return np.array(frames_keypoints)

def build_dataset():
    sequences = []
    labels = []
    label_map = {action: idx for idx, action in enumerate(ACTIONS)}

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            action_dir = os.path.join(DATASET_RAW, action)
            if not os.path.exists(action_dir):
                print(f"[WARN] Directory not found for '{action}', skipping...")
                continue

            entries = sorted(os.listdir(action_dir))
            count = 0
            print(f"[INFO] Processing class: '{action}' ({len(entries)} entries)...")

            for item in entries:
                item_path = os.path.join(action_dir, item)
                keypoints_seq = process_video_or_frames(item_path, holistic)
                if keypoints_seq is not None:
                    sequences.append(keypoints_seq)
                    labels.append(label_map[action])
                    count += 1

            print(f"       -> Extracted {count} sequences for '{action}'")

    X = np.array(sequences)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(ACTIONS)).astype(int)
    return X, y

def train():
    print("--- [1/3] Extracting keypoints from dataset/raw ---")
    X, y = build_dataset()

    if len(X) == 0:
        print("[ERROR] No valid data samples found. Check dataset/raw/<class_name>/")
        return

    print(f"\n--- [2/3] Dataset Shape: X = {X.shape}, y = {y.shape} ---")

    # Build LSTM Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQUENCE_LENGTH, KEYPOINTS_DIM)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=False, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(ACTIONS), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    print("\n--- [3/3] Training LSTM Model ---")
    model.fit(X, y, epochs=100, batch_size=16, validation_split=0.1, shuffle=True)

    model_path = os.path.join(OUTPUT_MODEL_DIR, "sign_lstm_model.keras")
    actions_path = os.path.join(OUTPUT_MODEL_DIR, "actions.npy")

    model.save(model_path)
    np.save(actions_path, ACTIONS)

    print(f"\n[SUCCESS] Model saved: {model_path}")
    print(f"[SUCCESS] Actions saved: {actions_path}")

if __name__ == "__main__":
    train()