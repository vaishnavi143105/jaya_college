import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

KNOWN_CLASSES = [
    "cold", "dizzy", "earache", "family", "fever",
    "goodbye", "headache", "hello", "house", "i",
    "i love you", "love", "no", "please", "sorry",
    "thank you", "unconscious", "yes", "you_are_welcome"
]

def locate_dataset_dir():
    # Recursively search for any folder matching a known class (e.g., 'hello' or 'cold')
    for root, dirs, _ in os.walk("."):
        dirs_lower = [d.lower() for d in dirs]
        if "hello" in dirs_lower and ("cold" in dirs_lower or "sorry" in dirs_lower):
            return os.path.abspath(root)
    return None

DATASET_DIR = locate_dataset_dir()

if not DATASET_DIR:
    print("[ERROR] Could not find the folder containing gesture classes.")
    exit(1)

print(f"[INFO] Discovered dataset directory at: {DATASET_DIR}")

OUTPUT_MODEL_DIR = os.path.join("ai", "models")
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Scan actual existing subdirectories
ACTIONS = np.array(sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]))
print(f"[INFO] Found {len(ACTIONS)} classes: {ACTIONS}")

SEQUENCE_LENGTH = 30
KEYPOINTS_DIM = 126

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

def process_sample(item_path, holistic):
    frames_keypoints = []

    if os.path.isdir(item_path):
        items = sorted(os.listdir(item_path))
        npy_files = [f for f in items if f.endswith(".npy")]
        img_files = [f for f in items if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if npy_files:
            for f in npy_files:
                frames_keypoints.append(np.load(os.path.join(item_path, f)))
        elif img_files:
            for f in img_files:
                img = cv2.imread(os.path.join(item_path, f))
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = holistic.process(rgb)
                    frames_keypoints.append(extract_keypoints(res))

    elif os.path.isfile(item_path):
        if item_path.endswith(".npy"):
            data = np.load(item_path)
            if len(data.shape) == 2:
                return data
        elif item_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            cap = cv2.VideoCapture(item_path)
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
            action_dir = os.path.join(DATASET_DIR, action)
            entries = sorted(os.listdir(action_dir))
            count = 0
            print(f"[INFO] Processing class: '{action}' ({len(entries)} entries)...")

            for item in entries:
                item_path = os.path.join(action_dir, item)
                keypoints_seq = process_sample(item_path, holistic)
                if keypoints_seq is not None:
                    sequences.append(keypoints_seq)
                    labels.append(label_map[action])
                    count += 1

            print(f"       -> Extracted {count} sequences for '{action}'")

    X = np.array(sequences)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(ACTIONS)).astype(int)
    return X, y

def train():
    print("\n--- [1/3] Extracting Hand Keypoints ---")
    X, y = build_dataset()

    if len(X) == 0:
        print("[ERROR] No valid data sequences extracted.")
        return

    print(f"\n--- [2/3] Dataset Shape: X = {X.shape}, y = {y.shape} ---")

    model = tf.keras.models.Sequential([
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

    print(f"\n[SUCCESS] Model saved to: {model_path}")
    print(f"[SUCCESS] Action labels saved to: {actions_path}")

if __name__ == "__main__":
    train()
