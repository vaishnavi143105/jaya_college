import os
import cv2
import numpy as np
import mediapipe as mp

# Source folder containing Japanese sign video folders
DATA_SOURCE = os.path.join('ai', 'dataset', 'raw02')

# Output destination for 30-frame sequence arrays
SAVE_PATH = os.path.join('ai', 'data', 'JSL_DATA')
os.makedirs(SAVE_PATH, exist_ok=True)

SEQUENCE_LENGTH = 30  # Standard sequence length (30 frames)
KEYPOINTS_DIM = 126   # 21 landmarks * 3 (x,y,z) * 2 hands

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    """Extracts left and right hand keypoints (126 elements)."""
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([lh, rh])

def process_video_file(video_path, holistic):
    """Reads a video, extracts hand keypoints, and normalizes length to 30 frames."""
    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = extract_keypoints(results)
        frames_keypoints.append(keypoints)
        
    cap.release()
    
    if len(frames_keypoints) == 0:
        return None
    
    # Resample or pad to exactly 30 frames
    if len(frames_keypoints) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(frames_keypoints) - 1, SEQUENCE_LENGTH, dtype=int)
        sampled = [frames_keypoints[i] for i in indices]
    else:
        padding = [frames_keypoints[0]] * (SEQUENCE_LENGTH - len(frames_keypoints))
        sampled = padding + frames_keypoints
        
    return np.array(sampled, dtype=np.float32)

def main():
    if not os.path.exists(DATA_SOURCE):
        print(f"[ERROR] Source folder '{DATA_SOURCE}' not found.")
        return

    actions = [d for d in os.listdir(DATA_SOURCE) if os.path.isdir(os.path.join(DATA_SOURCE, d))]
    actions.sort()
    
    print(f"[INFO] Found {len(actions)} actions in '{DATA_SOURCE}': {actions}")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            action_dir = os.path.join(DATA_SOURCE, action)
            dest_dir = os.path.join(SAVE_PATH, action)
            os.makedirs(dest_dir, exist_ok=True)
            
            video_files = [
                f for f in os.listdir(action_dir) 
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
            ]
            print(f"[EXTRACTING] Action '{action}' ({len(video_files)} videos)...")
            
            sample_count = 0
            for vid in video_files:
                vid_path = os.path.join(action_dir, vid)
                seq_data = process_video_file(vid_path, holistic)
                
                if seq_data is not None:
                    save_file = os.path.join(dest_dir, f"sample_{sample_count}.npy")
                    np.save(save_file, seq_data)
                    sample_count += 1
            
            print(f" -> Completed '{action}': {sample_count} sequences saved.")

    print(f"\n[DONE] Extraction complete! Preprocessed dataset saved to: {SAVE_PATH}")

if __name__ == '__main__':
    main()