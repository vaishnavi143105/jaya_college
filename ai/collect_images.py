import os
import time
import uuid
import cv2

# Define 6 target sign classes
LABELS = ['i', 'i_love_you', 'thankyou', 'need', 'help', 'yes']
NUMBER_IMGS = 15  # Number of images to capture per class

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATASET_DIR = os.path.join(BASE_DIR, 'ai', 'dataset', 'raw')

os.makedirs(RAW_DATASET_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not access the webcam.")
    exit()

print("==================================================")
print("MissVoice 4.0 - Dataset Collection Tool")
print("Signs to collect:", ", ".join(LABELS))
print("Images per sign:", NUMBER_IMGS)
print("==================================================")

for label in LABELS:
    class_dir = os.path.join(RAW_DATASET_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    print(f"\n[GET READY] Collecting images for: '{label.upper()}' in 5 seconds...")
    for countdown in range(5, 0, -1):
        print(f"Starting in {countdown}...")
        time.sleep(1)

    print(f"[RECORDING] Capturing frames for '{label}'...")

    img_count = 0
    while img_count < NUMBER_IMGS:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        # Generate unique image filename
        img_name = os.path.join(class_dir, f"{label}_{str(uuid.uuid1())[:8]}.jpg")
        cv2.imwrite(img_name, frame)
        img_count += 1

        # Display progress on screen
        cv2.putText(frame, f"Class: {label.upper()} ({img_count}/{NUMBER_IMGS})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (56, 189, 248), 2)
        cv2.imshow('Collecting Dataset - MissVoice', frame)

        # Delay between captures so you can adjust angles/positions slightly
        cv2.waitKey(500)

    print(f"[DONE] Collected {NUMBER_IMGS} images for '{label}'.")

cap.release()
cv2.destroyAllWindows()
print("\n[SUCCESS] Image collection complete. Check your 'ai/dataset/raw/' folder.")