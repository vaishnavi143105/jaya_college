import os
import cv2
import numpy as np
import tensorflow as tf

# Define relative paths from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ai', 'models', 'ssd_mobilenet', 'saved_model')
LABELMAP_PATH = os.path.join(BASE_DIR, 'ai', 'labelmap.pbtxt')

def load_label_map(labelmap_path):
    category_index = {}
    if not os.path.exists(labelmap_path):
        print(f"Warning: Labelmap not found at {labelmap_path}. Using default fallback.")
        return {1: 'i', 2: 'i_love_you', 3: 'thankyou', 4: 'need', 5: 'help', 6: 'yes'}

    with open(labelmap_path, 'r') as f:
        lines = f.readlines()
        current_id = None
        for line in lines:
            line = line.strip()
            if line.startswith('id:'):
                current_id = int(line.split(':')[1].strip())
            elif line.startswith('name:') and current_id is not None:
                name = line.split(':')[1].strip().replace("'", "").replace('"', '')
                category_index[current_id] = name
                current_id = None
    return category_index

print("[INFO] Loading TensorFlow SavedModel...")
model = tf.saved_model.load(MODEL_PATH)
detect_fn = model.signatures['serving_default']
category_index = load_label_map(LABELMAP_PATH)
print("[INFO] Model loaded successfully.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

print("[INFO] Starting video stream. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Preprocess image for SSD MobileNet
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = detect_fn(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    detected_text = "Scanning..."
    confidence_text = ""

    # Thresholding & visualization
    for i in range(len(scores)):
        if scores[i] > 0.60:
            class_id = classes[i]
            class_name = category_index.get(class_id, f"Sign_{class_id}")
            score_pct = int(scores[i] * 100)

            ymin, xmin, ymax, xmax = boxes[i]
            (startX, startY, endX, endY) = (int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height))

            # Draw bounding box and label
            label = f"{class_name.upper()} {score_pct}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (56, 189, 248), 2)
            cv2.putText(frame, label, (startX, max(startY - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2)

            detected_text = f"Detected: {class_name.upper()}"
            confidence_text = f"Confidence: {score_pct}%"
            break

    # HUD display on top left
    cv2.putText(frame, detected_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 185, 129), 2)
    if confidence_text:
        cv2.putText(frame, confidence_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)

    cv2.imshow('MissVoice 4.0 - Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()