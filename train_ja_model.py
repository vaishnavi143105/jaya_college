import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

DATA_PATH = os.path.join('ai', 'data', 'JSL_DATA')
MODELS_DIR = os.path.join('ai', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'sign_lstm_model_ja.keras')
ACTIONS_SAVE_PATH = os.path.join(MODELS_DIR, 'actions_ja.npy')

def load_dataset():
    actions = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    actions.sort()
    
    label_map = {label: num for num, label in enumerate(actions)}
    
    sequences, labels = [], []
    for action in actions:
        action_dir = os.path.join(DATA_PATH, action)
        for npy_file in os.listdir(action_dir):
            if npy_file.endswith('.npy'):
                res = np.load(os.path.join(action_dir, npy_file))
                sequences.append(res)
                labels.append(label_map[action])
                
    X = np.array(sequences, dtype=np.float32)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(actions)).astype(int)
    return X, y, np.array(actions)

def build_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape),
        Dropout(0.1),
        LSTM(64, return_sequences=False, activation='tanh'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model

def main():
    print("[INFO] Loading JSL dataset...")
    X, y, actions = load_dataset()
    print(f"[INFO] Dataset shape: X = {X.shape}, y = {y.shape}")
    print(f"[INFO] Actions ({len(actions)}): {list(actions)}")
    
    np.save(ACTIONS_SAVE_PATH, actions)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = build_model(input_shape=(30, 126), num_classes=len(actions))
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='loss', patience=30, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='categorical_accuracy', save_best_only=True, mode='max')
    ]
    
    print("\n[TRAINING] Starting model training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=8,
        callbacks=callbacks
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"\n[DONE] JSL model successfully trained and saved to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()