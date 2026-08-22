import base64
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Blueprint, jsonify, request
from deep_translator import GoogleTranslator

translation_bp = Blueprint('translation', __name__)

# Paths for English Model
MODEL_PATH_EN = os.path.join('ai', 'models', 'sign_lstm_model.keras')
ACTIONS_PATH_EN = os.path.join('ai', 'models', 'actions.npy')

# Paths for Japanese (JSL) Model
MODEL_PATH_JA = os.path.join('ai', 'models', 'sign_lstm_model_ja.keras')
ACTIONS_PATH_JA = os.path.join('ai', 'models', 'actions_ja.npy')

# Model registries
models = {'en': None, 'ja': None}
actions = {'en': [], 'ja': []}
predict_fns = {'en': None, 'ja': None}

# JSL sign gesture classes to display mappings
JSL_DICTIONARY = {
    'hello_j': ('こんにちは', 'HELLO'),
    'thankyou_j': ('ありがとう', 'THANK YOU'),
    'please_j': ('お願いします', 'PLEASE'),
    'yes_j': ('はい', 'YES'),
    'name_j': ('名前', 'NAME'),
}


def load_model_instance(lang_key, model_path, actions_path):
  """Loads and compiles an LSTM sign recognition model."""
  if os.path.exists(model_path) and os.path.exists(actions_path):
    try:
      mod = tf.keras.models.load_model(model_path)
      act = np.load(actions_path)

      @tf.function(
          input_signature=[
              tf.TensorSpec(shape=[1, 30, 126], dtype=tf.float32)
          ]
      )
      def _predict(input_tensor):
        return mod(input_tensor, training=False)

      models[lang_key] = mod
      actions[lang_key] = act
      predict_fns[lang_key] = _predict
      print(
          f'[AI] {lang_key.upper()} Model initialized with {len(act)} actions:'
          f' {list(act)}'
      )
    except Exception as e:
      print(f'[AI] Failed to load {lang_key.upper()} model: {e}')
  else:
    print(
        f'[AI] Notice: {lang_key.upper()} model or actions file not found at'
        f' {model_path}.'
    )


# Initialize Gesture Models
load_model_instance('en', MODEL_PATH_EN, ACTIONS_PATH_EN)
load_model_instance('ja', MODEL_PATH_JA, ACTIONS_PATH_JA)

# MediaPipe Setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Sequence Buffers
user_buffers = {'en': [], 'ja': []}


def extract_keypoints(results):
  """Extracts left and right hand keypoints (126 elements)."""
  lh = (
      np.array([
          [res.x, res.y, res.z]
          for res in results.left_hand_landmarks.landmark
      ]).flatten()
      if results.left_hand_landmarks
      else np.zeros(21 * 3)
  )
  rh = (
      np.array([
          [res.x, res.y, res.z]
          for res in results.right_hand_landmarks.landmark
      ]).flatten()
      if results.right_hand_landmarks
      else np.zeros(21 * 3)
  )
  return np.concatenate([lh, rh])


def translate_prediction(raw_prediction, source_lang):
  """Translates camera gesture predictions."""
  key = str(raw_prediction).strip().lower()

  if source_lang == 'ja':
    if key in JSL_DICTIONARY:
      local_text, remote_text = JSL_DICTIONARY[key]
    else:
      clean = key.replace('_j', '')
      local_text = clean
      remote_text = clean.upper()
    target_lang = 'en'
  else:
    clean = key.replace('_j', '')
    local_text = clean.upper()
    try:
      # Live Google translation for ASL classes
      remote_text = GoogleTranslator(source='en', target='ja').translate(clean)
    except Exception:
      remote_text = clean
    target_lang = 'ja'

  return local_text, remote_text, target_lang


# ==========================================
# FULL-SENTENCE GOOGLE TRANSLATE FOR SPEECH
# ==========================================
@translation_bp.route('/api/translate/text', methods=['POST'])
def translate_spoken_text():
  """Translates any arbitrary spoken sentence using Google Translate."""
  data = request.get_json() or {}
  spoken_text = str(data.get('text', '')).strip()
  source_lang = str(data.get('source_lang', 'en')).lower()

  if not spoken_text:
    return jsonify({'error': 'No text provided'}), 400

  target_lang = 'ja' if source_lang == 'en' else 'en'

  try:
    # Universal Live Google Translation
    translated_output = GoogleTranslator(
        source=source_lang, target=target_lang
    ).translate(spoken_text)

    local_display = spoken_text
    remote_display = translated_output

    print(
        f'[GOOGLE TRANSLATE] ({source_lang}->{target_lang}) "{spoken_text}" ->'
        f' "{translated_output}"'
    )

    return (
        jsonify({
            'local_text': local_display,
            'remote_text': remote_display,
            'target_lang': target_lang,
            'source_lang': source_lang,
        }),
        200,
    )

  except Exception as e:
    print(f'[TRANSLATION ERROR] {e}')
    return (
        jsonify({
            'local_text': spoken_text,
            'remote_text': spoken_text,
            'target_lang': target_lang,
            'source_lang': source_lang,
        }),
        200,
    )


# ==========================================
# GESTURE INFERENCE ROUTE
# ==========================================
@translation_bp.route('/api/translate/frame', methods=['POST'])
@translation_bp.route('/api/detect_sign', methods=['POST'])
def process_frame():
  global user_buffers

  data = request.get_json() or {}
  frame_b64 = data.get('frame') or data.get('image') or ''
  selected_lang = data.get('language', 'en').lower()

  active_lang = (
      selected_lang if predict_fns.get(selected_lang) is not None else 'en'
  )
  active_predict_fn = predict_fns.get(active_lang)
  active_actions = actions.get(active_lang)

  if active_predict_fn is None or len(active_actions) == 0:
    return jsonify({'error': f'Model not loaded for {selected_lang}'}), 503

  if not frame_b64:
    return jsonify({'error': 'No frame provided'}), 400

  try:
    if ',' in frame_b64:
      frame_b64 = frame_b64.split(',')[1]

    img_bytes = base64.b64decode(frame_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
      return jsonify({'error': 'Invalid image'}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    keypoints = extract_keypoints(results)
    has_hands = bool(results.left_hand_landmarks or results.right_hand_landmarks)

    if active_lang not in user_buffers:
      user_buffers[active_lang] = []

    user_buffers[active_lang].append(keypoints)
    user_buffers[active_lang] = user_buffers[active_lang][-30:]

    if len(user_buffers[active_lang]) < 30:
      current_seq = [keypoints] * (
          30 - len(user_buffers[active_lang])
      ) + user_buffers[active_lang]
    else:
      current_seq = user_buffers[active_lang]

    input_tensor = tf.convert_to_tensor([current_seq], dtype=tf.float32)
    res = active_predict_fn(input_tensor).numpy()[0]
    best_idx = int(np.argmax(res))
    confidence = float(res[best_idx])
    raw_predicted_sign = str(active_actions[best_idx])

    if confidence >= 0.35 and has_hands:
      local_text, remote_text, target_lang = translate_prediction(
          raw_predicted_sign, active_lang
      )
      return (
          jsonify({
              'detected_sign': local_text,
              'prediction': local_text,
              'local_text': local_text,
              'remote_text': remote_text,
              'target_lang': target_lang,
              'confidence': round(confidence, 2),
              'language': active_lang,
          }),
          200,
      )

    return (
        jsonify({
            'detected_sign': None,
            'prediction': None,
            'local_text': None,
            'remote_text': None,
            'confidence': round(confidence, 2),
            'language': active_lang,
        }),
        200,
    )

  except Exception as e:
    return jsonify({'error': str(e)}), 500