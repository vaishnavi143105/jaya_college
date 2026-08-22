# MissVoice 4.0 🌐🤟

**MissVoice 4.0** is an AI-powered WebRTC video communication platform that provides real-time, bidirectional translation between **Sign Language** and **Spoken Voice** across English and Japanese.

---

## 🌟 Key Features

* **Dual-Language Sign Recognition (ASL & JSL):** Extracts 126 hand landmarks using **MediaPipe Holistic** and classifies dynamic sequences using custom multi-layer **LSTM** models.
* **Universal Speech-to-Text & Live Translation:** Transcribes continuous speech and translates arbitrary sentences into the remote peer's language using the **Google Translate API** via `deep-translator`.
* **Dynamic Mode Switching:**
  * **`SIGN: ON`** $\rightarrow$ Activates real-time camera AI inference for gesture recognition.
  * **`SIGN: OFF`** $\rightarrow$ Activates continuous Speech-to-Text (STT) voice transcription.
* **Subtitles & Text-to-Speech (TTS):** Synchronizes real-time subtitles across peers and speaks translated text using the native Web Speech API.
* **Peer-to-Peer Encrypted Calling:** Low-latency video and audio streaming via **WebRTC** with persistent **WebSocket** signaling.

---

## 🛠️ System Architecture & Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Web Speech API |
| **WebRTC & Signaling** | WebRTC (`RTCPeerConnection`), WebSockets (`Flask-Sock`) |
| **Computer Vision** | OpenCV (`cv2`), MediaPipe Holistic (126 hand landmarks) |
| **AI / Machine Learning** | TensorFlow / Keras (Stacked LSTM Networks), NumPy, Scikit-learn |
| **Backend & Translation** | Python, Flask, `deep-translator` (Google Translate API) |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/vaishnavi143105/jaya_college.git](https://github.com/vaishnavi143105/jaya_college.git)
cd jaya_college

Contributors & Team Members
Thesjesvani S — team leader / backend developer

Sujaramamoothy — frontend developer

Breshma — testing/ security

vaishnavi devi G - AI developer /  data trainer
