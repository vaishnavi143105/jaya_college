import os
import socket
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from dotenv import load_dotenv

from backend.database.connection import init_db
from backend.routes.auth import auth_bp
from backend.routes.users import users_bp
from backend.routes.signconnect import signconnect_bp
from backend.routes.translation import translation_bp
from backend.socket_server import handle_socket

load_dotenv()

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)
sock = Sock(app)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(users_bp)
app.register_blueprint(signconnect_bp)
app.register_blueprint(translation_bp)

# WebSocket Endpoint for WebRTC Signaling & Subtitles
@sock.route('/ws/<user_id>')
def ws_route(ws, user_id):
    handle_socket(ws, user_id)

# Serve Frontend Pages & Static Assets
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static_pages(filename):
    file_path = os.path.join('frontend', filename)
    if os.path.exists(file_path):
        return send_from_directory('frontend', filename)
    return send_from_directory('frontend', 'index.html')

def get_local_ip():
    """Detect local IP on the shared local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == '__main__':
    try:
        init_db()
    except Exception as e:
        print(f"[WARN] Database initialization notice: {e}")

    port = int(os.getenv("PORT", 8000))
    local_ip = get_local_ip()

    print("\n" + "=" * 55)
    print("  MISSVOICE P2P WEBRTC SERVER")
    print(f"  Laptop 1 (Host):  http://localhost:{port}")
    print(f"  Laptop 2 (Peer):  http://{local_ip}:{port}")
    print("=" * 55 + "\n")

    # threaded=True ensures WebRTC signaling & AI inference do not block each other
    # use_reloader=False prevents TensorFlow/MediaPipe double-allocation on startup
    app.run(host='0.0.0.0', port=port, threaded=True, debug=True, use_reloader=False)