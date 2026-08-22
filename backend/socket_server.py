import json

connected_users = {}

def handle_socket(ws, user_id):
    connected_users[user_id] = ws
    print(f"[WS] User connected: {user_id}")
    try:
        while True:
            message_raw = ws.receive()
            if not message_raw:
                break
            
            data = json.loads(message_raw)
            target = data.get("target")
            
            # Forward WebRTC signaling (offer/answer/ICE candidate/captions)
            if target and target in connected_users:
                connected_users[target].send(json.dumps(data))
    except Exception as e:
        print(f"[WS] Connection error with {user_id}: {e}")
    finally:
        if user_id in connected_users:
            del connected_users[user_id]
        print(f"[WS] User disconnected: {user_id}")