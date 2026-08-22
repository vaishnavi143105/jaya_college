import time
from flask import Blueprint, request, jsonify

videocall_bp = Blueprint("videocall", __name__, url_prefix="/api/videocall")

# In-memory storage for active sessions and signaling cache
active_calls = {}
signaling_data = {}  # session_id -> {"offer": None, "answer": None, "candidates": []}

CALL_TIMEOUT_SECONDS = 60  # Stale session threshold


def cleanup_stale_calls():
    """Remove inactive or unanswered calls older than threshold."""
    now = time.time()
    expired = [
        sid for sid, call in active_calls.items()
        if call.get("status") == "ringing" and (now - call.get("created_at", 0)) > CALL_TIMEOUT_SECONDS
    ]
    for sid in expired:
        active_calls.pop(sid, None)
        signaling_data.pop(sid, None)


@videocall_bp.route("/initiate", methods=["POST"])
def initiate_call():
    cleanup_stale_calls()
    data = request.get_json() or {}
    caller_id = (data.get("caller_id") or "").strip()
    receiver_id = (data.get("receiver_id") or "").strip()

    if not caller_id or not receiver_id:
        return jsonify({"error": "Missing caller_id or receiver_id"}), 400

    if caller_id.lower() == receiver_id.lower():
        return jsonify({"error": "Cannot call yourself"}), 400

    # Prevent concurrent duplicate sessions between same peers
    for sid, call in list(active_calls.items()):
        if {call["caller_id"], call["receiver_id"]} == {caller_id, receiver_id} and call["status"] in ["ringing", "connected"]:
            return jsonify({
                "status": "existing",
                "session_id": sid,
                "caller": call["caller_id"],
                "receiver": call["receiver_id"]
            }), 200

    call_session_id = f"{caller_id}_{receiver_id}_{int(time.time())}"

    call_record = {
        "session_id": call_session_id,
        "caller_id": caller_id,
        "receiver_id": receiver_id,
        "status": "ringing",
        "created_at": time.time(),
        "connected_at": None
    }

    active_calls[call_session_id] = call_record
    signaling_data[call_session_id] = {"offer": None, "answer": None, "candidates": []}
    print(f"[CALL INITIATED] {caller_id} -> {receiver_id} | Session: {call_session_id}")

    return jsonify({
        "status": "initiated",
        "session_id": call_session_id,
        "caller": caller_id,
        "receiver": receiver_id
    }), 200


@videocall_bp.route("/check_incoming/<user_id>", methods=["GET"])
def check_incoming(user_id):
    """Allows dashboard polling if WebSocket signaling is delayed."""
    cleanup_stale_calls()
    uid = (user_id or "").strip()

    for sid, call in active_calls.items():
        if call["receiver_id"] == uid and call["status"] == "ringing":
            return jsonify({
                "incoming": True,
                "session_id": sid,
                "caller_id": call["caller_id"]
            }), 200

    return jsonify({"incoming": False}), 200


@videocall_bp.route("/accept", methods=["POST"])
def accept_call():
    data = request.get_json() or {}
    session_id = data.get("session_id")
    receiver_id = data.get("receiver_id")

    if not session_id or session_id not in active_calls:
        return jsonify({"error": "Invalid or expired session"}), 404

    active_calls[session_id]["status"] = "connected"
    active_calls[session_id]["connected_at"] = time.time()
    print(f"[CALL ACCEPTED] Session: {session_id} by {receiver_id}")

    return jsonify({
        "status": "connected",
        "session_id": session_id,
        "caller": active_calls[session_id]["caller_id"],
        "receiver": receiver_id
    }), 200


@videocall_bp.route("/decline", methods=["POST"])
def decline_call():
    data = request.get_json() or {}
    session_id = data.get("session_id")

    if session_id and session_id in active_calls:
        active_calls[session_id]["status"] = "declined"
        ended = active_calls.pop(session_id, None)
        signaling_data.pop(session_id, None)
        print(f"[CALL DECLINED] Session: {session_id}")
        return jsonify({"status": "declined", "session_id": session_id}), 200

    return jsonify({"status": "declined"}), 200


@videocall_bp.route("/end", methods=["POST"])
def end_call():
    data = request.get_json() or {}
    session_id = data.get("session_id")

    if session_id and session_id in active_calls:
        ended_call = active_calls.pop(session_id, None)
        signaling_data.pop(session_id, None)
        print(f"[CALL ENDED] Session: {session_id}")
        return jsonify({"status": "ended", "session_id": session_id}), 200

    return jsonify({"status": "ended"}), 200


@videocall_bp.route("/status/<session_id>", methods=["GET"])
def get_call_status(session_id):
    if session_id in active_calls:
        return jsonify(active_calls[session_id]), 200
    return jsonify({"status": "inactive"}), 404