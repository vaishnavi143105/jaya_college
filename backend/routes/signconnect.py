from flask import Blueprint, request, jsonify
from backend.database.connection import get_db_connection

signconnect_bp = Blueprint("signconnect", __name__, url_prefix="/api/connections")

def resolve_user_id(cursor, identifier):
    """Helper to convert string/numeric MissVoice IDs into database integer IDs."""
    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        return int(identifier)
    
    cursor.execute("SELECT id FROM users WHERE missvoice_id = %s OR email = %s", (identifier, identifier))
    row = cursor.fetchone()
    return row["id"] if row else None

@signconnect_bp.route("/request", methods=["POST"])
def send_request():
    data = request.get_json() or {}
    sender_raw = data.get("sender_id")
    receiver_raw = data.get("receiver_id")

    if not sender_raw or not receiver_raw:
        return jsonify({"error": "Missing sender_id or receiver_id"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        sender_id = resolve_user_id(cursor, sender_raw)
        receiver_id = resolve_user_id(cursor, receiver_raw)

        if not sender_id or not receiver_id:
            return jsonify({"error": "User not found"}), 404

        if sender_id == receiver_id:
            return jsonify({"error": "Cannot connect to yourself"}), 400

        cursor.execute(
            """
            SELECT * FROM connections 
            WHERE (sender_id = %s AND receiver_id = %s) 
               OR (sender_id = %s AND receiver_id = %s)
            """,
            (sender_id, receiver_id, receiver_id, sender_id)
        )
        existing = cursor.fetchone()
        if existing:
            return jsonify({"message": "Connection or request already exists", "status": existing["status"]}), 200

        cursor.execute(
            "INSERT INTO connections (sender_id, receiver_id, status) VALUES (%s, %s, 'pending')",
            (sender_id, receiver_id)
        )
        conn.commit()
        return jsonify({"message": "Connection request sent successfully"}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@signconnect_bp.route("/accept", methods=["POST"])
def accept_request():
    data = request.get_json() or {}
    sender_raw = data.get("sender_id")
    receiver_raw = data.get("receiver_id")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        sender_id = resolve_user_id(cursor, sender_raw)
        receiver_id = resolve_user_id(cursor, receiver_raw)

        if not sender_id or not receiver_id:
            return jsonify({"error": "User not found"}), 404

        cursor.execute(
            """
            UPDATE connections 
            SET status = 'accepted' 
            WHERE sender_id = %s AND receiver_id = %s AND status = 'pending'
            """,
            (sender_id, receiver_id)
        )
        conn.commit()
        return jsonify({"message": "Connection accepted"}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@signconnect_bp.route("/<user_identifier>", methods=["GET"])
def get_connections(user_identifier):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        user_id = resolve_user_id(cursor, user_identifier)
        if not user_id:
            return jsonify({"connections": [], "pending_requests": []}), 200

        # Accepted active connections
        cursor.execute(
            """
            SELECT u.id, u.name, u.email, u.missvoice_id 
            FROM users u
            JOIN connections c ON (
                (c.sender_id = %s AND c.receiver_id = u.id) OR 
                (c.receiver_id = %s AND c.sender_id = u.id)
            )
            WHERE c.status = 'accepted'
            """,
            (user_id, user_id)
        )
        connections = cursor.fetchall()

        # Pending requests for this user
        cursor.execute(
            """
            SELECT u.id, u.name, u.email, u.missvoice_id 
            FROM users u
            JOIN connections c ON c.sender_id = u.id
            WHERE c.receiver_id = %s AND c.status = 'pending'
            """,
            (user_id,)
        )
        pending = cursor.fetchall()

        return jsonify({
            "connections": connections,
            "pending_requests": pending
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()