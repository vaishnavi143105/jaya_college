from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from backend.database.connection import get_db_connection

users_bp = Blueprint('users', __name__)

@users_bp.route('/api/users/search', methods=['GET'])
def search_user():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({'detail': 'Search query is empty.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT missvoice_id, name, email FROM users WHERE missvoice_id = %s",
            (query,)
        )
        user = cursor.fetchone()
        if not user:
            return jsonify({'detail': 'User not found.'}), 404
        return jsonify(user), 200
    finally:
        cursor.close()
        conn.close()

@users_bp.route('/api/users/<missvoice_id>', methods=['GET'])
def get_profile(missvoice_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT missvoice_id, name, email FROM users WHERE missvoice_id = %s",
            (missvoice_id,)
        )
        user = cursor.fetchone()
        if not user:
            return jsonify({'detail': 'User not found.'}), 404
        return jsonify(user), 200
    finally:
        cursor.close()
        conn.close()

@users_bp.route('/api/users/profile/update', methods=['PUT'])
def update_profile():
    data = request.get_json() or {}
    mv_id = data.get('missvoice_id')
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        if password:
            pw_hash = generate_password_hash(password)
            cursor.execute(
                "UPDATE users SET name = %s, email = %s, password_hash = %s WHERE missvoice_id = %s",
                (name, email, pw_hash, mv_id)
            )
        else:
            cursor.execute(
                "UPDATE users SET name = %s, email = %s WHERE missvoice_id = %s",
                (name, email, mv_id)
            )
        return jsonify({'message': 'Profile updated successfully.'}), 200
    except Exception as e:
        return jsonify({'detail': str(e)}), 500
    finally:
        cursor.close()
        conn.close()