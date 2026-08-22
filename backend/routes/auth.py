import random
import string
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from backend.database.connection import get_db_connection

auth_bp = Blueprint('auth', __name__)

def generate_missvoice_id(name):
    clean_name = "".join(filter(str.isalnum, name.lower()))
    random_digits = "".join(random.choices(string.digits, k=4))
    return f"@{clean_name}_{random_digits}"

@auth_bp.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not name or not email or not password:
        return jsonify({'detail': 'All fields are required.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({'detail': 'Email is already registered.'}), 400

        # Generate unique MissVoice ID
        while True:
            mv_id = generate_missvoice_id(name)
            cursor.execute("SELECT id FROM users WHERE missvoice_id = %s", (mv_id,))
            if not cursor.fetchone():
                break

        pw_hash = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (missvoice_id, name, email, password_hash) VALUES (%s, %s, %s, %s)",
            (mv_id, name, email, pw_hash)
        )

        return jsonify({
            'message': 'User registered successfully.',
            'missvoice_id': mv_id,
            'name': name
        }), 201
    except Exception as e:
        return jsonify({'detail': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@auth_bp.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    identifier = data.get('identifier', '').strip()
    password = data.get('password', '')

    if not identifier or not password:
        return jsonify({'detail': 'Identifier and password are required.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            "SELECT * FROM users WHERE email = %s OR missvoice_id = %s",
            (identifier.lower(), identifier)
        )
        user = cursor.fetchone()

        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'detail': 'Invalid credentials.'}), 401

        return jsonify({
            'access_token': f"token_{user['missvoice_id']}",
            'missvoice_id': user['missvoice_id'],
            'name': user['name']
        }), 200
    except Exception as e:
        return jsonify({'detail': str(e)}), 500
    finally:
        cursor.close()
        conn.close()