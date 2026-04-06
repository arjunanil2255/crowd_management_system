from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import json
import os
from datetime import datetime
from collections import deque
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

app = Flask(__name__)
app.secret_key = 'crowdsense_secret_key_2024'

CONFIG_FILE = 'config.json'
USERS_FILE  = 'users.json'


# ── Users helpers ────────────────────────────────────────────────

def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        else:
            default = {
                "users": [
                    {
                        "id": "1",
                        "name": "Admin",
                        "email": "admin@crowdsense.com",
                        "password": generate_password_hash("admin123"),
                        "role": "admin",
                        "status": "approved"
                    }
                ],
                "pending_requests": []
            }
            save_users(default)
            return default
    except Exception as e:
        print(f"Error loading users: {e}")
        return {"users": [], "pending_requests": []}


def save_users(data):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving users: {e}")
        return False


def find_user_by_email(email):
    data = load_users()
    for user in data['users']:
        if user['email'] == email:
            return user
    return None


def login_required(role=None):
    def decorator(f):
        from functools import wraps
        @wraps(f)
        def wrapped(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if role and session.get('role') != role:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return wrapped
    return decorator


# ── Config helpers ───────────────────────────────────────────────

def load_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            default_config = {
                "location_name": "Canteen",
                "max_capacity": 20,
                "camera_source": "0",
                "cameras": []
            }
            save_config(default_config)
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"location_name": "Canteen", "max_capacity": 20, "camera_source": "0", "cameras": []}


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_latest_data():
    try:
        if not os.path.exists('crowd_data.json'):
            return None
        with open('crowd_data.json', 'r') as f:
            valid_entry = None
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if all(k in entry for k in ['timestamp', 'count', 'status', 'location']):
                            valid_entry = entry
                    except:
                        pass
            return valid_entry
    except Exception as e:
        print(f"Error reading data: {e}")
    return None


def get_all_data(limit=20):
    try:
        if not os.path.exists('crowd_data.json'):
            return []
        with open('crowd_data.json', 'r') as f:
            valid_lines = deque((line for line in f if line.strip()), maxlen=limit)
            data = []
            for line in valid_lines:
                try:
                    entry = json.loads(line)
                    if all(k in entry for k in ['timestamp', 'count', 'status', 'location']):
                        data.append(entry)
                except:
                    pass
            return data
    except Exception as e:
        print(f"Error reading history: {e}")
    return []


# ── Auth Routes ──────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        role = session.get('role')
        if role == 'admin':    return redirect(url_for('home'))
        if role == 'security': return redirect(url_for('security'))
        if role == 'user':     return redirect(url_for('user_view'))

    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        user = find_user_by_email(email)

        if not user:
            error = 'No account found with this email.'
        elif user.get('status') == 'pending':
            error = 'Your account is pending admin approval.'
        elif user.get('status') == 'rejected':
            error = 'Your account request was rejected.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'
        else:
            session['user_id'] = user['id']
            session['name']    = user['name']
            session['email']   = user['email']
            session['role']    = user['role']

            if user['role'] == 'admin':    return redirect(url_for('home'))
            if user['role'] == 'security': return redirect(url_for('security'))
            if user['role'] == 'user':     return redirect(url_for('user_view'))

    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error   = None
    success = None

    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        role     = request.form.get('role', 'user')

        if not name or not email or not password:
            error = 'All fields are required.'
        elif find_user_by_email(email):
            error = 'An account with this email already exists.'
        elif role not in ['user', 'security']:
            error = 'Invalid role selected.'
        else:
            data   = load_users()
            new_id = str(uuid.uuid4())[:8]

            new_user = {
                "id":       new_id,
                "name":     name,
                "email":    email,
                "password": generate_password_hash(password),
                "role":     role,
                "status":   "approved" if role == 'user' else "pending"
            }

            data['users'].append(new_user)
            save_users(data)

            if role == 'security':
                success = 'Security account request submitted! Please wait for admin approval.'
            else:
                success = 'Account created! You can now login.'

    return render_template('register.html', error=error, success=success)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── Protected Routes ─────────────────────────────────────────────

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    latest  = get_latest_data()
    history = get_all_data(20)
    config  = load_config()
    return render_template('index.html', data=latest, history=history, config=config)


@app.route('/security')
def security():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') not in ['admin', 'security']:
        return redirect(url_for('login'))
    config = load_config()
    latest = get_latest_data()
    return render_template('security.html', config=config, data=latest)


@app.route('/user')
def user_view():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') not in ['admin', 'user']:
        return redirect(url_for('login'))
    latest = get_latest_data()
    return render_template('user.html', data=latest)


@app.route('/settings')
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    config = load_config()
    return render_template('settings.html', config=config)


# ── Admin User Management ────────────────────────────────────────

@app.route('/admin/users')
def admin_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    data = load_users()
    return render_template('admin_users.html', users=data['users'])


@app.route('/admin/add_user', methods=['POST'])
def admin_add_user():
    if session.get('role') != 'admin':
        return jsonify({'success': False, 'message': 'Unauthorized'})
    try:
        data     = load_users()
        req      = request.json
        new_id   = str(uuid.uuid4())[:8]
        new_user = {
            "id":       new_id,
            "name":     req.get('name'),
            "email":    req.get('email').strip().lower(),
            "password": generate_password_hash(req.get('password')),
            "role":     req.get('role', 'user'),
            "status":   "approved"
        }
        if find_user_by_email(new_user['email']):
            return jsonify({'success': False, 'message': 'Email already exists'})
        data['users'].append(new_user)
        save_users(data)
        return jsonify({'success': True, 'message': f"User {new_user['name']} added!"})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/admin/approve_user', methods=['POST'])
def admin_approve_user():
    if session.get('role') != 'admin':
        return jsonify({'success': False})
    try:
        user_id = request.json.get('id')
        data    = load_users()
        for user in data['users']:
            if user['id'] == user_id:
                user['status'] = 'approved'
                break
        save_users(data)
        return jsonify({'success': True, 'message': 'User approved!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/admin/reject_user', methods=['POST'])
def admin_reject_user():
    if session.get('role') != 'admin':
        return jsonify({'success': False})
    try:
        user_id = request.json.get('id')
        data    = load_users()
        for user in data['users']:
            if user['id'] == user_id:
                user['status'] = 'rejected'
                break
        save_users(data)
        return jsonify({'success': True, 'message': 'User rejected!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/admin/delete_user', methods=['POST'])
def admin_delete_user():
    if session.get('role') != 'admin':
        return jsonify({'success': False})
    try:
        user_id  = request.json.get('id')
        data     = load_users()
        data['users'] = [u for u in data['users'] if u['id'] != user_id]
        save_users(data)
        return jsonify({'success': True, 'message': 'User deleted!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/admin/get_users')
def admin_get_users():
    if session.get('role') != 'admin':
        return jsonify({'success': False})
    data = load_users()
    safe_users = [
        {k: v for k, v in u.items() if k != 'password'}
        for u in data['users']
    ]
    return jsonify({'users': safe_users})


# ── Existing Routes ──────────────────────────────────────────────

@app.route('/update_camera', methods=['POST'])
def update_camera():
    if session.get('role') != 'admin': return jsonify({'success': False})
    data = request.json
    config = load_config()
    config['camera_source'] = data.get('camera_source', '0')
    save_config(config)
    return jsonify({'success': True, 'message': 'Camera updated!'})


@app.route('/get_camera', methods=['GET'])
def get_camera():
    config = load_config()
    return jsonify({'camera_source': config.get('camera_source', '0')})


@app.route('/update_settings', methods=['POST'])
def update_settings():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    try:
        config = load_config()
        config['location_name'] = request.form.get('location_name', 'Canteen')
        config['max_capacity']  = int(request.form.get('max_capacity', 20))
        if save_config(config):
            return redirect(url_for('settings', success=1))
        else:
            return redirect(url_for('settings', error=1))
    except Exception as e:
        print(f"Error updating settings: {e}")
        return redirect(url_for('settings', error=1))


@app.route('/api/config')
def api_config():
    config = load_config()
    return jsonify(config)


@app.route('/api/latest')
def api_latest():
    if 'user_id' not in session:
        return jsonify({})
    latest = get_latest_data()
    return jsonify(latest if latest else {})


@app.route('/api/history')
def api_history():
    if 'user_id' not in session:
        return jsonify([])
    history = get_all_data(50)
    return jsonify(history)


@app.route('/get_cameras', methods=['GET'])
def get_cameras():
    config = load_config()
    return jsonify({'cameras': config.get('cameras', [])})


@app.route('/add_camera', methods=['POST'])
def add_camera():
    if session.get('role') != 'admin': return jsonify({'success': False})
    try:
        data    = request.json
        config  = load_config()
        cameras = config.get('cameras', [])
        new_id  = max([c['id'] for c in cameras], default=0) + 1
        new_camera = {
            'id':      new_id,
            'name':    data.get('name', f'Camera {new_id}'),
            'url':     data.get('url', '0'),
            'enabled': True
        }
        cameras.append(new_camera)
        config['cameras'] = cameras
        save_config(config)
        return jsonify({'success': True, 'camera': new_camera})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    if session.get('role') != 'admin': return jsonify({'success': False})
    try:
        camera_id = request.json.get('id')
        config    = load_config()
        config['cameras'] = [c for c in config.get('cameras', []) if c['id'] != camera_id]
        save_config(config)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    if session.get('role') != 'admin': return jsonify({'success': False})
    try:
        camera_id = request.json.get('id')
        config    = load_config()
        for cam in config.get('cameras', []):
            if cam['id'] == camera_id:
                cam['enabled'] = not cam['enabled']
                break
        save_config(config)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/update_cameras', methods=['POST'])
def update_cameras():
    if session.get('role') != 'admin': return jsonify({'success': False})
    try:
        data   = request.json
        config = load_config()
        config['cameras'] = data.get('cameras', [])
        save_config(config)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting CrowdSense...")
    print("=" * 60)
    print("🔐 Login URL:     http://127.0.0.1:5000/login")
    print("📊 Dashboard URL: http://127.0.0.1:5000")
    print("⚙️  Settings URL:  http://127.0.0.1:5000/settings")
    print("=" * 60)
    print("Default Admin: admin@crowdsense.com / admin123")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)