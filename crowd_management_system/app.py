from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
from datetime import datetime

app = Flask(__name__)

# Configuration file for settings
CONFIG_FILE = 'config.json'


def load_config():
    """Load configuration from file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "location_name": "Canteen",
                "max_capacity": 20
            }
            save_config(default_config)
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {"location_name": "Canteen", "max_capacity": 20}


def save_config(config):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_latest_data():
    """Read the latest crowd data from JSON file"""
    try:
        if not os.path.exists('crowd_data.json'):
            return None

        with open('crowd_data.json', 'r') as f:
            lines = f.readlines()
            if lines:
                # Get last line (most recent data)
                latest = json.loads(lines[-1])
                return latest
    except Exception as e:
        print(f"Error reading data: {e}")
        return None


def get_all_data(limit=20):
    """Get recent crowd data for history"""
    try:
        if not os.path.exists('crowd_data.json'):
            return []

        with open('crowd_data.json', 'r') as f:
            lines = f.readlines()
            # Get last 'limit' entries
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            data = [json.loads(line) for line in recent_lines]
            return data
    except Exception as e:
        print(f"Error reading history: {e}")
        return []


@app.route('/')
def home():
    """Main dashboard page"""
    latest = get_latest_data()
    history = get_all_data(20)
    config = load_config()
    return render_template('index.html', data=latest, history=history, config=config)

@app.route('/update_camera', methods=['POST'])
def update_camera():
    data = request.json
    config = load_config()
    config['camera_source'] = data.get('camera_source', '0')
    save_config(config)
    return jsonify({'success': True, 'message': 'Camera updated!'})

@app.route('/get_camera', methods=['GET'])
def get_camera():
    config = load_config()
    return jsonify({'camera_source': config.get('camera_source', '0')})


@app.route('/settings')
def settings():
    """Settings page"""
    config = load_config()
    return render_template('settings.html', config=config)


@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update settings from form"""
    try:
        config = {
            "location_name": request.form.get('location_name', 'Canteen'),
            "max_capacity": int(request.form.get('max_capacity', 20))
        }

        if save_config(config):
            return redirect(url_for('settings', success=1))
        else:
            return redirect(url_for('settings', error=1))
    except Exception as e:
        print(f"Error updating settings: {e}")
        return redirect(url_for('settings', error=1))


@app.route('/api/config')
def api_config():
    """API endpoint for current configuration"""
    config = load_config()
    return jsonify(config)


@app.route('/api/latest')
def api_latest():
    """API endpoint for latest data (for auto-refresh)"""
    latest = get_latest_data()
    return jsonify(latest if latest else {})


@app.route('/api/history')
def api_history():
    """API endpoint for historical data"""
    history = get_all_data(50)
    return jsonify(history)

@app.route('/get_cameras', methods=['GET'])
def get_cameras():
    config = load_config()
    cameras = config.get('cameras', [])
    return jsonify({'cameras': cameras})

@app.route('/update_cameras', methods=['POST'])
def update_cameras():
    try:
        data = request.json
        config = load_config()
        config['cameras'] = data.get('cameras', [])
        save_config(config)
        return jsonify({'success': True, 'message': 'Cameras updated!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/add_camera', methods=['POST'])
def add_camera():
    try:
        data = request.json
        config = load_config()
        cameras = config.get('cameras', [])

        # Generate new camera ID
        new_id = max([c['id'] for c in cameras], default=0) + 1

        new_camera = {
            'id': new_id,
            'name': data.get('name', f'Camera {new_id}'),
            'url': data.get('url', '0'),
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
    try:
        data = request.json
        camera_id = data.get('id')
        config = load_config()
        cameras = config.get('cameras', [])
        cameras = [c for c in cameras if c['id'] != camera_id]
        config['cameras'] = cameras
        save_config(config)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    try:
        data = request.json
        camera_id = data.get('id')
        config = load_config()
        cameras = config.get('cameras', [])
        for cam in cameras:
            if cam['id'] == camera_id:
                cam['enabled'] = not cam['enabled']
                break
        config['cameras'] = cameras
        save_config(config)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    print("=" * 60)
    print(" Starting Crowd Management Dashboard...")
    print("=" * 60)
    print(" Dashboard URL: http://127.0.0.1:5000")
    print(" Settings URL: http://127.0.0.1:5000/settings")
    print("=" * 60)
    print(" Make sure crowd_counter.py is running to generate data!")
    print(" Press CTRL+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)