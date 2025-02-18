from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Create static folder if not exists
if not os.path.exists('static'):
    os.makedirs('static')

# ======================
# Core Routes
# ======================

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API documentation"""
    return jsonify({
        "system": "PalmAuth 2.0",
        "endpoints": {
            "/authenticate": "POST: Process palm authentication",
            "/enroll": "POST: Register new user",
            "/status": "GET: System health check"
        }
    })

@app.route('/favicon.ico')
def favicon():
    """Serve favicon for browser requests"""
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/status', methods=['GET'])
def health_check():
    """System health monitoring endpoint"""
    return jsonify({
        "status": "operational",
        "version": "2.0.1",
        "uptime": "99.999%"
    })

# ======================
# Authentication Logic
# ======================

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Main authentication endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Process image
        file = request.files['image'].read()
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Add your authentication logic here
        return jsonify({
            "status": "authenticated",
            "confidence": 0.9997,
            "user_id": "PAX-1234"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/enroll', methods=['POST'])
def enroll():
    """User enrollment endpoint"""
    try:
        # Ensure required fields exist: 'user_id' in form and 'image' in files
        if 'user_id' not in request.form or 'image' not in request.files:
            return jsonify({"error": "Missing required fields: 'user_id' and/or 'image'"}), 400

        # Process enrollment
        user_id = request.form['user_id']
        file = request.files['image'].read()
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Add your enrollment logic here
        return jsonify({
            "status": "enrolled",
            "user_id": user_id,
            "biometric_hash": "a9f8e7c6d5"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# Error Handling
# ======================

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 handler"""
    endpoints = [str(rule) for rule in app.url_map.iter_rules()]
    return jsonify({
        "error": "Endpoint not found",
        "valid_endpoints": endpoints
    }), 404

if __name__ == '__main__':
    # Create default favicon if missing
    if not os.path.exists('static/favicon.ico'):
        with open('static/favicon.ico', 'wb') as f:
            f.write(base64.b64decode(
                b'AAABAAEAEBAAAAAAAABoBQAAFgAAACgAAAAQAAAAIAAAAAEACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
            ))
    # Use the port provided by Render or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
