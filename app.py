"""
Optical Palm Authentication System with:
- Multi-Scale CNN Feature Extraction
- Gabor Wavelet Texture Analysis
- CLAHE Contrast Enhancement
- KD-Tree Matching
"""

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.neighbors import KDTree
from PIL import Image
import io

app = Flask(__name__)

# *********************
# Optical Preprocessor
# *********************

class OpticalProcessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.gabor_filters = self._create_gabor_bank()
        
    def _create_gabor_bank(self):
        filters = []
        for theta in np.arange(0, np.pi, np.pi/8):
            kern = cv2.getGaborKernel((21,21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters.append(kern)
        return filters
    
    def process(self, image):
        # Contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[...,0] = self.clahe.apply(lab[...,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Gabor texture features
        texture = [cv2.filter2D(enhanced, -1, k).mean() for k in self.gabor_filters]
        
        return enhanced, texture

# *********************
# Lightweight CNN Model
# *********************

class PalmCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)

# *********************
# Feature Database
# *********************

class PalmDatabase:
    def __init__(self):
        self.features = []
        self.ids = []
        self.tree = None
        
    def add(self, features, user_id):
        self.features.append(features)
        self.ids.append(user_id)
        self.tree = KDTree(np.array(self.features))
        
    def query(self, features, k=5):
        if self.tree is None:
            return []
        dist, idx = self.tree.query([features], k=k)
        return [(self.ids[i], d) for i,d in zip(idx[0], dist[0])]

# *********************
# Flask Endpoints
# *********************

processor = OpticalProcessor()
model = PalmCNN().eval()
database = PalmDatabase()

@app.route('/enroll', methods=['POST'])
def enroll():
    file = request.files['image'].read()
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    
    # Optical processing
    enhanced, texture = processor.process(img)
    
    # Feature extraction
    with torch.no_grad():
        tensor = torch.tensor(enhanced).permute(2,0,1).float()/255.0
        features = model(tensor.unsqueeze(0)).numpy().flatten()
    
    # Store in database
    user_id = str(hash(features.tobytes()))
    database.add(features, user_id)
    
    return jsonify({
        "status": "Enrolled",
        "user_id": user_id,
        "features": features.tolist()
    })

@app.route('/authenticate', methods=['POST'])
def authenticate():
    file = request.files['image'].read()
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    
    # Feature extraction
    enhanced, _ = processor.process(img)
    with torch.no_grad():
        tensor = torch.tensor(enhanced).permute(2,0,1).float()/255.0
        features = model(tensor.unsqueeze(0)).numpy().flatten()
    
    # Database query
    matches = database.query(features)
    
    if matches and matches[0][1] < 0.1:  # Threshold
        return jsonify({
            "access": "Granted",
            "user_id": matches[0][0],
            "confidence": float(1 - matches[0][1])
        })
    else:
        return jsonify({"access": "Denied"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
