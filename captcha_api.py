#!/usr/bin/env python3
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

class SimpleCaptchaSolver:
    def preprocess(self, img):
        return cv2.resize(img, (64, 64))
    
    def extract_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        return np.append(hist, edge_ratio)
    
    def compare(self, feat1, feat2):
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def solve(self, target, options):
        target = self.preprocess(target)
        options = [self.preprocess(opt) for opt in options]
        
        target_feat = self.extract_features(target)
        
        scores = []
        for opt in options:
            opt_feat = self.extract_features(opt)
            score = self.compare(target_feat, opt_feat)
            scores.append(float(score))
        
        best_idx = int(np.argmax(scores))
        
        return {
            'answer': best_idx + 1,
            'confidence': scores[best_idx],
            'scores': scores
        }

solver = SimpleCaptchaSolver()

def b64_to_cv2(b64_str):
    img_data = base64.b64decode(b64_str.split(',')[-1])
    img_array = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'CAPTCHA Solver API',
        'version': '2.0',
        'status': 'online',
        'endpoints': {
            '/health': 'GET - Health check',
            '/solve': 'POST - Solve CAPTCHA'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'solver': 'SimpleCaptchaSolver v2.0'})

@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        
        if not data or 'target' not in data or 'options' not in data:
            return jsonify({
                'status': 0,
                'error': 'Missing required fields: target, options'
            }), 400
        
        target = b64_to_cv2(data['target'])
        options = [b64_to_cv2(opt) for opt in data['options']]
        
        result = solver.solve(target, options)
        
        return jsonify({'status': 1, 'result': result})
    
    except Exception as e:
        return jsonify({'status': 0, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print("="*60)
    print("🚀 CAPTCHA Solver API - Production Mode")
    print(f"🌐 Port: {port}")
    print("="*60)
    app.run(host='0.0.0.0', port=port, debug=False)
