import os
import cv2
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ---------------------------
# Model Path Handling
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model paths
primary_model_path = os.path.join(BASE_DIR, "models", "trained_models", "crop_weed_model.pt")
fallback_model_path = os.path.join(BASE_DIR, "training", "runs", "detect", "desktop_crop_weed2", "weights", "best.pt")

if os.path.exists(primary_model_path):
    model_path = primary_model_path
elif os.path.exists(fallback_model_path):
    model_path = fallback_model_path
else:
    model_path = None

model = YOLO(model_path) if model_path else None

# ---------------------------
# Utilities
# ---------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def image_to_base64(image):
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

def calculate_yield_impact(crop_count, weed_count, weed_proximity):
    if crop_count == 0:
        return 0, 0
    base_yield = crop_count * 10
    yield_reduction = weed_proximity * 2
    predicted_yield = max(0, base_yield - yield_reduction)
    yield_percentage = (predicted_yield / base_yield) * 100 if base_yield > 0 else 0
    return predicted_yield, yield_percentage

def analyze_weed_impact(detection_results):
    crops, weeds = [], []
    for box in detection_results.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy()
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        if class_id == 0:  # crop
            crops.append({'bbox': bbox, 'confidence': confidence, 'center': center})
        elif class_id == 1:  # weed
            weeds.append({'bbox': bbox, 'confidence': confidence, 'center': center})
    return crops, weeds

def calculate_weed_proximity(crops, weeds):
    weed_proximity_score = 0
    proximity_details = []
    for weed in weeds:
        min_distance = float('inf')
        closest_crop = None
        for crop in crops:
            distance = np.linalg.norm(np.array(weed['center']) - np.array(crop['center']))
            if distance < min_distance:
                min_distance = distance
                closest_crop = crop
        if min_distance < float('inf'):
            impact = 1 / (min_distance + 1)
            weed_proximity_score += impact
            proximity_details.append({
                'weed_id': len(proximity_details) + 1,
                'distance': min_distance,
                'impact': impact,
                'closest_crop_confidence': closest_crop['confidence'] if closest_crop else 0
            })
    return weed_proximity_score, proximity_details

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'})

    try:
        image = Image.open(file.stream)
        image_cv = preprocess_image(image)
        original_image = image_cv.copy()

        if model is None:
            return jsonify({'error': 'Model not found. Please train the model first.'})

        results = model(image_cv)
        result = results[0]

        crops, weeds = analyze_weed_impact(result)
        crop_count, weed_count = len(crops), len(weeds)
        weed_proximity_score, proximity_details = calculate_weed_proximity(crops, weeds)
        predicted_yield, yield_percentage = calculate_yield_impact(crop_count, weed_count, weed_proximity_score)

        annotated_image = result.plot()
        img_str = image_to_base64(annotated_image)
        original_img_str = image_to_base64(original_image)

        analysis_report = {
            'total_crops': crop_count,
            'total_weeds': weed_count,
            'crop_to_weed_ratio': crop_count / weed_count if weed_count > 0 else float('inf'),
            'weed_proximity_score': round(weed_proximity_score, 2),
            'predicted_yield': round(predicted_yield, 2),
            'yield_percentage': round(yield_percentage, 2),
            'yield_impact': f"{max(0, 100 - yield_percentage):.1f}% reduction",
            'proximity_details': proximity_details[:5]
        }

        return jsonify({
            'success': True,
            'crop_count': crop_count,
            'weed_count': weed_count,
            'weed_proximity': round(weed_proximity_score, 2),
            'predicted_yield': round(predicted_yield, 2),
            'yield_percentage': round(yield_percentage, 2),
            'analysis_report': analysis_report,
            'annotated_image': f"data:image/jpeg;base64,{img_str}",
            'original_image': f"data:image/jpeg;base64,{original_img_str}",
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/model-status')
def model_status():
    status = {
        'model_loaded': model is not None,
        'model_path': model_path if model_path else "Not found",
        'model_exists': os.path.exists(model_path) if model_path else False
    }
    return jsonify(status)

# ---------------------------
# Run Server
# ---------------------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    print("Crop & Weed Detection Flask App")
    print("=" * 40)
    print(f"Model status: {'Loaded' if model else 'Not found'}")
    if model:
        print(f"Model path: {model_path}")
    print("Server starting on http://localhost:5000")
    print("Press Ctrl+C to stop the server")

    app.run(debug=True, host='0.0.0.0', port=5000)
