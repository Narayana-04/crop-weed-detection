import cv2
import numpy as np
from PIL import Image
import io
import base64

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL Image to OpenCV format
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    return image

def image_to_base64(image):
    """Convert image to base64 string for web display"""
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        return None
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    image_data = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def calculate_yield_impact(crop_count, weed_count, weed_proximity):
    """Calculate yield impact based on detection results"""
    if crop_count == 0:
        return 0, 0
    
    base_yield = crop_count * 10  # hypothetical base yield per plant
    yield_reduction = weed_proximity * 2  # reduction factor
    predicted_yield = max(0, base_yield - yield_reduction)
    
    # Calculate yield percentage (0-100%)
    yield_percentage = (predicted_yield / base_yield) * 100 if base_yield > 0 else 0
    
    return predicted_yield, yield_percentage

def create_analysis_report(crop_count, weed_count, weed_proximity, predicted_yield, yield_percentage):
    """Create a comprehensive analysis report"""
    return {
        'total_crops': crop_count,
        'total_weeds': weed_count,
        'crop_to_weed_ratio': crop_count / weed_count if weed_count > 0 else float('inf'),
        'weed_proximity_score': round(weed_proximity, 2),
        'predicted_yield': round(predicted_yield, 2),
        'yield_percentage': round(yield_percentage, 2),
        'yield_impact': f"{max(0, 100 - yield_percentage):.1f}% reduction"
    }