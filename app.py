"""
FallSync - Real-time Fall Detection Application
Flask backend for serving fall detection predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from datetime import datetime
import logging
from fall_detection_model import FallDetectionModel

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
model_initialized = False

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'fall_detection_model.h5')
SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.pkl')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.8))


def initialize_model():
    """Initialize the fall detection model"""
    global model, model_initialized
    
    try:
        model = FallDetectionModel()
        
        # Try to load existing model
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model.load(MODEL_PATH, SCALER_PATH)
            model_initialized = True
            logger.info("Model loaded successfully from disk")
        else:
            logger.warning("Model files not found. Model not initialized.")
            logger.info("Train the model first using: python fall_detection_model.py")
            model_initialized = False
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        model_initialized = False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': model_initialized,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict_fall():
    """
    Predict fall event from accelerometer data
    
    Expected JSON:
    {
        "accelerometer_data": [
            [x1, y1, z1],
            [x2, y2, z2],
            ...
        ]
    }
    
    Returns:
    {
        "is_fall": boolean,
        "fall_probability": float,
        "confidence": float,
        "max_acceleration": float,
        "mean_acceleration": float,
        "peak_jerk": float,
        "timestamp": string
    }
    """
    if not model_initialized:
        return jsonify({
            'error': 'Model not initialized',
            'message': 'Please train the model first'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'accelerometer_data' not in data:
            return jsonify({
                'error': 'Missing accelerometer_data'
            }), 400
        
        acceleration_sequence = data['accelerometer_data']
        
        # Validate data
        if not isinstance(acceleration_sequence, list):
            return jsonify({
                'error': 'accelerometer_data must be a list'
            }), 400
        
        if len(acceleration_sequence) < 60:
            return jsonify({
                'error': f'Insufficient data. Expected at least 60 samples, got {len(acceleration_sequence)}'
            }), 400
        
        # Make prediction
        prediction = model.predict_single(acceleration_sequence)
        
        # Add metadata
        prediction['timestamp'] = datetime.now().isoformat()
        prediction['alert'] = prediction['is_fall'] and prediction['confidence'] > CONFIDENCE_THRESHOLD
        
        return jsonify(prediction), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple acceleration sequences
    
    Expected JSON:
    {
        "sequences": [
            [[x1, y1, z1], ...],
            [[x1, y1, z1], ...],
            ...
        ]
    }
    """
    if not model_initialized:
        return jsonify({
            'error': 'Model not initialized'
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'sequences' not in data:
            return jsonify({
                'error': 'Missing sequences'
            }), 400
        
        sequences = data['sequences']
        predictions = []
        
        for seq in sequences:
            try:
                prediction = model.predict_single(seq)
                prediction['timestamp'] = datetime.now().isoformat()
                prediction['alert'] = prediction['is_fall'] and prediction['confidence'] > CONFIDENCE_THRESHOLD
                predictions.append(prediction)
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': 'FallSync CNN',
        'model_type': 'Convolutional Neural Network',
        'window_size': 60,
        'sampling_rate_hz': 20,
        'window_duration_seconds': 3,
        'input_features': 3,
        'output_classes': 2,
        'model_initialized': model_initialized,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    if not model_initialized or model is None:
        return jsonify({
            'error': 'Model not initialized'
        }), 503
    
    return jsonify({
        'status': 'Model statistics',
        'model_ready': True,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Initialize model on startup
    initialize_model()
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )