<!-- markdownlint-disable MD033 MD041 -->

# 🚨 FallSync - Real-time Fall Detection System

**FallSync** is an AI-powered real-time fall detection system that uses accelerometer data and deep learning to detect falls with high accuracy. It's designed for elderly care, sports safety, and workplace accident prevention.

## ✨ Features

- 🤖 **Deep Learning CNN Model** - Convolutional Neural Network trained on accelerometer data
- 📱 **Real-time Predictions** - Process accelerometer data in milliseconds
- 🌐 **Web Dashboard** - Interactive interface for testing and monitoring
- 📊 **REST API** - Easy integration with IoT devices and mobile apps
- 📈 **Visualization** - Real-time charts and acceleration magnitude graphs
- 🎯 **High Accuracy** - Optimized precision and recall metrics
- 📦 **Mobile Ready** - TensorFlow Lite model for on-device inference
- 🔄 **Batch Processing** - Handle multiple predictions simultaneously

## 🏗️ Architecture

```
FallSync/
├── fall_detection_model.py    # Core ML model
├── app.py                      # Flask backend API
├── train_model.py              # Training script
├── static/
│   ├── index.html             # Web dashboard
│   ├── app.js                 # Frontend logic
│   └── styles.css             # Dashboard styling
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fallSync.git
cd fallSync
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (optional - use pre-trained if available)
```bash
python train_model.py
```

This will generate:
- `fall_detection_model.h5` - Keras model
- `scaler.pkl` - Feature scaler
- `fall_detection_model.tflite` - Mobile model
- `model_metrics.json` - Training metrics

4. **Start the server**
```bash
python app.py
```

The application will start at `http://localhost:5000`

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns model status and connection info.

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "accelerometer_data": [
    [x1, y1, z1],
    [x2, y2, z2],
    ...  (60 samples minimum)
  ]
}
```

**Response:**
```json
{
  "is_fall": true,
  "fall_probability": 0.92,
  "no_fall_probability": 0.08,
  "confidence": 0.92,
  "max_acceleration": 8.45,
  "mean_acceleration": 2.34,
  "peak_jerk": 3.21,
  "alert": true,
  "timestamp": "2024-06-12T10:30:45.123456"
}
```

### Batch Predictions
```bash
POST /batch-predict
Content-Type: application/json

{
  "sequences": [
    [[x1, y1, z1], ...],
    [[x1, y1, z1], ...],
    ...
  ]
}
```

### Model Information
```bash
GET /model-info
```

## 🎯 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Window Size** | 60 samples (3 seconds at 20Hz) |
| **Sampling Rate** | 20 Hz |
| **Input Features** | 3 (X, Y, Z acceleration) |
| **Output Classes** | 2 (Fall, No-Fall) |
| **Architecture** | CNN with 3 convolutional blocks |
| **Input Shape** | (batch_size, 60, 3) |
| **Confidence Threshold** | 0.8 |

## 📊 Model Architecture

```
Input: (60, 3)
  ↓
Conv1D(32) + BatchNorm + MaxPool(2)
  ↓
Conv1D(64) + BatchNorm + MaxPool(2)
  ↓
Conv1D(128) + BatchNorm + GlobalAvgPool
  ↓
Dense(128, relu) + Dropout(0.5)
  ↓
Dense(64, relu) + Dropout(0.3)
  ↓
Dense(2, softmax)
  ↓
Output: [prob_no_fall, prob_fall]
```

## 💾 Data Format

Accelerometer data should be provided as a JSON array of `[x, y, z]` acceleration values:

```json
[
  [0.1, 0.2, -9.8],
  [0.15, 0.25, -9.75],
  [0.12, 0.22, -9.78],
  ...  // minimum 60 samples
]
```

**Units:** m/s² (meters per second squared)

## 🔧 Configuration

Set environment variables to customize the application:

```bash
# Model paths
export MODEL_PATH="fall_detection_model.h5"
export SCALER_PATH="scaler.pkl"

# Server
export PORT=5000
export DEBUG=False

# Prediction
export CONFIDENCE_THRESHOLD=0.8
```

## 🧪 Testing

### Test with Demo Data
1. Open `http://localhost:5000` in your browser
2. Click "🎯 Load Demo Data" button
3. Click "📊 Predict Fall" to see results

### Test via API
```bash
# Get demo data from the web interface
# Then use curl to make API requests

curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @data.json
```

## 📈 Performance Metrics

After training, view metrics in `model_metrics.json`:

- **Accuracy** - Overall correctness of predictions
- **Precision** - True positives / (true positives + false positives)
- **Recall** - True positives / (true positives + false negatives)
- **Loss** - Model optimization loss

## 🔐 Production Deployment

### Docker
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t fallsync .
docker run -p 5000:5000 fallsync
```

### Cloud Deployment
Deploy on cloud platforms:
- **Heroku** - `git push heroku main`
- **Google Cloud** - Cloud Run
- **AWS** - Lambda + API Gateway
- **Azure** - App Service

## 📱 Mobile Integration

Use the TensorFlow Lite model (`fall_detection_model.tflite`) for on-device inference:

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="fall_detection_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input
interpreter.set_tensor(input_details[0]['index'], input_data)

# Get prediction
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## ⚠️ Disclaimer

FallSync is designed as a supplementary monitoring tool and should not be used as the sole means of fall detection in critical care environments. Always consult healthcare professionals for proper safety protocols.

## 🔗 Links

- [Documentation](https://github.com/karuna19ww/fallSync)
- [Issues](https://github.com/karuna19ww/fallSync/issues)
- [Discussions](https://github.com/karuna19ww/fallSync/discussions)

## 👥 Authors

**Karuna** - Initial implementation and ML model design

## 🙏 Acknowledgments

- TensorFlow & Keras for deep learning framework
- Flask for web framework
- Chart.js for visualization
- Open source community

---

**Made with ❤️ for safer communities**

Last updated: June 2024